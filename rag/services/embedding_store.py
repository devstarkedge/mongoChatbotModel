"""
The embeddings collection in MongoDB: the single source of truth.

    {
      _id: "projects:65a...:0",            # collection:document_id:chunk  (deterministic)
      embedding: [0.01, ...],              # normalised vector
      text: "Collection: projects\nProject: ...",
      metadata: {collection, primary_id, chunk_index, chunk_count, field_names,
                 reference_ids, display_name, content_hash, model, indexed_at},
      relationships: [{field, ref_id, ref_collection, ref_name}]
    }

Text, metadata, relationships and the vector live in ONE document, so they can never drift apart.
The FAISS HNSW index (vector_index.py) is rebuilt from this collection.
"""
import logging
from datetime import datetime, timezone

import numpy as np
from django.conf import settings
from pymongo import ReplaceOne
from pymongo.errors import PyMongoError

from . import mongo_service
from .exceptions import MongoServiceError

logger = logging.getLogger(__name__)
NO_VECTOR = {"embedding": 0}


def collection():
    return mongo_service.get_embeddings_collection()


def ensure_indexes():
    c = collection()
    try:
        c.create_index("metadata.collection")
        c.create_index("relationships.ref_id")
        c.create_index([("text", "text")], name="text_search")   # powers keyword (hybrid) search
    except Exception as exc:
        logger.warning("Could not create all indexes: %s", exc)


def upsert_many(records: list, vectors: np.ndarray):
    """Insert or replace records together with their vectors."""
    now = datetime.now(timezone.utc)
    ops = []
    for record, vector in zip(records, vectors):
        doc = dict(record)
        doc["metadata"] = {**record["metadata"], "model": settings.EMBEDDING_MODEL_ID, "indexed_at": now}
        doc["embedding"] = [float(x) for x in vector]
        ops.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
    try:
        for start in range(0, len(ops), 500):
            collection().bulk_write(ops[start:start + 500], ordered=False)
    except PyMongoError as exc:
        logger.error("Saving embeddings failed: %s", type(exc).__name__)
        raise MongoServiceError("Could not save embeddings to MongoDB.") from exc


def existing_hashes(source_collection: str) -> dict:
    """{embedding_id: content_hash} for one source collection (used to skip unchanged documents)."""
    try:
        cursor = collection().find({"metadata.collection": source_collection}, {"metadata.content_hash": 1})
        return {d["_id"]: d.get("metadata", {}).get("content_hash") for d in cursor}
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def delete_ids(ids: list):
    try:
        for start in range(0, len(ids), 1000):
            collection().delete_many({"_id": {"$in": ids[start:start + 1000]}})
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def delete_collections(names) -> int:
    """Remove every embedding whose source collection is in `names`. Returns the number deleted."""
    names = [str(name) for name in (names or []) if str(name).strip()]
    if not names:
        return 0
    try:
        return int(collection().delete_many({"metadata.collection": {"$in": names}}).deleted_count)
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def count() -> int:
    try:
        return collection().count_documents({})
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def distinct_models() -> list:
    try:
        return [m for m in collection().distinct("metadata.model") if m]
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def get_many(ids: list) -> dict:
    """{id: document without the vector}"""
    if not ids:
        return {}
    try:
        return {d["_id"]: d for d in collection().find({"_id": {"$in": list(ids)}}, NO_VECTOR)}
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def find_referencing_batch(primary_ids: list, limit_per_id: int = 3, allowed_ids=None) -> dict:
    """Records (first chunk only) pointing AT any of the given document ids in a single batched query."""
    if not primary_ids:
        return {}
    try:
        query = {"relationships.ref_id": {"$in": primary_ids}, "metadata.chunk_index": 0}
        if allowed_ids is not None:
            if not allowed_ids:
                return {}
            query["_id"] = {"$in": list(allowed_ids)}
        cursor = collection().find(query, NO_VECTOR).limit(limit_per_id * len(primary_ids))
        grouped = {pid: [] for pid in primary_ids}
        for d in cursor:
            for rel in d.get("relationships", []):
                ref_id = rel.get("ref_id")
                if ref_id in grouped and len(grouped[ref_id]) < limit_per_id:
                    if d not in grouped[ref_id]:
                        grouped[ref_id].append(d)
        return grouped
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def find_referencing(primary_id: str, exclude_collection_ids=(), limit: int = 3, allowed_ids=None) -> list:
    """Records (first chunk only) whose relationships point AT the given document id."""
    try:
        query = {"relationships.ref_id": primary_id, "metadata.chunk_index": 0}
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            query["_id"] = {"$in": list(allowed_ids)}
        cursor = collection().find(query, NO_VECTOR).limit(limit + len(exclude_collection_ids))
        return [d for d in cursor if d["_id"] not in exclude_collection_ids][:limit]
    except PyMongoError as exc:
        raise MongoServiceError() from exc


def iter_vectors(batch_size: int = 2000):
    """Stream (ids, source_collections, vectors) for building the HNSW index."""
    try:
        cursor = collection().find({}, {"embedding": 1, "metadata.collection": 1}).batch_size(batch_size)
        ids, cols, vecs = [], [], []
        for doc in cursor:
            if not doc.get("embedding"):
                continue
            ids.append(doc["_id"])
            cols.append(doc["metadata"]["collection"])
            vecs.append(doc["embedding"])
            if len(ids) >= batch_size:
                yield ids, cols, np.asarray(vecs, dtype="float32")
                ids, cols, vecs = [], [], []
        if ids:
            yield ids, cols, np.asarray(vecs, dtype="float32")
    except PyMongoError as exc:
        raise MongoServiceError() from exc
