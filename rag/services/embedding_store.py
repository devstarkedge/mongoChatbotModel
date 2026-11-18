"""Qdrant storage for vectors and their searchable chunk payloads."""
import logging
import threading
import time
import uuid
from datetime import date, datetime, timezone

import numpy as np
from django.conf import settings

from .exceptions import ConfigError, VectorStoreError

logger = logging.getLogger(__name__)
_client = None
_lock = threading.RLock()
_COLLECTION = None


def _qdrant():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                try:
                    from qdrant_client import QdrantClient
                    url = (getattr(settings, "QDRANT_URL", "") or "").strip()
                    path = (getattr(settings, "QDRANT_PATH", "") or "").strip()
                    if path:
                        _client = QdrantClient(path=path)
                    elif url:
                        _client = QdrantClient(url=url, api_key=getattr(settings, "QDRANT_API_KEY", "") or None,
                                               timeout=getattr(settings, "QDRANT_TIMEOUT", 10))
                    else:
                        raise ConfigError("Set QDRANT_URL (or QDRANT_PATH for local development).")
                except ImportError as exc:
                    raise VectorStoreError("Qdrant client is missing. Install requirements.txt.") from exc
                except ConfigError:
                    raise
                except Exception as exc:
                    logger.exception("Could not connect to Qdrant")
                    raise VectorStoreError("Could not connect to Qdrant. Check QDRANT_URL and credentials.") from exc
    return _client


def _name():
    return getattr(settings, "QDRANT_COLLECTION", "rag_chunks")


def _point_id(embedding_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(embedding_id)))


def _revision_point_id():
    return _point_id(f"{_name()}:revision")


def _json_safe(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def ensure_collection(dimension=None):
    """Ensure the Qdrant collection exists and its vector size matches the embedding model."""
    from qdrant_client import models
    client = _qdrant()
    name = _name()
    try:
        if not client.collection_exists(name):
            if dimension is None:
                return False
            client.create_collection(collection_name=name, vectors_config=models.VectorParams(
                size=int(dimension), distance=models.Distance.COSINE))
            for field, schema in (("metadata.collection", models.PayloadSchemaType.KEYWORD),
                                  ("metadata.workspace_ids", models.PayloadSchemaType.KEYWORD),
                                  ("metadata.model", models.PayloadSchemaType.KEYWORD),
                                  ("metadata.primary_id", models.PayloadSchemaType.KEYWORD),
                                  ("embedding_id", models.PayloadSchemaType.KEYWORD),
                                  ("kind", models.PayloadSchemaType.KEYWORD),
                                  ("relationships[].ref_collection", models.PayloadSchemaType.KEYWORD),
                                  ("relationships[].ref_id", models.PayloadSchemaType.KEYWORD),
                                  ("text", models.TextIndexParams(
                                      type="text", tokenizer=models.TokenizerType.WORD,
                                      min_token_len=2, max_token_len=30, lowercase=True))):
                client.create_payload_index(collection_name=name, field_name=field, field_schema=schema,
                                            wait=True)
            return True
        info = client.get_collection(name)
        size = getattr(getattr(info.config.params, "vectors", None), "size", None)
        if dimension is not None and size is not None and int(size) != int(dimension):
            raise VectorStoreError(f"Qdrant collection '{name}' uses {size}-dimensional vectors; "
                                   f"the configured model creates {dimension}. Re-ingest with a new collection.")
        return True
    except VectorStoreError:
        raise
    except Exception as exc:
        logger.exception("Could not prepare Qdrant collection")
        raise VectorStoreError("Could not prepare the Qdrant collection.") from exc


_indexes_checked = False
_index_lock = threading.Lock()


def reset_indexes_checked():
    global _indexes_checked
    _indexes_checked = False


def ensure_indexes(force: bool = False):
    """Ensure all required payload indexes exist on the Qdrant collection (self-healing)."""
    global _indexes_checked
    if _indexes_checked and not force:
        return
    with _index_lock:
        if _indexes_checked and not force:
            return
        if not ensure_collection():
            return
        from qdrant_client import models
        client = _qdrant()
        name = _name()
        try:
            info = client.get_collection(name)
            existing_schemas = info.payload_schema or {}
        except Exception as exc:
            logger.warning("Could not inspect Qdrant payload schema for %s: %s", name, exc)
            existing_schemas = {}

        required_keywords = (
            "metadata.collection",
            "metadata.workspace_ids",
            "metadata.model",
            "metadata.primary_id",
            "embedding_id",
            "kind",
            "relationships[].ref_collection",
            "relationships[].ref_id",
        )
        for field in required_keywords:
            if field not in existing_schemas:
                try:
                    logger.info("Self-healing Qdrant index: creating index for %s", field)
                    client.create_payload_index(collection_name=name, field_name=field,
                                                field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
                except Exception as exc:
                    logger.warning("Could not ensure Qdrant payload index for %s (%s)", field, type(exc).__name__)

        if "metadata.chunk_index" not in existing_schemas:
            try:
                logger.info("Self-healing Qdrant index: creating index for metadata.chunk_index (integer)")
                client.create_payload_index(collection_name=name, field_name="metadata.chunk_index",
                                            field_schema=models.PayloadSchemaType.INTEGER, wait=True)
            except Exception as exc:
                logger.warning("Could not ensure Qdrant payload index for metadata.chunk_index (%s)", type(exc).__name__)

        if "text" not in existing_schemas:
            try:
                logger.info("Self-healing Qdrant index: creating text index")
                client.create_payload_index(collection_name=name, field_name="text",
                                            field_schema=models.TextIndexParams(
                                                type="text", tokenizer=models.TokenizerType.WORD,
                                                min_token_len=2, max_token_len=30, lowercase=True),
                                            wait=True)
            except Exception as exc:
                logger.warning("Could not ensure Qdrant text payload index: %s", exc)

        _indexes_checked = True


def count():
    if not ensure_collection():
        return 0
    try:
        from qdrant_client import models
        flt = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))])
        return int(_qdrant().count(collection_name=_name(), count_filter=flt, exact=True).count)
    except Exception as exc:
        raise VectorStoreError("Could not count Qdrant vectors.") from exc


def upsert_many(records: list, vectors: np.ndarray):
    from qdrant_client import models
    if not records:
        return
    ensure_collection(vectors.shape[1])
    points = []
    for record, vector in zip(records, vectors):
        payload = {key: _json_safe(value) for key, value in record.items() if key != "_id"}
        metadata = payload.setdefault("metadata", {})
        metadata["model"] = settings.EMBEDDING_MODEL_ID
        metadata["indexed_at"] = datetime.now(timezone.utc).isoformat()
        payload["embedding_id"] = record["_id"]
        payload["kind"] = "chunk"
        points.append(models.PointStruct(id=_point_id(record["_id"]), vector=vector.tolist(), payload=payload))
    try:
        batch_size = settings.QDRANT_UPSERT_BATCH_SIZE
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            for attempt in range(4):
                try:
                    _qdrant().upsert(collection_name=_name(), points=batch, wait=True)
                    break
                except Exception as exc:
                    # Upserts use deterministic point IDs, so retrying after an ambiguous timeout is idempotent.
                    retryable = {"ResponseHandlingException", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
                                 "ConnectError", "ReadError", "WriteError", "RemoteProtocolError"}
                    current = exc
                    is_retryable = False
                    while current is not None:
                        if type(current).__name__ in retryable:
                            is_retryable = True
                            break
                        current = current.__cause__ or current.__context__
                    if not is_retryable or attempt == 3:
                        raise
                    delay = 2 ** attempt
                    logger.warning("Transient Qdrant upsert failure; retrying batch %d/%d in %ds (%s)",
                                   start // batch_size + 1, (len(points) + batch_size - 1) // batch_size,
                                   delay, type(exc).__name__)
                    time.sleep(delay)
        bump_revision()
    except Exception as exc:
        logger.exception("Saving embeddings to Qdrant failed")
        raise VectorStoreError("Could not save embeddings to Qdrant.") from exc


def _scroll(filter_obj=None, with_vectors=False, batch_size=1000):
    if not ensure_collection():
        return
    offset = None
    from qdrant_client import models
    if filter_obj is None:
        filter_obj = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))])
    else:
        filter_obj = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="chunk")),
                                         *list(filter_obj.must or [])],
                                   should=filter_obj.should, must_not=filter_obj.must_not)
    while True:
        try:
            points, offset = _qdrant().scroll(collection_name=_name(), scroll_filter=filter_obj,
                                               limit=batch_size, offset=offset,
                                               with_payload=True, with_vectors=with_vectors)
        except Exception as exc:
            logger.exception("Qdrant scroll failed while reading collection '%s'", _name())
            raise VectorStoreError("Could not read points from Qdrant.") from exc
        for point in points:
            payload = dict(point.payload or {})
            payload.pop("embedding_id", None)
            payload["_id"] = (point.payload or {}).get("embedding_id", str(point.id))
            if with_vectors:
                payload["embedding"] = point.vector
            yield payload
        if offset is None:
            break


def existing_hashes(source_collection: str) -> dict:
    from qdrant_client import models
    flt = models.Filter(must=[models.FieldCondition(key="metadata.collection", match=models.MatchValue(value=source_collection))])
    return {d["_id"]: {"hash": d.get("metadata", {}).get("content_hash"),
                       "workspace_ids": d.get("metadata", {}).get("workspace_ids") or []}
            for d in _scroll(flt)}


def distinct_models() -> list:
    from qdrant_client import models
    flt = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))])
    models_found = set()
    for d in _scroll(flt, batch_size=20):
        m = d.get("metadata", {}).get("model")
        if m:
            models_found.add(m)
            break                
    return sorted(models_found)


def update_workspace_ids(pairs: list):
    for embedding_id, workspace_ids in pairs:
        try:
            _qdrant().set_payload(collection_name=_name(), payload={"workspace_ids": workspace_ids or []},
                                  points=[_point_id(embedding_id)], key="metadata", wait=True)
        except Exception as exc:
            raise VectorStoreError("Could not update Qdrant workspace metadata.") from exc
    if pairs:
        bump_revision()


def delete_ids(ids: list):
    if not ids:
        return
    try:
        _qdrant().delete(collection_name=_name(), points_selector=[_point_id(i) for i in ids], wait=True)
        bump_revision()
    except Exception as exc:
        raise VectorStoreError("Could not delete Qdrant vectors.") from exc


def delete_collections(names) -> int:
    from qdrant_client import models
    names = [str(name) for name in names or [] if str(name).strip()]
    if not names or not ensure_collection():
        return 0        
    before = count()
    flt = models.Filter(must=[models.FieldCondition(key="metadata.collection", match=models.MatchAny(any=names))])
    try:
        _qdrant().delete(collection_name=_name(), points_selector=models.FilterSelector(filter=flt), wait=True)
        removed = before - count()
        if removed:
            bump_revision()
        return removed
    except Exception as exc:
        raise VectorStoreError("Could not remove blocked vectors from Qdrant.") from exc


def get_many(ids: list) -> dict:
    if not ids or not ensure_collection():
        return {}
    try:
        points = _qdrant().retrieve(collection_name=_name(), ids=[_point_id(i) for i in ids],
                                    with_payload=True, with_vectors=False)
        out = {}
        for point in points:
            payload = dict(point.payload or {})
            source_id = payload.pop("embedding_id", None)
            if source_id is not None:
                payload["_id"] = source_id
                out[source_id] = payload
        return out
    except Exception as exc:
        raise VectorStoreError("Could not retrieve chunks from Qdrant.") from exc


def find_ids_by_workspace(workspace_id: str) -> frozenset:
    from qdrant_client import models
    flt = models.Filter(must=[models.FieldCondition(key="metadata.workspace_ids", match=models.MatchValue(value=workspace_id))])
    return frozenset(d["_id"] for d in _scroll(flt))


def find_referencing_batch(primary_ids: list, limit_per_id: int = 3, allowed_ids=None) -> dict:
    from qdrant_client import models
    grouped = {str(pid): [] for pid in primary_ids}
    if not primary_ids:
        return grouped
    for primary_id in primary_ids:
        nested = models.NestedCondition(nested=models.Nested(key="relationships", filter=models.Filter(must=[
            models.FieldCondition(key="ref_id", match=models.MatchValue(value=str(primary_id)))
        ])))
        must = [models.FieldCondition(key="metadata.chunk_index", match=models.MatchValue(value=0)), nested]
        if allowed_ids is not None:
            if not allowed_ids:
                continue
            must.append(models.FieldCondition(key="embedding_id", match=models.MatchAny(any=list(allowed_ids))))
        for doc in _scroll(models.Filter(must=must)):
            grouped[str(primary_id)].append(doc)
            if len(grouped[str(primary_id)]) >= limit_per_id:
                break
    return grouped


def find_referencing(primary_id: str, exclude_collection_ids=(), limit: int = 3, allowed_ids=None) -> list:
    docs = find_referencing_batch([primary_id], limit_per_id=limit + len(exclude_collection_ids),
                                  allowed_ids=allowed_ids).get(str(primary_id), [])
    excluded = set(exclude_collection_ids)
    return [doc for doc in docs if doc["_id"] not in excluded][:limit]


def find_ids_for_user(user_id: str, workspace_id=None) -> frozenset:
    from qdrant_client import models
    user_id = str(user_id)
    common = [models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))]
    if workspace_id is not None:
        common.append(models.FieldCondition(key="metadata.workspace_ids", match=models.MatchValue(value=workspace_id)))

    try:
        # Fetch profile chunks through indexed scalar metadata.
        profile_filter = models.Filter(must=[*common,
            models.FieldCondition(key="metadata.collection", match=models.MatchValue(value="users")),
            models.FieldCondition(key="metadata.primary_id", match=models.MatchValue(value=user_id)),
        ])
        result = {doc["_id"] for doc in _scroll(profile_filter)}

        # Let Qdrant narrow candidates by array fields, then verify both values occur
        # on the same relationship entry to avoid cross-matching different refs.
        relationship_filter = models.Filter(must=[*common,
            models.FieldCondition(key="relationships[].ref_collection", match=models.MatchValue(value="users")),
            models.FieldCondition(key="relationships[].ref_id", match=models.MatchValue(value=user_id)),
        ])
        for doc in _scroll(relationship_filter):
            if any(rel.get("ref_collection") == "users" and str(rel.get("ref_id")) == user_id
                   for rel in doc.get("relationships", [])):
                result.add(doc["_id"])
        return frozenset(result)
    except Exception:
        logger.exception("Filtered Qdrant access lookup failed; falling back to payload scan")

    # Compatibility fallback for Qdrant server versions that do not support the
    # optimized array-field filter. Keep the same authorization check in Python.
    result = set()
    for doc in _scroll():
        metadata = doc.get("metadata", {})
        if workspace_id is not None and workspace_id not in (metadata.get("workspace_ids") or []):
            continue
        is_profile = metadata.get("collection") == "users" and metadata.get("primary_id") == user_id
        is_referenced = any(rel.get("ref_collection") == "users" and str(rel.get("ref_id")) == user_id
                            for rel in doc.get("relationships", []))
        if is_profile or is_referenced:
            result.add(doc["_id"])
    return frozenset(result)


def distinct_field_names(source_collection: str) -> set:
    from qdrant_client import models
    flt = models.Filter(must=[models.FieldCondition(key="metadata.collection", match=models.MatchValue(value=source_collection))])
    return {field for d in _scroll(flt) for field in d.get("metadata", {}).get("field_names", [])}


def all_texts():
    return (d.get("text", "") for d in _scroll())


def search_text(query: str, vector, limit: int, collections=None, allowed_ids=None) -> list:
    from qdrant_client import models
    must = [models.FieldCondition(key="kind", match=models.MatchValue(value="chunk")),
            models.FieldCondition(key="text", match=models.MatchText(text=query))]
    if collections:
        must.append(models.FieldCondition(key="metadata.collection", match=models.MatchAny(any=collections)))
    if allowed_ids is not None:
        # IDs are payload values because Qdrant point IDs must be UUIDs or unsigned integers.
        if not allowed_ids:
            return []
        must.append(models.FieldCondition(key="embedding_id", match=models.MatchAny(any=list(allowed_ids))))
    filt = models.Filter(must=must)
    try:
        result = _qdrant().query_points(collection_name=_name(), query=np.asarray(vector, dtype="float32").tolist(),
                                        query_filter=filt, limit=limit, with_payload=True).points
    except Exception as exc:
        raise VectorStoreError("Qdrant text search failed. Check the collection and embedding dimensions.") from exc
    return [((p.payload or {}).get("embedding_id"), float(p.score)) for p in result
            if (p.payload or {}).get("embedding_id")]


def vector_search(vector, limit: int, collections=None, allowed_ids=None) -> list:
    from qdrant_client import models
    if not ensure_collection():
        return []
    must = [models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))]
    if collections:
        must.append(models.FieldCondition(key="metadata.collection", match=models.MatchAny(any=collections)))
    if allowed_ids is not None:
        if not allowed_ids:
            return []
        must.append(models.FieldCondition(key="embedding_id", match=models.MatchAny(any=list(allowed_ids))))
        filt = models.Filter(must=must)
    else:
        filt = models.Filter(must=must) if must else None
    try:
        points = _qdrant().query_points(collection_name=_name(), query=np.asarray(vector, dtype="float32").tolist(),
                                        query_filter=filt, limit=limit, with_payload=True).points
    except Exception as exc:
        raise VectorStoreError("Qdrant vector search failed. Check the collection and embedding dimensions.") from exc
    return [((p.payload or {}).get("embedding_id"), float(p.score)) for p in points
            if (p.payload or {}).get("embedding_id")]


def list_ids():
    return [d["_id"] for d in _scroll()]


def bump_revision():
    """Update a small Qdrant marker so other app workers invalidate cached answers after writes."""
    from qdrant_client import models
    ensure_collection()
    info = _qdrant().get_collection(_name())
    size = int(info.config.params.vectors.size)
    _qdrant().upsert(collection_name=_name(), points=[models.PointStruct(
        id=_revision_point_id(), vector=[1.0] + [0.0] * (size - 1),
        payload={"kind": "revision", "revision": datetime.now(timezone.utc).isoformat()})], wait=True)


def revision() -> str:
    if not ensure_collection():
        return "empty"
    try:
        point = _qdrant().retrieve(collection_name=_name(), ids=[_revision_point_id()],
                                   with_payload=True, with_vectors=False)
        if point:
            return str((point[0].payload or {}).get("revision", "unknown"))
        return str(count())
    except Exception as exc:
        raise VectorStoreError("Could not read the Qdrant collection revision.") from exc


def reset_for_tests():
    global _client
    _client = None
