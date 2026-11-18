"""
Ingestion pipeline: MongoDB source records -> text -> relationships -> embeddings -> Qdrant

  for each collection:
    for each BATCH of documents:
       detect relationships + bulk-load referenced names      (relationships.py)
       convert every document to chunk records                (document_converter.py)
       skip chunks whose content hash is unchanged            (incremental)
       embed only the new/changed chunks in one batch         (embedding_service.py)
       bulk upsert vectors and payloads into Qdrant           (embedding_store.py)
    delete embeddings of documents that no longer exist
  Qdrant indexes each upserted vector

Running it again is cheap: only new/changed documents are embedded, and deleted ones are removed.
A background job runner lets POST /api/ingest/ start it without blocking the web server.
"""
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from django.conf import settings

from . import (document_converter, embedding_service, embedding_store, mongo_service,
               relationships, structured_query, vector_index, query_correction)
from .exceptions import (ChatbotError, EmptyDatabaseError, IngestionBusyError, IngestionError)

logger = logging.getLogger(__name__)

# Fields that link a document directly to its workspace (used to stamp
# metadata.workspace_ids on every embedding so requests can be workspace-scoped).
_WS_DIRECT_FIELDS = ("workspaceId", "workspace_id", "workspace", "lastActiveWorkspace")
# Indirect links resolved one hop through a parent collection that has direct workspace fields.
_WS_REF_CHAIN = {"boards": ("project", "projects"), "cards": ("board", "boards"),
                 "tasks": ("projectId", "projects")}


def _direct_workspace_ids(doc) -> list:
    ids = []
    for field in _WS_DIRECT_FIELDS:
        value = doc.get(field)
        if value is None or value == "" or isinstance(value, (dict, list)):
            continue
        sid = str(value)
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def _workspace_maps(db, available) -> dict:
    """{collection: {document_id: [workspace ids]}} for tagging embeddings at ingest time."""
    maps = {}
    for name in available:
        if name == "workspaces":          # a workspace belongs to itself
            maps[name] = {str(doc["_id"]): [str(doc["_id"])] for doc in db[name].find({}, {"_id": 1})}
            continue
        coll_map = {}
        projection = {"_id": 1, **{field: 1 for field in _WS_DIRECT_FIELDS}}
        for doc in db[name].find({}, projection):
            ws_ids = _direct_workspace_ids(doc)
            if ws_ids:
                coll_map[str(doc["_id"])] = ws_ids
        if coll_map:
            maps[name] = coll_map
    # users are tagged with every workspace they are a member of (via memberships docs)
    member_map = {}
    for coll in ("workspacememberships", "memberships"):
        if coll not in available:
            continue
        for doc in db[coll].find({}, {"user": 1, "userId": 1, "memberId": 1,
                                      "workspace": 1, "workspaceId": 1}):
            uid = doc.get("user") or doc.get("userId") or doc.get("memberId")
            ws_id = doc.get("workspace") or doc.get("workspaceId")
            if uid is None or ws_id in (None, ""):
                continue
            user_ws = member_map.setdefault(str(uid), [])
            if str(ws_id) not in user_ws:
                user_ws.append(str(ws_id))
    if member_map:
        users_map = maps.get("users", {})
        for uid, ws_ids in member_map.items():
            target = users_map.setdefault(uid, [])
            for ws_id in ws_ids:
                if ws_id not in target:
                    target.append(ws_id)
        maps["users"] = users_map
    # one indirect hop: child -> parent -> workspace (e.g. cards -> boards -> projects).
    # Direct fields always win: docs already tagged are left untouched.
    for child, (field, parent) in _WS_REF_CHAIN.items():
        parent_map = maps.get(parent)
        if not parent_map:
            continue
        child_map = maps.get(child, {})
        for doc in db[child].find({}, {"_id": 1, field: 1}):
            sid = str(doc["_id"])
            if sid in child_map:
                continue
            ref = doc.get(field)
            ws_ids = parent_map.get(str(ref)) if ref is not None else None
            if ws_ids:
                child_map[sid] = list(ws_ids)
        if child_map:
            maps[child] = child_map
    return maps


def _workspace_ids_for(name: str, doc, ws_maps: dict) -> list:
    ws_ids = _direct_workspace_ids(doc)
    if name == "users":                   # direct field + every membership workspace
        for ws_id in ws_maps.get("users", {}).get(str(doc.get("_id")), []):
            if ws_id not in ws_ids:
                ws_ids.append(ws_id)
        return ws_ids
    if ws_ids:
        return ws_ids
    if name == "workspaces":
        return [str(doc["_id"])]
    return list(ws_maps.get(name, {}).get(str(doc.get("_id")), []))


def run_ingestion(collections=None, full=False, max_docs=0, batch_size=None, log=print) -> dict:
    started = time.perf_counter()
    batch_size = batch_size or settings.BATCH_SIZE
    available = mongo_service.list_collections()
    log("Removing embeddings for blocked collections ...")
    blocked_removed = embedding_store.delete_collections(mongo_service.blocked_collections())
    if blocked_removed:
        log(f"  removed {blocked_removed} embeddings from blocked collections")
    if not available:
        raise EmptyDatabaseError("The database has no collections to index.")
    if collections:
        blocked = [c for c in collections if mongo_service.is_blocked_collection(c)]
        if blocked:
            raise IngestionError(f"Collection(s) blocked from the RAG index: {', '.join(blocked)}.")
        unknown = [c for c in collections if c not in available]
        if unknown:
            raise IngestionError(f"Unknown collection(s): {', '.join(unknown)}. Available: {', '.join(available)}")
        targets = list(collections)
    else:
        targets = available

    embedding_store.ensure_indexes()
    old_models = [m for m in embedding_store.distinct_models() if m != settings.EMBEDDING_MODEL_ID]
    if old_models:
        if set(targets) != set(available):
            raise IngestionError(f"Existing embeddings were made with {old_models}. Re-embed ALL collections "
                                 f"(run without --collections) after changing the embedding model.")
        log(f"Embedding model changed ({', '.join(old_models)} -> {settings.EMBEDDING_MODEL_ID}): re-embedding everything.")
        full = True

    resolver = relationships.RelationshipResolver(mongo_service.get_db(), available)
    log("Resolving workspace links ...")
    ws_maps = _workspace_maps(mongo_service.get_db(), available)
    tagged = sum(len(m) for m in ws_maps.values())
    log(f"  workspace links found for {tagged} documents across {len(ws_maps)} collections")
    log(f"Collections to index: {', '.join(targets)}  (batch size {batch_size}, "
        f"{'full re-embed' if full else 'incremental'})")
    summary = {"mode": "full" if full else "incremental", "collections": {},
               "blocked_removed": blocked_removed}

    for name in targets:
        stats = {"documents": 0, "chunks": 0, "embedded": 0, "unchanged": 0, "removed": 0}
        existing = embedding_store.existing_hashes(name)
        seen = set()
        for batch in mongo_service.iter_batches(name, batch_size, max_docs):
            resolver.prefetch(name, batch)
            pending, meta_updates = [], []
            for doc in batch:
                records = document_converter.to_records(name, doc, resolver)
                if records:
                    stats["documents"] += 1
                ws_ids = _workspace_ids_for(name, doc, ws_maps)
                for record in records:
                    stats["chunks"] += 1
                    seen.add(record["_id"])
                    if ws_ids:
                        record["metadata"]["workspace_ids"] = ws_ids
                    prior = existing.get(record["_id"])
                    if not full and prior and prior.get("hash") == record["metadata"]["content_hash"]:
                        if (prior.get("workspace_ids") or []) != (ws_ids or []):
                            meta_updates.append((record["_id"], ws_ids or []))
                        stats["unchanged"] += 1
                    else:
                        pending.append(record)
            if meta_updates:
                embedding_store.update_workspace_ids(meta_updates)
            if pending:
                vectors = embedding_service.embed_texts([r["text"] for r in pending])
                embedding_store.upsert_many(pending, vectors)
                stats["embedded"] += len(pending)
            log(f"  {name}: {stats['documents']} documents, {stats['embedded']} embedded so far")
        if not max_docs:   # only delete stale embeddings after a complete read of the collection
            stale = [i for i in existing if i not in seen]
            if stale:
                embedding_store.delete_ids(stale)
                stats["removed"] = len(stale)
        summary["collections"][name] = stats

    if embedding_store.count() == 0:
        raise EmptyDatabaseError("No documents with indexable content were found.")
    log("Checking Qdrant vector collection ...")
    index = vector_index.rebuild_index()
    query_correction.clear_vocabulary_cache()
    summary["total_vectors"] = len(index.ids)
    try:
        log("Saving the query catalog ...")
        structured_query.build_catalog(force=True, resolver=resolver)
    except Exception as exc:       # list/count questions rebuild it on demand
        logger.warning("Could not save the catalog: %s", exc)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary


# ------------------------------------------------------- background job runner
_job_lock = threading.Lock()
_current = None


def start_job(collections=None, full=False, max_docs=0) -> dict:
    global _current
    with _job_lock:
        if _current and _current["state"] == "running":
            raise IngestionBusyError()
        job = {"id": uuid.uuid4().hex[:8], "state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
               "finished_at": None, "log": [], "summary": None, "error": None}
        _current = job

    def log(message):
        job["log"] = (job["log"] + [message])[-50:]
        logger.info("[ingest %s] %s", job["id"], message)

    def worker():
        try:
            job["summary"] = run_ingestion(collections, full, max_docs, log=log)
            job["state"] = "finished"
        except ChatbotError as exc:
            job["state"], job["error"] = "failed", exc.message
        except Exception:
            logger.exception("Ingestion job crashed")
            job["state"], job["error"] = "failed", "Unexpected error. See the server log."
        job["finished_at"] = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=worker, daemon=True, name=f"ingest-{job['id']}").start()
    return dict(job)


def job_status() -> dict:
    return dict(_current) if _current else {"state": "idle"}
