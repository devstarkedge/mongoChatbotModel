"""
FAISS HNSW search index with Roaring Bitmap-backed traversal selectors.

Files in vector_store/: index.faiss (vectors) + index_meta.json (row -> embedding _id)
FAISS row i <-> ids[i] <-> the MongoDB embedding document with that _id.
"""
import hashlib
import json
import logging
import threading
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from django.conf import settings

from . import embedding_store
from .exceptions import VectorStoreError

logger = logging.getLogger(__name__)
INDEX_FILE, META_FILE = "index.faiss", "index_meta.json"
BACKEND = "faiss-hnsw-roaring"
FILTER_CACHE_SIZE = 128
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 200
MISSING = "The search index has not been built yet. Run: python manage.py ingest"


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise VectorStoreError("FAISS is not installed. Run: pip install faiss-cpu") from exc
    return faiss


def _bitmap_type():
    try:
        from pyroaring import BitMap
    except ImportError as exc:
        raise VectorStoreError("pyroaring is not installed. Run: pip install pyroaring") from exc
    return BitMap


class VectorIndex:
    def __init__(self):
        self.index = None
        self.ids, self.collections = [], []
        self.model, self.dimension, self.built_at = "", 0, ""
        self._id_to_row = {}
        self._collection_bitmaps = {}
        self._filter_cache = OrderedDict()

    def _build_collection_bitmaps(self):
        BitMap = _bitmap_type()
        rows_by_collection = defaultdict(BitMap)
        for row, collection in enumerate(self.collections):
            rows_by_collection[collection].add(row)
        self._collection_bitmaps = dict(rows_by_collection)
        self._id_to_row = {doc_id: row for row, doc_id in enumerate(self.ids)}
        self._filter_cache.clear()

    # -------------------------------------------------------------- build/save/load
    def build_from_store(self):
        faiss = _faiss()
        ids, cols, blocks = [], [], []
        for batch_ids, batch_cols, matrix in embedding_store.iter_vectors():
            if blocks and matrix.shape[1] != blocks[0].shape[1]:
                raise VectorStoreError("Stored embeddings have different sizes. Run a full ingest: "
                                       "python manage.py ingest --full")
            ids.extend(batch_ids)
            cols.extend(batch_cols)
            blocks.append(matrix)
        if not ids:
            raise VectorStoreError(MISSING)
        matrix = np.ascontiguousarray(np.vstack(blocks), dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms != 0)
        self.dimension = int(matrix.shape[1])
        self.index = faiss.IndexHNSWFlat(self.dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = max(HNSW_EF_SEARCH, settings.TOP_K * 3)
        self.index.add(matrix)
        self.ids, self.collections = ids, cols
        self._build_collection_bitmaps()
        models = embedding_store.distinct_models()
        self.model = models[0] if len(models) == 1 else settings.EMBEDDING_MODEL_ID
        self.built_at = datetime.now(timezone.utc).isoformat()

    def save(self, directory):
        faiss = _faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tmp_index, tmp_meta = directory / (INDEX_FILE + ".tmp"), directory / (META_FILE + ".tmp")
        faiss.write_index(self.index, str(tmp_index))
        tmp_index.replace(directory / INDEX_FILE)
        tmp_meta.write_text(json.dumps({
            "backend": BACKEND, "ids": self.ids, "collections": self.collections, "model": self.model,
            "dimension": self.dimension, "built_at": self.built_at,
        }), encoding="utf-8")
        tmp_meta.replace(directory / META_FILE)

    def load(self, directory):
        faiss = _faiss()
        directory = Path(directory)
        try:
            meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
            if meta.get("backend") != BACKEND:
                raise VectorStoreError("The search index uses an old backend. Rebuild it with: "
                                       "python manage.py ingest")
            self.ids, self.collections = meta["ids"], meta["collections"]
            self.model, self.dimension = meta.get("model", ""), int(meta["dimension"])
            self.built_at = meta.get("built_at", "")
            self.index = faiss.read_index(str(directory / INDEX_FILE))
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("Failed to load vector index")
            raise VectorStoreError("The search index looks corrupted. Rebuild it with: "
                                   "python manage.py ingest") from exc
        if (not hasattr(self.index, "hnsw") or self.index.metric_type != faiss.METRIC_INNER_PRODUCT
                or self.index.ntotal != len(self.ids) or self.index.ntotal != len(self.collections)):
            raise VectorStoreError("The search index and its id list do not match. Rebuild it with: "
                                   "python manage.py ingest")
        self.index.hnsw.efSearch = max(HNSW_EF_SEARCH, settings.TOP_K * 3)
        self._build_collection_bitmaps()

    def _filter_selector(self, collections, allowed_ids, filter_key):
        if not collections and allowed_ids is None:
            return None, None
        faiss = _faiss()
        BitMap = _bitmap_type()
        selected_collections = tuple(sorted(set(collections or ())))
        if filter_key is None:
            digest = hashlib.sha256()
            if allowed_ids is not None:
                for doc_id in sorted(allowed_ids):
                    digest.update(doc_id.encode("utf-8"))
                    digest.update(b"\0")
            filter_key = digest.hexdigest()
        cache_key = (filter_key, selected_collections)
        cached = self._filter_cache.get(cache_key)
        if cached is not None:
            self._filter_cache.move_to_end(cache_key)
            return cached[0], cached[1]

        bitmap = None
        if allowed_ids is not None:
            bitmap = BitMap(
                self._id_to_row[doc_id] for doc_id in allowed_ids if doc_id in self._id_to_row
            )
        if collections:
            collection_bitmap = BitMap()
            for collection in selected_collections:
                collection_bitmap |= self._collection_bitmaps.get(collection, BitMap())
            bitmap = collection_bitmap if bitmap is None else bitmap & collection_bitmap

        mask = np.zeros((len(self.ids) + 7) // 8, dtype=np.uint8)
        if bitmap:
            rows = np.fromiter(bitmap, dtype=np.int64, count=len(bitmap))
            np.bitwise_or.at(mask, rows >> 3, (1 << (rows & 7)).astype(np.uint8))
        selector = faiss.IDSelectorBitmap(len(self.ids), faiss.swig_ptr(mask))
        self._filter_cache[cache_key] = (bitmap, (selector, mask))
        if len(self._filter_cache) > FILTER_CACHE_SIZE:
            self._filter_cache.popitem(last=False)
        return bitmap, (selector, mask)

    # --------------------------------------------------------------------- search
    def search(self, query_vector, k: int, collections=None, allowed_ids=None, filter_key=None) -> list:
        """Return cosine-ranked pairs with a bitmap selector applied during HNSW traversal."""
        if self.index is None or not self.ids or allowed_ids is not None and not allowed_ids:
            return []
        query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError("Embedding size mismatch. Run a full ingest: python manage.py ingest --full")
        bitmap, selector_parts = self._filter_selector(collections, allowed_ids, filter_key)
        if bitmap is not None and not bitmap:
            return []

        result_count = min(k, len(bitmap)) if bitmap is not None else min(k, len(self.ids))
        allowed_collections = set(collections) if collections else None

        def make_results(labels, distances):
            out = []
            for row, distance in zip(labels[0], distances[0]):
                row = int(row)
                if row < 0 or row >= len(self.ids):
                    continue
                if bitmap is not None and row not in bitmap:
                    continue
                if allowed_collections and self.collections[row] not in allowed_collections:
                    continue
                if allowed_ids is not None and self.ids[row] not in allowed_ids:
                    continue
                out.append((self.ids[row], float(distance)))
            return out

        if selector_parts is None:
            distances, labels = self.index.search(query, result_count)
            return make_results(labels, distances)

        ef_search = min(len(self.ids), max(HNSW_EF_SEARCH, result_count, settings.TOP_K * 3))
        while True:
            params = _faiss().SearchParametersHNSW()
            params.efSearch = ef_search
            params.sel = selector_parts[0]
            distances, labels = self.index.search(query, result_count, params=params)
            out = make_results(labels, distances)
            if len(out) == result_count or ef_search == len(self.ids):
                return out
            ef_search = min(len(self.ids), max(ef_search + 1, ef_search * 2))


# ------------------------------------------------------- shared instance (Django)
_index = None
_stamp = None
_lock = threading.RLock()


def _stamp_now():
    try:
        return (Path(settings.VECTOR_STORE_DIR) / META_FILE).stat().st_mtime_ns
    except FileNotFoundError:
        return None


def current_version() -> str:
    """Changes whenever a new index is written (used to invalidate the answer cache)."""
    return str(_stamp_now())


def rebuild_index() -> VectorIndex:
    """Rebuild the FAISS HNSW index from the embeddings collection, save it and use it immediately."""
    global _index, _stamp
    with _lock:
        fresh = VectorIndex()
        fresh.build_from_store()
        fresh.save(settings.VECTOR_STORE_DIR)
        _index, _stamp = fresh, _stamp_now()
        logger.info("FAISS HNSW index rebuilt: %d vectors, dim %d", len(fresh.ids), fresh.dimension)
        return fresh


def get_index() -> VectorIndex:
    """Return the loaded index. Reloads when another process wrote a newer one; rebuilds if missing."""
    global _index, _stamp
    with _lock:
        stamp = _stamp_now()
        if _index is not None and stamp is not None and stamp == _stamp:
            return _index
        if stamp is not None:
            fresh = VectorIndex()
            try:
                fresh.load(settings.VECTOR_STORE_DIR)
                _index, _stamp = fresh, stamp
                logger.info("FAISS HNSW index loaded: %d vectors", len(fresh.ids))
                return _index
            except VectorStoreError:
                logger.warning("Index files unusable, rebuilding from MongoDB")
        if embedding_store.count() == 0:
            raise VectorStoreError(MISSING)
        return rebuild_index()


def reset_for_tests():
    global _index, _stamp
    _index, _stamp = None, None
