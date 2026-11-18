"""Qdrant vector search adapter; Qdrant persists vectors and chunk payloads."""
import logging
from datetime import datetime, timezone

from django.conf import settings

from . import embedding_store
from .exceptions import VectorStoreError

MISSING = "The Qdrant search collection has not been populated yet. Run: python manage.py ingest"

logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self):
        self.ids = []
        self.model = ""
        self.dimension = 0
        self.built_at = ""

    def search(self, query_vector, k: int, collections=None, allowed_ids=None, filter_key=None) -> list:
        return embedding_store.vector_search(query_vector, k, collections, allowed_ids)


_index = None


def current_version() -> str:
    """Read the Qdrant revision marker so writes invalidate answer caches across workers."""
    try:
        return f"{settings.QDRANT_COLLECTION}:{embedding_store.revision()}"
    except Exception:
        return "unavailable"


def rebuild_index() -> VectorIndex:
    """Qdrant indexes points on write, so verify the collection and return its current status."""
    global _index
    embedding_store.ensure_collection()
    embedding_store.ensure_indexes()
    if embedding_store.count() == 0:
        raise VectorStoreError(MISSING)
    _index = VectorIndex()
    models = embedding_store.distinct_models()
    _index.model = models[0] if len(models) == 1 else settings.EMBEDDING_MODEL_ID
    _index.ids = range(embedding_store.count())
    _index.built_at = datetime.now(timezone.utc).isoformat()
    logger.info("Qdrant index ready: %d vectors", len(_index.ids))
    return _index


def get_index() -> VectorIndex:
    global _index
    if not embedding_store.ensure_collection():
        raise VectorStoreError(MISSING)
    embedding_store.ensure_indexes()
    if _index is None:
        _index = VectorIndex()
        models = embedding_store.distinct_models()
        _index.model = models[0] if len(models) == 1 else settings.EMBEDDING_MODEL_ID
        _index.ids = range(embedding_store.count())
    return _index


def reset_for_tests():
    global _index
    _index = None
