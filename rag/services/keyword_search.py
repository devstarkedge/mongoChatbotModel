"""
Keyword half of hybrid search: MongoDB full-text ($text) search over the embedded text.
Finds exact words, names and IDs that pure semantic search can miss. If the text index is
unavailable the function returns [] and search simply falls back to vectors only.
"""
import logging

from . import embedding_service, embedding_store, mongo_service

logger = logging.getLogger(__name__)


def search(query: str, limit: int, collections=None, allowed_ids=None) -> list:
    """-> [(embedding_id, text_score)] best first."""
    try:
        blocked = sorted(mongo_service.blocked_collections())
        allowed = list(collections) if collections else None
        if allowed is None:
            allowed = [name for name in mongo_service.list_collections() if name not in blocked]
        if not allowed:
            return []
        allowed = [name for name in allowed if not mongo_service.is_blocked_collection(name)]
        return embedding_store.search_text(query, embedding_service.embed_query(query), limit,
                                           allowed, allowed_ids)
    except Exception as exc:
        logger.warning("Keyword search unavailable (%s); using vector search only.", type(exc).__name__)
        return []
