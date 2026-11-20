"""Optional Mem0 Platform memory for cross-chat user preferences."""
import hashlib
import logging

from django.conf import settings

logger = logging.getLogger(__name__)
_client = None


def _get_client():
    global _client
    if not settings.MEM0_ENABLED or not settings.MEM0_API_KEY:
        return None
    if _client is None:
        from mem0 import MemoryClient
        _client = MemoryClient(api_key=settings.MEM0_API_KEY)
    return _client


def _user_id(identity, workspace_id):
    value = f"{identity}\0{workspace_id or ''}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def search(question, identity, workspace_id=None):
    try:
        client = _get_client()
        if client is None:
            return ""
        result = client.search(question, filters={"user_id": _user_id(identity, workspace_id)})
        memories = result.get("results", [])
        return "\n".join(str(item.get("memory", "")) for item in memories[:3] if item.get("memory"))[:2000]
    except Exception as exc:
        logger.warning("Mem0 memory search failed (%s).", type(exc).__name__)
        return ""


def add_question(question, identity, workspace_id=None):
    # shortcut: chat deletion does not remove cross-chat memories; add account-level forget controls before production.
    try:
        client = _get_client()
        if client is not None:
            client.add([{"role": "user", "content": question}],
                       user_id=_user_id(identity, workspace_id))
    except Exception as exc:
        logger.warning("Mem0 memory write failed (%s).", type(exc).__name__)


def delete_all(identity, workspace_id=None):
    try:
        client = _get_client()
        if client is None:
            return False
        client.delete_all(user_id=_user_id(identity, workspace_id))
        return True
    except Exception as exc:
        logger.warning("Mem0 memory deletion failed (%s).", type(exc).__name__)
        return False
