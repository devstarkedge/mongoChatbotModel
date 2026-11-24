"""Mem0 Platform memory for cross-chat and cross-session user preferences and context."""
import hashlib
import logging
import threading

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
    """Search Mem0 for relevant memories about the current user and topic."""
    try:
        client = _get_client()
        if client is None or not identity:
            return ""
        result = client.search(question, filters={"user_id": _user_id(identity, workspace_id)})
        memories = result.get("results", [])
        if not memories:
            return ""
        items = []
        for item in memories[:5]:
            text = str(item.get("memory", "")).strip()
            if text:
                items.append(f"- {text}")
        return "\n".join(items)[:2000]
    except Exception as exc:
        logger.warning("Mem0 memory search failed (%s).", type(exc).__name__)
        return ""


def add(question, answer, identity, workspace_id=None):
    """Store the user-assistant exchange in Mem0 for cross-session recall."""
    try:
        client = _get_client()
        if client is not None and identity:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": str(answer)[:1500]},
            ]
            client.add(messages, user_id=_user_id(identity, workspace_id))
    except Exception as exc:
        logger.warning("Mem0 memory write failed (%s).", type(exc).__name__)


def add_async(question, answer, identity, workspace_id=None):
    """Run Mem0 write in a background daemon thread so it never delays the response."""
    if not settings.MEM0_ENABLED or not settings.MEM0_API_KEY or not identity:
        return
    threading.Thread(
        target=add,
        args=(question, answer, identity, workspace_id),
        daemon=True,
    ).start()


def add_question(question, identity, workspace_id=None):
    """Backward-compatible helper to save memory for a user."""
    add_async(question, "", identity, workspace_id)


def delete_all(identity, workspace_id=None):
    try:
        client = _get_client()
        if client is None or not identity:
            return False
        client.delete_all(user_id=_user_id(identity, workspace_id))
        return True
    except Exception as exc:
        logger.warning("Mem0 memory deletion failed (%s).", type(exc).__name__)
        return False
