from collections import OrderedDict, deque
import threading
import time

from django.db import transaction
from django.db.models import Q

from ..anonymous import get_owner, owner_session_qs
from ..models import ChatMessage, ChatSession

MAX_TURNS = 20
MAX_SESSIONS = 5000
SESSION_TTL_SECONDS = 3600

_lock = threading.Lock()
_sessions = OrderedDict()


def _expire(now):
    while _sessions:
        session_id, item = next(iter(_sessions.items()))
        if now - item["updated"] <= SESSION_TTL_SECONDS:
            break
        _sessions.pop(session_id, None)


def can_access(session_id, owner_id=None):
    with _lock:
        item = _sessions.get(session_id)
        return item is None or item.get("owner_id") == owner_id


def get_state(session_id, owner_id=None):
    now = time.monotonic()
    with _lock:
        _expire(now)
        item = _sessions.get(session_id)
        if item is None or now - item["updated"] > SESSION_TTL_SECONDS:
            _sessions.pop(session_id, None)
            return {"turns": [], "offset": 0}
        if item.get("owner_id") != owner_id:
            return {"turns": [], "offset": 0}
        item["updated"] = now
        _sessions.move_to_end(session_id)
        return {"turns": list(item["turns"]), "offset": item["offset"]}


def add(session_id, question, answer, owner_id=None):
    now = time.monotonic()
    with _lock:
        _expire(now)
        item = _sessions.get(session_id)
        if item is not None and item.get("owner_id") != owner_id:
            raise PermissionError("Chat session belongs to another user.")
        if item is None:
            item = {"turns": deque(maxlen=MAX_TURNS), "offset": 0, "updated": now,
                    "title": question, "owner_id": owner_id}
            _sessions[session_id] = item
        if len(item["turns"]) == MAX_TURNS:
            item["offset"] += 1
        item["turns"].append({"question": question, "answer": answer})
        item["updated"] = now
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def replace(session_id, index, question, answer, owner_id=None):
    now = time.monotonic()
    with _lock:
        _expire(now)
        item = _sessions.get(session_id)
        if item is not None and item.get("owner_id") != owner_id:
            raise PermissionError("Chat session belongs to another user.")
        local_index = index - item["offset"] if item is not None else -1
        if item is None or not 0 <= local_index < len(item["turns"]):
            raise IndexError("The turn being edited no longer exists.")
        turns = list(item["turns"])
        turns[local_index] = {"question": question, "answer": answer}
        item["turns"] = deque(turns, maxlen=MAX_TURNS)
        item["updated"] = now
        _sessions.move_to_end(session_id)


def clear(session_id, owner_id=None):
    with _lock:
        item = _sessions.get(session_id)
        if item is not None and item.get("owner_id") == owner_id:
            _sessions.pop(session_id, None)


def rename(session_id, title, owner_id=None):
    with _lock:
        item = _sessions.get(session_id)
        if item is None or item.get("owner_id") != owner_id:
            return False
        item["title"] = title[:120]
        item["updated"] = time.monotonic()
        _sessions.move_to_end(session_id)
        return True


def list_sessions(request):
    qs = owner_session_qs(request, ChatSession.objects.all())
    return [
        {
            "session_id": str(session.id),
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": session.messages.count(),
        }
        for session in qs
    ]


def get_session(request, session_id):
    try:
        return owner_session_qs(request, ChatSession.objects.all()).filter(id=session_id).first()
    except (TypeError, ValueError):
        return None


def session_exists(session_id):
    return ChatSession.objects.filter(pk=session_id).exists()


def get_messages(session):
    return [
        {
            "role": message.role,
            "content": message.content,
            "sources": message.sources,
            "created_at": message.created_at.isoformat(),
        }
        for message in session.messages.order_by("created_at", "id")
    ]


def get_history_state(session):
    if session is None:
        return {"turns": [], "offset": 0}
    messages = list(session.messages.order_by("created_at", "id"))
    history = []
    for index in range(0, len(messages) - 1, 2):
        user_message, bot_message = messages[index:index + 2]
        if user_message.role == ChatMessage.ROLE_USER and bot_message.role == ChatMessage.ROLE_BOT:
            history.append({"question": user_message.content, "answer": bot_message.content})
    offset = max(len(history) - MAX_TURNS, 0)
    return {"turns": history[-MAX_TURNS:], "offset": offset}


def get_history(session):
    return get_history_state(session)["turns"]


def append_message(session, role, content, sources=None):
    return ChatMessage.objects.create(
        session=session, role=role, content=content, sources=sources or [],
    )


def create_session(request, title="New chat", session_id=None):
    user, anon_key = get_owner(request)
    return ChatSession.objects.create(
        pk=session_id,
        user=user,
        anon_key=anon_key,
        title=title,
    )


def save_exchange(request, session, question, answer, sources=None, session_id=None, edit_index=None):
    with transaction.atomic():
        if session is None:
            if edit_index is not None:
                raise ValueError("A chat turn cannot be edited without an existing session.")
            session = create_session(
                request,
                title=question[:200] or "New chat",
                session_id=session_id,
            )
        else:
            session = ChatSession.objects.select_for_update().get(pk=session.pk)
            if edit_index is not None:
                user_messages = list(
                    session.messages.filter(role=ChatMessage.ROLE_USER).order_by("created_at", "id")
                )
                if not 0 <= edit_index < len(user_messages):
                    raise ValueError("The chat turn being edited no longer exists.")
                target = user_messages[edit_index]
                session.messages.filter(
                    Q(created_at__gt=target.created_at)
                    | Q(created_at=target.created_at, id__gte=target.id)
                ).delete()
        append_message(session, ChatMessage.ROLE_USER, question)
        append_message(session, ChatMessage.ROLE_BOT, answer, sources=sources)
        return session


def rename_session(request, session_id, title):
    session = get_session(request, session_id)
    if session is None:
        return False
    session.title = title
    session.save(update_fields=["title", "updated_at"])
    return True


def delete_session(request, session_id):
    session = get_session(request, session_id)
    if session is None:
        return False
    session.delete()
    return True
