"""Resolve the public scope and legacy account scopes used by the retrieval service."""
from dataclasses import dataclass
import hashlib
import logging
import re

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import PyMongoError

from . import embedding_store, mongo_service
from .document_converter import pick_display_name
from .exceptions import AuthorizationError, ConfigError, InvalidRequestError, MongoServiceError, VectorStoreError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccessScope:
    user_id: str
    is_admin: bool
    embedding_ids: frozenset[str] | None
    cache_key: str
    chat_owner_id: str
    workspace_id: str | None = None
    user_name: str | None = None
    user_email: str | None = None
    user_role: str | None = None
    user_company: str | None = None


def _id_values(user_id: str) -> list:
    values = [user_id]
    if len(user_id) == 24:
        try:
            values.append(ObjectId(user_id))
        except (InvalidId, ValueError):
            pass
    if user_id.lstrip("-").isdigit():
        values.append(int(user_id))
    return values


def mentions_current_user(question: str, user_id: str, current_user_name: str | None = None) -> bool:
    """True when the question also refers to the signed-in account."""
    text = question or ""
    if current_user_name:
        name = " ".join(current_user_name.split())
        if name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
            return True
        words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]+", name) if len(word) >= 3]
        if words and any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE) for word in words):
            return True
    if user_id and re.search(rf"(?<!\w){re.escape(str(user_id))}(?!\w)", text, re.IGNORECASE):
        return True
    return bool(re.search(r"\b(?:me|my|mine|myself|i|i'm|i am)\b", text, re.IGNORECASE))


def mentions_another_user(question: str, user_id: str, current_user_name: str | None = None) -> bool:
    """Detect explicit references to another account without returning its data."""
    tokens = {token for token in re.findall(r"[\w.@+-]+", question) if len(token) >= 3}
    if not tokens:
        return False
    # Exclude tokens that match the current user's own name — they're referring to themselves.
    if current_user_name:
        tokens = {t for t in tokens if t.casefold() not in current_user_name.casefold().split()}
    fields = (
        "name", "fullName", "full_name", "displayName", "display_name",
        "firstName", "first_name", "lastName", "last_name", "username", "email",
    )
    token_pattern = "|".join(re.escape(token) for token in tokens)
    pattern = rf"(?<!\w)(?:{token_pattern})(?!\w)"
    try:
        users = mongo_service.get_db()["users"].find(
            {
                "_id": {"$nin": _id_values(user_id)},
                "$or": [{field: {"$regex": pattern, "$options": "i"}} for field in fields],
            },
            {"_id": 1},
        )
        return next(users, None) is not None
    except (PyMongoError, ConfigError, MongoServiceError) as exc:
        raise MongoServiceError("Could not verify the requested employee.") from exc


def _workspace_scoped_ids(workspace_id: str) -> frozenset:
    """Embedding ids tagged with the given workspace (used to scope admins)."""
    try:
        return embedding_store.find_ids_by_workspace(workspace_id)
    except (PyMongoError, ConfigError, MongoServiceError, VectorStoreError) as exc:
        raise MongoServiceError("Could not load workspace data permissions.") from exc


def _validate_workspace(db, workspace_id: str) -> str:
    """Reject malformed or unknown workspace ids. Returns the canonical string id."""
    workspace_id = str(workspace_id).strip()
    if not workspace_id or len(workspace_id) > 128 or any(ch.isspace() for ch in workspace_id):
        raise InvalidRequestError("'workspace_id' must be a valid workspace identifier.")
    try:
        workspace = db["workspaces"].find_one({"_id": {"$in": _id_values(workspace_id)}}, {"_id": 1})
    except (PyMongoError, ConfigError, MongoServiceError) as exc:
        raise MongoServiceError("Could not verify the workspace.") from exc
    if not workspace:
        raise InvalidRequestError("Workspace not found.")
    return str(workspace["_id"])


def _verify_workspace_membership(db, user, workspace_id: str) -> None:
    """Non-admins must be a member of the requested workspace (fail closed)."""
    if str(user.get("lastActiveWorkspace") or "") == workspace_id:
        return
    user_variants = _id_values(str(user["_id"]))
    ws_variants = _id_values(workspace_id)
    try:
        for collection_name in ("workspacememberships", "memberships"):
            for user_field in ("user", "userId", "memberId"):
                for ws_field in ("workspace", "workspaceId"):
                    doc = db[collection_name].find_one(
                        {user_field: {"$in": user_variants}, ws_field: {"$in": ws_variants}},
                        {"_id": 1})
                    if doc:
                        return
    except (PyMongoError, ConfigError, MongoServiceError) as exc:
        raise MongoServiceError("Could not verify workspace membership.") from exc
    raise AuthorizationError("You are not a member of this workspace.")


def resolve_scope(user_id: str | None, email: str | None = None,
                  workspace_id: str | None = None) -> AccessScope:
    """Return a no-data anonymous scope or a database-verified account scope.

    `workspace_id` (optional) restricts retrieval to one workspace. It must be a real
    workspace, and non-admins must be a member of it.
    """
    if isinstance(user_id, str) and user_id.startswith("guest:"):
        owner_id = user_id
        return AccessScope(owner_id, False, frozenset(), "public:anonymous", owner_id,
                           user_name=None, user_email=None, user_role=None, user_company=None)

    if not isinstance(user_id, str) or not user_id.strip():
        raise AuthorizationError()
    user_id = user_id.strip()

    try:
        user = mongo_service.get_db()["users"].find_one(
            {"_id": {"$in": _id_values(user_id)}},
            {"_id": 1, "role": 1, "email": 1, "name": 1, "fullName": 1, "full_name": 1,
             "displayName": 1, "display_name": 1, "firstName": 1, "first_name": 1,
             "lastName": 1, "last_name": 1, "username": 1, "lastActiveWorkspace": 1},
        )
    except (PyMongoError, ConfigError, MongoServiceError) as exc:
        raise MongoServiceError("Could not verify account permissions.") from exc
    if not user:
        raise AuthorizationError("The account could not be verified.")
    if email and str(user.get("email", "")).strip().casefold() != email.strip().casefold():
        raise AuthorizationError("The account email does not match the supplied user ID.")

    is_admin = str(user.get("role", "")).strip().casefold() == "admin"
    user_name = pick_display_name(user)
    if not user_name:
        first = user.get("firstName") or user.get("first_name")
        last = user.get("lastName") or user.get("last_name")
        if first or last:
            user_name = " ".join(p for p in [first, last] if p).strip()
        elif user.get("username"):
            user_name = user["username"]
        elif user.get("email"):
            user_name = user["email"]
    user_email = str(user.get("email", "")).strip() or None
    user_role = str(user.get("role", "")).strip() or None

    # Resolve the user's company from lastActiveWorkspace reference.
    user_company = None
    last_workspace = user.get("lastActiveWorkspace")
    if last_workspace:
        sid = str(last_workspace)
        try:
            workspace = mongo_service.get_db()["workspaces"].find_one(
                {"_id": {"$in": [sid, ObjectId(sid) if len(sid) == 24 else sid]}}
            )
            if workspace:
                user_company = pick_display_name(workspace)
        except Exception:
            user_company = None

    # Optional workspace scoping: validate the requested workspace and, for
    # non-admins, verify membership (or that it is their active workspace).
    if workspace_id is not None:
        workspace_id = _validate_workspace(mongo_service.get_db(), workspace_id)
        if not is_admin:
            _verify_workspace_membership(mongo_service.get_db(), user, workspace_id)

    if is_admin:
        if workspace_id is None:
            return AccessScope(user_id, True, None, f"{user_id}:admin:all", f"{user_id}:account",
                               user_name=user_name, user_email=user_email, user_role=user_role,
                               user_company=user_company)
        ids = _workspace_scoped_ids(workspace_id)
        return AccessScope(
            user_id, True, ids, f"{user_id}:admin:ws:{hashlib.sha256(workspace_id.encode('utf-8')).hexdigest()[:16]}",
            f"{user_id}:account", workspace_id=workspace_id,
            user_name=user_name, user_email=user_email, user_role=user_role,
            user_company=user_company,
        )

    try:
        ids = embedding_store.find_ids_for_user(str(user["_id"]), workspace_id)
    except VectorStoreError as exc:
        logger.exception("Qdrant access-scope lookup failed")
        raise MongoServiceError("Could not load account data permissions.") from exc
    except (PyMongoError, ConfigError, MongoServiceError) as exc:
        raise MongoServiceError("Could not load account data permissions.") from exc

    # The digest keeps cache entries isolated when the user's indexed links change.
    scope_digest = hashlib.sha256("\0".join(sorted(ids)).encode("utf-8")).hexdigest()
    cache_key = f"{user_id}:employee:{scope_digest}"
    if workspace_id is not None:
        cache_key += f":ws:{hashlib.sha256(workspace_id.encode('utf-8')).hexdigest()[:16]}"
    return AccessScope(
        user_id, False, ids, cache_key,
        f"{user_id}:account", workspace_id=workspace_id,
        user_name=user_name, user_email=user_email, user_role=user_role,
        user_company=user_company,
    )
