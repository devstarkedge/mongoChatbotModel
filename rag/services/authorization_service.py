"""Resolve the public scope and legacy account scopes used by the retrieval service."""
from dataclasses import dataclass
import hashlib
import re

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import PyMongoError

from . import embedding_store, mongo_service
from .document_converter import pick_display_name
from .exceptions import AuthorizationError, ConfigError, MongoServiceError


@dataclass(frozen=True)
class AccessScope:
    user_id: str
    is_admin: bool
    embedding_ids: frozenset[str] | None
    cache_key: str
    chat_owner_id: str
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


def resolve_scope(user_id: str | None, email: str | None = None) -> AccessScope:
    """Return a no-data anonymous scope or a database-verified account scope."""
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
    workspace_id = user.get("lastActiveWorkspace")
    if workspace_id:
        sid = str(workspace_id)
        try:
            workspace = mongo_service.get_db()["workspaces"].find_one(
                {"_id": {"$in": [sid, ObjectId(sid) if len(sid) == 24 else sid]}}
            )
            if workspace:
                user_company = pick_display_name(workspace)
        except Exception:
            user_company = None
    if is_admin:
        return AccessScope(user_id, True, None, f"{user_id}:admin:all", f"{user_id}:account",
                           user_name=user_name, user_email=user_email, user_role=user_role,
                           user_company=user_company)

    query = {
        "$or": [
            {"metadata.collection": "users", "metadata.primary_id": str(user["_id"])},
            {"relationships": {"$elemMatch": {
                "ref_collection": "users", "ref_id": str(user["_id"]),
            }}},
        ],
    }
    try:
        ids = frozenset(doc["_id"] for doc in embedding_store.collection().find(query, {"_id": 1}))
    except (PyMongoError, ConfigError, MongoServiceError) as exc:
        raise MongoServiceError("Could not load account data permissions.") from exc

    # The digest keeps cache entries isolated when the user's indexed links change.
    scope_digest = hashlib.sha256("\0".join(sorted(ids)).encode("utf-8")).hexdigest()
    return AccessScope(
        user_id, False, ids, f"{user_id}:employee:{scope_digest}",
        f"{user_id}:account",
        user_name=user_name, user_email=user_email, user_role=user_role,
        user_company=user_company,
    )
