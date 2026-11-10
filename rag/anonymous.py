from uuid import uuid4


def get_owner(request):
    """Returns (user, anon_key). Exactly one is non-null."""
    user = getattr(request, "user", None)
    if user and user.is_authenticated:
        return user, ""

    anon_key = request.session.get("chat_owner_id")
    if not anon_key:
        anon_key = uuid4().hex
        request.session["chat_owner_id"] = anon_key
    return None, anon_key


def owner_session_qs(request, queryset):
    """Filter a ChatSession queryset to the current owner."""
    user, anon_key = get_owner(request)
    if user is not None:
        return queryset.filter(user=user)
    return queryset.filter(user__isnull=True, anon_key=anon_key)