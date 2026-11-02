from uuid import uuid4


def _owner_id(value):
    return f"guest:{value}"


def get_chat_owner(request):
    owner_id = request.session.get("chat_owner_id")
    if not owner_id:
        owner_id = uuid4().hex
        request.session["chat_owner_id"] = owner_id
    return _owner_id(owner_id)


async def aget_chat_owner(request):
    owner_id = await request.session.aget("chat_owner_id")
    if not owner_id:
        owner_id = uuid4().hex
        await request.session.aset("chat_owner_id", owner_id)
    return _owner_id(owner_id)
