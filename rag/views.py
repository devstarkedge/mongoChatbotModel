from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .services.anonymous_session import get_chat_owner


@ensure_csrf_cookie
def index(request):
    get_chat_owner(request)
    return render(request, "rag/index.html")
