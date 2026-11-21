from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .services.anonymous_session import get_chat_owner


@ensure_csrf_cookie
def index(request):
    get_chat_owner(request)
    if request.method == "HEAD":
        return HttpResponse()
    return render(request, "rag/index.html")
