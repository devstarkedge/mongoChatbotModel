from django.urls import path, re_path
from . import views
from . import api_views
from django.urls import path
from django.http import JsonResponse
from django.contrib import admin


urlpatterns = [
    # UI Web Interface
    path("", views.index, name="index"),
    path('health', health_check),

    # API endpoints supporting /ask, /api/ask, /api/v1/ask with or without trailing slash
    re_path(r"^(?:api/)?(?:v1/)?ask/?$", api_views.AskView.as_view(), name="ask"),
    re_path(r"^(?:api/)?(?:v1/)?search/?$", api_views.SearchView.as_view(), name="search"),
    re_path(r"^(?:api/)?(?:v1/)?health/?$", api_views.HealthView.as_view(), name="health"),
    re_path(r"^(?:api/)?(?:v1/)?ingest/?$", api_views.IngestView.as_view(), name="ingest"),
    re_path(r"^(?:api/)?(?:v1/)?chats/?$", api_views.ChatListView.as_view(), name="chats_list"),
    re_path(r"^(?:api/)?(?:v1/)?chats/(?P<session_id>[^/]+)/rename/?$", api_views.ChatRenameView.as_view(), name="rename_chat"),
    re_path(r"^(?:api/)?(?:v1/)?chats/(?P<session_id>[^/]+)/delete/?$", api_views.ChatDeleteView.as_view(), name="chat_delete"),
    re_path(r"^(?:api/)?(?:v1/)?chats/(?P<session_id>[^/]+)/?$", api_views.ChatDetailView.as_view(), name="chat_detail"),
    
]




