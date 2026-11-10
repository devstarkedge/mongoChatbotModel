from django.urls import path, re_path

from . import views
from . import api_views

urlpatterns = [
    # UI Web Interface
    path("", views.index, name="index"),

    # Direct root endpoints (/health/, /ask/, /search/, /chats/...)
    path("health/", api_views.HealthView.as_view(), name="root_health"),
    path("ask/", api_views.AskView.as_view(), name="root_ask"),
    path("search/", api_views.SearchView.as_view(), name="root_search"),
    path("ingest/", api_views.IngestView.as_view(), name="root_ingest"),
    path("chats/", api_views.ChatListView.as_view(), name="root_chats"),
    path("chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="root_chat_rename"),
    path("chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="root_chat_delete"),
    path("chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="root_chat_detail"),

    # Explicit /api/ and /api/v1/ endpoints
    path("api/v1/ask/", api_views.AskView.as_view(), name="api_ask"),
    path("api/v1/search/", api_views.SearchView.as_view(), name="api_search"),
    path("api/v1/health/", api_views.HealthView.as_view(), name="api_health"),
    path("api/v1/ingest/", api_views.IngestView.as_view(), name="api_ingest"),
    path("api/v1/chats/", api_views.ChatListView.as_view(), name="api_chats"),
    path("api/v1/chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="api_chat_rename"),
    path("api/v1/chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="api_chat_delete"),
    path("api/v1/chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="api_chat_detail"),
    path("api/ask/", api_views.AskView.as_view(), name="ask_api"),
    path("api/search/", api_views.SearchView.as_view(), name="search_api"),
    path("api/health/", api_views.HealthView.as_view(), name="health_api"),
    path("api/ingest/", api_views.IngestView.as_view(), name="ingest_api"),
    path("api/chats/", api_views.ChatListView.as_view(), name="chats_api"),
    path("api/chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="rename_chat_api"),
    path("api/chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="chat_delete_api"),
    path("api/chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="chat_detail_api"),

    # Flexible regex endpoints (supporting without trailing slashes like /ask and /health)
    re_path(r"^(?:api/)?(?:v1/)?ask/?$", api_views.AskView.as_view(), name="ask_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?search/?$", api_views.SearchView.as_view(), name="search_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?health/?$", api_views.HealthView.as_view(), name="health_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?ingest/?$", api_views.IngestView.as_view(), name="ingest_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?chats/?$", api_views.ChatListView.as_view(), name="chats_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?chats/(?P<session_id>[^/]+)/rename/?$", api_views.ChatRenameView.as_view(), name="rename_chat_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?chats/(?P<session_id>[^/]+)/delete/?$", api_views.ChatDeleteView.as_view(), name="chat_delete_flexible"),
    re_path(r"^(?:api/)?(?:v1/)?chats/(?P<session_id>[^/]+)/?$", api_views.ChatDetailView.as_view(), name="chat_detail_flexible"),
]
