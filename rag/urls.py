from django.urls import path

from . import views
from . import api_views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/v1/ask/", api_views.AskView.as_view(), name="api_ask"),
    path("api/v1/search/", api_views.SearchView.as_view(), name="api_search"),
    path("api/v1/health/", api_views.HealthView.as_view(), name="api_health"),
    path("api/v1/ingest/", api_views.IngestView.as_view(), name="api_ingest"),
    path("api/v1/chats/", api_views.ChatListView.as_view(), name="api_chats"),
    path("api/v1/chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="api_chat_rename"),
    path("api/v1/chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="api_chat_delete"),
    path("api/v1/chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="api_chat_detail"),
    path("api/ask/", api_views.AskView.as_view(), name="ask"),
    path("api/search/", api_views.SearchView.as_view(), name="search"),
    path("api/health/", api_views.HealthView.as_view(), name="health"),
    path("api/ingest/", api_views.IngestView.as_view(), name="ingest"),
    path("api/chats/", api_views.ChatListView.as_view(), name="chats_list"),
    path("api/chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="rename_chat"),
    path("api/chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="chat_delete"),
    path("api/chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="chat_detail"),
]
