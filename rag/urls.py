from django.urls import path

from . import views
from . import api_views

urlpatterns = [
    path("", views.index, name="index"),  # Home page / UI entry point
    path("api/v1/ask/", api_views.AskView.as_view(), name="api_ask"),  # Ask a question and get an AI answer (with RAG)
    path("ask/", api_views.AskView.as_view(), name="ask_alias"),  # Alias of ask endpoint (same task)
    path("api/v1/search/", api_views.SearchView.as_view(), name="api_search"),  # Semantic search over documents
    path("api/v1/health/", api_views.HealthView.as_view(), name="api_health"),  # Service health check
    path("api/v1/ingest/", api_views.IngestView.as_view(), name="api_ingest"),  # Ingest documents into vector store
    path("api/v1/chats/", api_views.ChatListView.as_view(), name="api_chats"),  # List user chat sessions
    path("api/v1/chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="api_chat_rename"),  # Rename a chat session
    path("api/v1/chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="api_chat_delete"),  # Delete a chat session
    path("api/v1/chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="api_chat_detail"),  # Get chat session details/messages
    path("api/ask/", api_views.AskView.as_view(), name="ask"),  # Ask a question (v0 path)
    path("api/search/", api_views.SearchView.as_view(), name="search"),  # Semantic search (v0 path)
    path("api/health/", api_views.HealthView.as_view(), name="health"),  # Health check (v0 path)
    path("api/ingest/", api_views.IngestView.as_view(), name="ingest"),  # Ingest documents (v0 path)
    path("api/chats/", api_views.ChatListView.as_view(), name="chats_list"),  # List chats (v0 path)
    path("api/chats/<str:session_id>/rename/", api_views.ChatRenameView.as_view(), name="rename_chat"),  # Rename chat (v0 path)
    path("api/chats/<str:session_id>/delete/", api_views.ChatDeleteView.as_view(), name="chat_delete"),  # Delete chat (v0 path)
    path("api/chats/<str:session_id>/", api_views.ChatDetailView.as_view(), name="chat_detail"),  # Chat details (v0 path)
]
