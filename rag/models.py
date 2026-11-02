import uuid
from django.conf import settings
from django.db import models

class ChatSession(models.Model):
    """A conversation thread. One per 'new chat'."""
    id = models.CharField(primary_key=True, default=uuid.uuid4, max_length=64, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.CASCADE, related_name="chat_sessions",
    )
    anon_key = models.CharField(max_length=64, blank=True, db_index=True)  # anonymous session key
    title = models.CharField(max_length=200, default="New chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)


class ChatMessage(models.Model):
    ROLE_USER = "user"
    ROLE_BOT = "bot"
    ROLE_CHOICES = [(ROLE_USER, "User"), (ROLE_BOT, "Bot")]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=8, choices=ROLE_CHOICES)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)   # optional RAG citations
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_at",)




# No Django models: data lives in MongoDB (see rag/services/embedding_store.py).
# rag/models.py
# from django.db import models

# class ChatSession(models.Model):
#     session_id = models.CharField(max_length=64, unique=True, db_index=True)
#     title = models.CharField(max_length=120, default="New chat")
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     class Meta:
#         ordering = ["-updated_at"]

#     def __str__(self):
#         return f"{self.title} ({self.session_id[:8]})"