
"""Versioned Django REST Framework API for chatbot integrations."""
import hashlib
import hmac
from uuid import uuid4

from django.conf import settings
import jwt
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from . import api_serializers
from .services import authorization_service, chat_memory, ingestion_service, rag_service
from .services.anonymous_session import get_chat_owner
from .services.exceptions import AuthError, ChatbotError, InvalidRequestError


def _error_response(exc):
    if isinstance(exc, ChatbotError):
        return Response({"error": exc.message}, status=exc.status_code)
    raise exc


def _validate_user_token(request, user_id=None, email=None):
    authorization = request.headers.get("Authorization", "")
    if not authorization and not user_id:
        return None
    if not settings.CHAT_JWT_SECRET:
        raise AuthError("Account-scoped chat requires JWT_SECRET to be configured.")

    scheme, _, token = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise AuthError("A valid account token is required.")
    try:
        claims = jwt.decode(token, settings.CHAT_JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError as exc:
        raise AuthError("The account token is invalid or expired.") from exc

    token_user_id = claims.get("id") or claims.get("_id") or claims.get("sub")
    if not token_user_id:
        raise AuthError("The account token does not contain a user ID.")
    token_user_id = str(token_user_id)
    if user_id and token_user_id != str(user_id):
        raise AuthError("The account token does not match the supplied user ID.")
    token_email = claims.get("email")
    if email and token_email and email.strip().casefold() != str(token_email).strip().casefold():
        raise AuthError("The account token does not match the supplied email.")
    authorization_service.resolve_scope(token_user_id, email=email or token_email)
    return token_user_id


def _set_chat_owner(request, user_id):
    if user_id:
        owner_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        request.session["chat_owner_id"] = f"account:{owner_hash[:56]}"
    elif str(request.session.get("chat_owner_id", "")).startswith("account:"):
        request.session["chat_owner_id"] = uuid4().hex


class AskView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = api_serializers.AskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            token_user_id = _validate_user_token(request, data.get("user_id"), data.get("email"))
            http_request = request._request
            _set_chat_owner(http_request, token_user_id)
            owner_id = get_chat_owner(http_request)
            session_id = data.get("session_id")
            session = chat_memory.get_session(http_request, session_id) if session_id else None
            if session_id and session is None and chat_memory.session_exists(session_id):
                return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)

            history_state = chat_memory.get_history_state(session)
            edit_index = data.get("edit_index")
            service_edit_index = None
            history = history_state["turns"]
            if edit_index is not None:
                offset = history_state["offset"]
                if not offset <= edit_index < offset + len(history):
                    raise InvalidRequestError(
                        f"This turn is outside the {chat_memory.MAX_TURNS}-turn edit history. "
                        "Refresh the chat and edit a more recent turn."
                    )
                service_edit_index = edit_index - offset

            result = rag_service.ask(
                data["question"],
                top_k=data.get("top_k"),
                collections=data.get("collections") or None,
                hops=data.get("hops"),
                edit_index=service_edit_index,
                user_id=token_user_id or owner_id,
                email=data.get("email") or None,
                history=history,
            )
            session = chat_memory.save_exchange(
                http_request,
                session,
                data["question"],
                result["answer"],
                sources=result.get("sources", []),
                session_id=session_id,
                edit_index=edit_index,
            )
            return Response({
                **result,
                "sources": result.get("sources", []),
                "session_id": str(session.pk),
            })
        except ChatbotError as exc:
            return _error_response(exc)


class SearchView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = api_serializers.SearchRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            token_user_id = _validate_user_token(request, data.get("user_id"), data.get("email"))
            _set_chat_owner(request._request, token_user_id)
            result = rag_service.search(
                data["query"],
                top_k=data.get("top_k"),
                collections=data.get("collections") or None,
                hybrid=data.get("hybrid"),
                expand=data["expand"],
                hops=data.get("hops"),
                user_id=token_user_id or get_chat_owner(request._request),
                email=data.get("email") or None,
            )
            return Response(result)
        except ChatbotError as exc:
            return _error_response(exc)


class HealthView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            return Response(rag_service.health())
        except ChatbotError as exc:
            return _error_response(exc)


class ChatListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            user_id = _validate_user_token(request)
            _set_chat_owner(request._request, user_id)
            return Response({"sessions": chat_memory.list_sessions(request._request)})
        except ChatbotError as exc:
            return _error_response(exc)


class ChatDetailView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, session_id):
        try:
            user_id = _validate_user_token(request)
            _set_chat_owner(request._request, user_id)
            session = chat_memory.get_session(request._request, session_id)
            if session is None:
                return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)
            return Response({
                "session_id": str(session.pk),
                "title": session.title,
                "turns": chat_memory.get_history(session),
                "offset": max(session.messages.filter(role="user").count() - chat_memory.MAX_TURNS, 0),
                "messages": chat_memory.get_messages(session),
            })
        except ChatbotError as exc:
            return _error_response(exc)


class ChatRenameView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, session_id):
        serializer = api_serializers.RenameChatSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        title = serializer.validated_data["title"]
        try:
            user_id = _validate_user_token(request)
            _set_chat_owner(request._request, user_id)
            if not chat_memory.rename_session(request._request, session_id, title):
                return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)
            return Response({"success": True, "title": title})
        except ChatbotError as exc:
            return _error_response(exc)


class ChatDeleteView(APIView):
    permission_classes = [AllowAny]

    def delete(self, request, session_id):
        try:
            user_id = _validate_user_token(request)
            _set_chat_owner(request._request, user_id)
            chat_memory.delete_session(request._request, session_id)
            return Response({"success": True})
        except ChatbotError as exc:
            return _error_response(exc)

    def post(self, request, session_id):
        return self.delete(request, session_id)


class IngestView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def _authorize(self, request):
        token = settings.INGEST_TOKEN
        if not token and settings.DEBUG:
            return
        if not token:
            raise AuthError("Ingestion over HTTP is disabled. Configure the server-side ingest token.")
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {token}"
        if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            raise AuthError()

    def get(self, request):
        try:
            self._authorize(request)
            return Response(ingestion_service.job_status())
        except ChatbotError as exc:
            return _error_response(exc)

    def post(self, request):
        try:
            self._authorize(request)
            serializer = api_serializers.IngestRequestSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            data = serializer.validated_data
            job = ingestion_service.start_job(
                collections=data.get("collections") or None,
                full=data["full"],
                max_docs=data.get("max_docs", 0),
            )
            return Response(job, status=status.HTTP_202_ACCEPTED)
        except ChatbotError as exc:
            return _error_response(exc)


