from django.contrib.auth import authenticate
from django.urls import path, re_path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username") or request.data.get("email")
        password = request.data.get("password")
        user = authenticate(request, username=username, password=password)
        if not user:
            return Response({"success": False, "error": "Invalid credentials"}, status=401)
        refresh = RefreshToken.for_user(user)
        return Response({
            "success": True,
            "token": str(refresh.access_token),
            "refresh": str(refresh),
            "user": {
                "id": user.id,
                "name": user.get_full_name() or user.username,
                "email": user.email,
                "role": "admin" if user.is_staff else "user",
            },
        })


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        u = request.user
        return Response({
            "success": True,
            "user": {
                "id": u.id,
                "name": u.get_full_name() or u.username,
                "email": u.email,
                "role": "admin" if u.is_staff else "user",
            },
        })


urlpatterns = [
    path("login/", LoginView.as_view(), name="auth_login_path"),
    path("me/", MeView.as_view(), name="auth_me_path"),
    re_path(r"^login/?$", LoginView.as_view(), name="auth_login"),
    re_path(r"^me/?$", MeView.as_view(), name="auth_me"),
]
