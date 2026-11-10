from django.contrib import admin
<<<<<<< HEAD
from django.urls import include, path, re_path
from django.http import JsonResponse
from django.urls import path


def health_check(request):
    return JsonResponse({"status": "ok"})

urlpatterns = [
    path("admin/", admin.site.urls),
    re_path(r"^(?:api/)?auth/", include("rag.auth_urls")),
    path("", include("rag.urls")),
    path('health', health_check),
]
=======
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("rag.urls")),                       # rag.urls already contains api/v1/...
    path("api/auth/", include("rag.auth_urls")),         # login / me for React
]
>>>>>>> master
