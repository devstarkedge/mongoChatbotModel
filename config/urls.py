from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("rag.urls")),                       # rag.urls already contains api/v1/...
    path("api/auth/", include("rag.auth_urls")),         # login / me for React
]
