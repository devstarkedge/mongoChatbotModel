from django.http import HttpResponse, JsonResponse


def index(request):
    """API-only root. Returns JSON info for GET, empty 200 for HEAD (UptimeRobot)."""
    if request.method == "HEAD":
        return HttpResponse()
    return JsonResponse({
        "service": "mongo-chatbot",
        "status": "ok",
        "docs": "/api/v1/health/",
    })
