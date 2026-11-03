import os
import sys

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# Auto-apply database migrations on cloud startup to ensure session tables exist
if os.getenv("RENDER"):
    try:
        import django
        from django.core.management import call_command

        django.setup()
        call_command("migrate", interactive=False)
    except Exception as exc:
        print(f"[WSGI Startup] Auto-migration note: {exc}", file=sys.stderr)

application = get_wsgi_application()
