#!/usr/bin/env python
"""Django's command-line utility (runserver, build_index, test, ...)."""
import os
import sys


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
<<<<<<< HEAD

    # If running "runserver" in a cloud environment (Render, etc.)
    if len(sys.argv) >= 2 and sys.argv[1] == "runserver":
        port = os.getenv("PORT")
        # Automatically bind to 0.0.0.0:$PORT if not explicitly specified
        if port and len(sys.argv) == 2:
            sys.argv.extend([f"0.0.0.0:{port}", "--noreload"])

        # Auto-apply database migrations on cloud startup to ensure session tables exist
        if os.getenv("RENDER"):
            try:
                import django
                from django.core.management import call_command

                django.setup()
                call_command("migrate", interactive=False)
            except Exception as exc:
                print(f"[Startup] Auto-migration note: {exc}", file=sys.stderr)

=======
>>>>>>> master
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Activate your virtual environment and run "
            "'pip install -r requirements.txt'."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
