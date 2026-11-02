#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run database migrations (creates django_session and app tables)
python manage.py migrate

# Collect static files for WhiteNoise
python manage.py collectstatic --no-input
