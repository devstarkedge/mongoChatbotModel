"""
All configuration lives here and comes from environment variables (.env).
Nothing sensitive is hardcoded or sent to the browser.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _list(name, default=""):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

# LLM backend: "openai" (any OpenAI-compatible API: Groq, OpenAI, ...), "gemini" (Google Gemini) or "ollama" (local model).
# The local profile uses smaller prompts and longer timeouts by default: a 4B model on a laptop is slower and has
# a smaller effective context than a hosted model.
_has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if _has_gemini else "openai").strip().lower()   # openai | ollama | gemini
_LOCAL = LLM_PROVIDER == "ollama"
_GEMINI = LLM_PROVIDER == "gemini"      # Google Gemini through its OpenAI-compatible endpoint

# ------------------------------------------------------------------ Django
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
CHAT_JWT_SECRET = os.getenv("JWT_SECRET", "")
DEBUG = _bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = _list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,.onrender.com")
_render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
if _render_host and _render_host not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(_render_host)

CSRF_TRUSTED_ORIGINS = _list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    "http://localhost,http://127.0.0.1,http://localhost:3000,http://127.0.0.1:3000,https://*.onrender.com",
)
if _render_host:
    CSRF_TRUSTED_ORIGINS.append(f"https://{_render_host}")
INSTALLED_APPS = INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "rag.apps.RagConfig",
]
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "rag.middleware.request_log_middleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = not DEBUG
SESSION_ENGINE = "django.contrib.sessions.backends.db"
CSRF_COOKIE_SAMESITE = "None" if not DEBUG else "Lax"
CSRF_COOKIE_SECURE = not DEBUG
SESSION_COOKIE_AGE = 60 * 60 * 12
CORS_ALLOWED_ORIGINS = _list(
    "DJANGO_CORS_ALLOWED_ORIGINS",
<<<<<<< HEAD
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,https://chatbot-frontend-1gbi.onrender.com",
)
CORS_ALLOW_CREDENTIALS = True
CORS_URLS_REGEX = r"^.*$"
CORS_ALLOW_ALL_ORIGINS = _bool("CORS_ALLOW_ALL_ORIGINS", True)
CORS_ALLOWED_ORIGIN_REGEXES = [
    r"^https:\/\/.*\.onrender\.com$",
    r"^http:\/\/localhost:\d+$",
]
CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]
CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]
=======
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
)
CORS_ALLOW_CREDENTIALS = True
CORS_URLS_REGEX = r"^/api/.*$"
>>>>>>> master
REST_FRAMEWORK = {
    "UNAUTHENTICATED_USER": None,
    "UNAUTHENTICATED_TOKEN": None,
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "EXCEPTION_HANDLER": "rag.api_errors.exception_handler",
}
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [BASE_DIR / "templates"],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",   # ← admin needs this
        "django.template.context_processors.debug",      
        "django.contrib.auth.context_processors.auth",      # ← admin needs this
        "django.contrib.messages.context_processors.messages",
        ]},
}]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
WHITENOISE_USE_FINDERS = True
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"std": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "std"}},
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {"httpx": {"level": "WARNING"}, "httpcore": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"}, "huggingface_hub": {"level": "WARNING"}},
}

# ----------------------------------------------------------------- MongoDB
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "")
# Embeddings are stored in their own collection (optionally in a different database).
EMBEDDINGS_COLLECTION = os.getenv("EMBEDDINGS_COLLECTION", "rag_embeddings")
EMBEDDINGS_DATABASE = os.getenv("EMBEDDINGS_DATABASE", "")
EXCLUDE_COLLECTIONS = _list("EXCLUDE_COLLECTIONS")
# Collections that are never indexed, searched, planned against, or exposed by the RAG pipeline.
# "activities" is always blocked in code as well (mongo_service.blocked_collections).
BLOCKED_COLLECTIONS = _list("BLOCKED_COLLECTIONS", "activities")
EXCLUDE_FIELDS = [f.lower() for f in _list("EXCLUDE_FIELDS")]
BATCH_SIZE = max(10, _int("BATCH_SIZE", 500))
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")   # required for POST /api/ingest/ outside DEBUG

# --------------------------------------------------------- Text conversion
MAX_FIELD_CHARS = _int("MAX_FIELD_CHARS", 20000)
MAX_DOC_CHARS = _int("MAX_DOC_CHARS", 50000)
CHUNK_SIZE = max(200, _int("CHUNK_SIZE", 700))
CHUNK_OVERLAP = max(0, min(_int("CHUNK_OVERLAP", 100), CHUNK_SIZE // 2))

# -------------------------------------------------------------- Embeddings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()  # local | openai
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2" if EMBEDDING_PROVIDER == "local" else "text-embedding-3-small",
)
EMBEDDING_MODEL_ID = f"{EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")      # only for provider=openai
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store")

# ------------------------------------------------------------------ Search
TOP_K = max(1, _int("TOP_K", 4 if _LOCAL else 5))
MIN_SCORE = _float("MIN_SCORE", 0.15)              # min cosine similarity for vector hits
HYBRID_SEARCH = _bool("HYBRID_SEARCH", True)       # vector + MongoDB $text keyword search
KEYWORD_WEIGHT = _float("KEYWORD_WEIGHT", 0.5)
RRF_K = _int("RRF_K", 60)
MAX_HOPS = max(0, _int("MAX_HOPS", 1))             # relationship hops followed in /ask
MAX_RELATED = max(0, _int("MAX_RELATED", 6))
INCLUDE_REVERSE = _bool("INCLUDE_REVERSE", True)   # also find records that point AT a hit
REVERSE_PER_RECORD = max(1, _int("REVERSE_PER_RECORD", 3))
MAX_CONTEXT_CHARS = _int("MAX_CONTEXT_CHARS", 6000 if _LOCAL else 12000)
<<<<<<< HEAD
=======
# Conversation memory: the last 50 turns are retained, but only this many characters of
# older turns are sent to the language model so the prompt stays within its context window.
MAX_HISTORY_CHARS = _int("MAX_HISTORY_CHARS", 8000 if _LOCAL else 24000)
>>>>>>> master
# List-style questions ("list all...", "which projects...", "how many...") automatically retrieve more.
# Set LIST_TOP_K=0 to turn this off.
LIST_TOP_K = max(0, _int("LIST_TOP_K", 20 if _LOCAL else 30))
LIST_CONTEXT_CHARS = _int("LIST_CONTEXT_CHARS", 9000 if _LOCAL else 30000)
LIST_MAX_TOKENS = _int("LIST_MAX_TOKENS", 900 if _LOCAL else 2048)

# ------------------------------------------- exact answers for list/count questions
STRUCTURED_QUERIES = _bool("STRUCTURED_QUERIES", True)   # list/count/total questions are answered from MongoDB directly
STRUCTURED_MAX_ROWS = max(1, _int("STRUCTURED_MAX_ROWS", 500))
STRUCTURED_PLANNER_TOKENS = _int("STRUCTURED_PLANNER_TOKENS", 1200)
CATALOG_TTL_SECONDS = _int("CATALOG_TTL_SECONDS", 600)   # how long the sampled schema is reused
CATALOG_SAMPLE_SIZE = max(20, _int("CATALOG_SAMPLE_SIZE", 200))
QUERY_TIMEOUT_MS = _int("QUERY_TIMEOUT_MS", 15000)

# ------------------------------------------------ Exact (structured) queries
# List / count questions ("list all departments", "how many tasks are medium priority") are answered by
# querying MongoDB directly, so totals are exact. Set STRUCTURED_QUERIES=false to use similarity search only.

STRUCT_MAX_RECORDS = max(1, _int("STRUCT_MAX_RECORDS", 40 if _LOCAL else 60))   # records handed to the LLM (total is always exact)
STRUCT_SCAN_MAX = _int("STRUCT_SCAN_MAX", 150)                 # no-LLM fallback: read a whole collection up to this size
STRUCT_TIMEOUT_MS = _int("STRUCT_TIMEOUT_MS", 10000)
# "on": the LLM plans list/count questions (filters). "off": no planning call (faster on a slow local model):
# a question that names one small collection still returns its complete list, but conditions are applied by the LLM.
# Accept both the documented on/off values and common boolean spellings.
# Keep this as a string because callers and tests may override it with "off".
STRUCT_PLANNER = "on" if _bool("STRUCT_PLANNER", True) else "off"
CATALOG_CHARS = _int("CATALOG_CHARS", 3500 if _LOCAL else 16000)    # size of the catalog shown to the planner
CATALOG_MAX_AGE_HOURS = _int("CATALOG_MAX_AGE_HOURS", 8760)       # saved catalog is reused (refresh with: ingest)

# Collection and domain synonyms: maps business terms to underlying MongoDB collections
DEFAULT_COLLECTION_ALIASES = {
    "project": "boards",
    "projects": "boards",
    "board": "boards",
    "task": "cards",
    "tasks": "cards",
    "card": "cards",
    "subtask": "subtasks",
    "subtasks": "subtasks",
    "subtasknanos": "subtasknanos",
    "subtasknano": "subtasknanos",
    "vacation": "cards",
    "leavepolicy": "leavepolicies",
    "leavepolicies": "leavepolicies",
    "leavepolicyversions": "leavepolicyversions",
    "credit": "leavepolicyversions",
    "credits": "leavepolicyversions",
    "credited": "leavepolicyversions",
    "crediting": "leavepolicyversions",
    "accrual": "leavepolicyversions",
    "accruals": "leavepolicyversions",
    "allowance": "leavepolicyversions",
    "allowances": "leavepolicyversions",
    "employee": "users",
    "employees": "users",
    "member": "users",
    "members": "users",
    "user": "users",
    "users": "users",
    "staff": "users",
    "person": "users",
<<<<<<< HEAD
    "people": "users",
=======
    "persons": "users",
    "people": "users",
    "peoples": "users",
>>>>>>> master
    "department": "departments",
    "departments": "departments",
    "workspace": "workspaces",
    "workspaces": "workspaces",
    "company": "workspaces",
    "companies": "workspaces",
    "organization": "workspaces",
    "organizations": "workspaces",
    "org": "workspaces",
    "team": "workspaces",
    "teams": "workspaces",
    "membership": "workspacememberships",
    "memberships": "workspacememberships",
    "time": "cards",
    "logged": "cards",
    "loggedtime": "cards",
    "timesheet": "cards",
    "hours": "cards",
    "worklog": "cards",
    "worklogs": "cards",
    "developer": "users",
    "developers": "users",
    "sale": "salesrows",
    "sales": "salesrows",
    "lead": "salesrows",
    "leads": "salesrows",
    "bid": "salesrows",
    "bids": "salesrows",
    "bidding": "salesrows",
    "proposal": "salesrows",
    "proposals": "salesrows",
    "announcement": "announcements",
    "announcements": "announcements",
    "notice": "announcements",
    "notices": "announcements",
    "news": "announcements",
    "milestone": "milestones",
    "milestones": "milestones",
    "payment": "milestones",
    "payments": "milestones",
    "holiday": "holidays",
    "holidays": "holidays",
    "column": "lists",
    "columns": "lists",
    "stage": "lists",
    "stages": "lists",
    "list": "lists",
    "lists": "lists",
    "comment": "comments",
    "comments": "comments",
    "notification": "notifications",
    "notifications": "notifications",
    "alert": "notifications",
    "alerts": "notifications",
    "attachment": "attachments",
    "attachments": "attachments",
    "file": "attachments",
    "files": "attachments",
    "role": "roles",
    "roles": "roles",
}
COLLECTION_ALIASES = DEFAULT_COLLECTION_ALIASES.copy()
if os.getenv("COLLECTION_ALIASES"):
    try:
        import json as _json
        COLLECTION_ALIASES.update(_json.loads(os.getenv("COLLECTION_ALIASES")))
    except Exception:
        pass

# Cards are the primary source for employee task time; subtasks are included
# because some records store their own nested loggedTime entries.
# LOGGED_TIME_COLLECTIONS = ("cards","boards", "subtasks","activities" ,"subtasknanos")
LOGGED_TIME_COLLECTIONS = ("cards",)


# --------------------------------------------------------------------- LLM
# Any OpenAI-compatible API. Defaults: Groq + gpt-oss-20b.
# OpenAI: LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
LLM_API_KEY = (os.getenv("LLM_API_KEY") or
               (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "") if _GEMINI else os.getenv("GROQ_API_KEY", "")))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/" if _GEMINI
                         else "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b" if _LOCAL else "gemini-3.5-flash-lite" if _GEMINI else "openai/gpt-oss-20b")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "" if _GEMINI else "0")   # never sent to Gemini (deprecated there)   # empty = provider default
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0) if not _GEMINI else None
# Name of the "maximum output tokens" request field. Gemini's OpenAI-compatible endpoint documents max_tokens.
LLM_MAX_TOKENS_PARAM = os.getenv("LLM_MAX_TOKENS_PARAM", "max_tokens" if _GEMINI else "max_completion_tokens")
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 600 if _LOCAL else 1500)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 180 if _LOCAL else 60)
# --- Ollama (LLM_PROVIDER=ollama): talks to the native API, which can set the context window per request
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip().rstrip("/")
OLLAMA_NUM_CTX = _int("OLLAMA_NUM_CTX", 8192)          # tokens; Ollama's own default is smaller and silently cuts prompts
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")   # keep the model in memory between questions
# Speed: reasoning models (gpt-oss) "think" before answering. LLM_REASONING_EFFORT=low makes them much faster.
# Leave empty if your provider does not support it (it is dropped automatically if the provider rejects it).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()
# Optional smaller/faster model just for the query-planning step of list/count questions.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "").strip()

# ------------------------------------------------------------------- Cache
CACHE_TTL_SECONDS = _int("CACHE_TTL_SECONDS", 600)   # 0 disables the answer cache
CACHE_MAX_ENTRIES = _int("CACHE_MAX_ENTRIES", 500)
<<<<<<< HEAD
_default_warmup = not bool(os.getenv("RENDER"))
WARMUP_ON_START = _bool("WARMUP_ON_START", _default_warmup)
=======
WARMUP_ON_START = _bool("WARMUP_ON_START", True)
>>>>>>> master
