# AGENTS.md — MongoDB RAG Chatbot

## Project Overview
**Mongo RAG Pro** is an enterprise-grade retrieval-augmented generation system that answers plain-language questions across complex MongoDB schemas. It converts raw documents into natural language, resolves relational references across collections without hardcoded schemas, embeds chunks into Qdrant Vector DB, performs hybrid semantic + keyword search with multi-hop graph expansion, executes deterministic structured queries for aggregate/list calculations, and synthesizes answers via LLMs (Google Gemini, OpenAI, Groq, Ollama) while enforcing strict role-based access control and data privacy.

The system operates in two main pipelines:
- **Ingestion Pipeline**: `python manage.py ingest` or `POST /api/v1/ingest/` — reads MongoDB collections, resolves relationships, converts documents to text, chunks, embeds, and stores in Qdrant.
- **Ask/Query Pipeline**: `POST /api/v1/ask/` — processes user questions through conversational checks, personal routing, structured query planning, hybrid semantic retrieval, and LLM answer generation.

## Technology Stack
- **Framework**: Django 5.0+ with Django REST Framework
- **Database**: MongoDB Atlas (operational store) + Qdrant (vector store)
- **LLM Providers**: Google Gemini (primary), OpenAI/Groq (fallback), Ollama (local)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local) or text-embedding-3-small (OpenAI)
- **Vector DB**: Qdrant Cloud or local embedded
- **API**: REST endpoints under `/api/v1/`

## Project Structure
```
MONGO_CHATBOT/
├── config/                  # Django settings, URLs, WSGI/ASGI
├── rag/                     # Main application logic
│   ├── services/            # Core services (orchestration, querying, embeddings, LLM, etc.)
│   ├── models.py            # SQLite chat session/models
│   ├── views.py             # DRF viewsets/controllers
│   ├── urls.py              # API routing
│   ├── tests.py             # Test suite
│   ├── middleware.py        # Request logging middleware
│   ├── apps.py              # App config
│   └── ...
├── manage.py                # Django CLI entrypoint
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Dev dependencies
├── .env                     # Environment variables (never commit)
└── README.md                # Full documentation
```

## Key Conventions & Rules

### Environment & Configuration
- **Never commit `.env`** — it contains secrets (MongoURI, API keys, JWT secret). The `.env.example` template is provided.
- Environment variables are loaded via `python-debug` in `config/settings.py`.
- Key vars: `MONGO_URI`, `MONGO_DATABASE`, `QDRANT_URL`/`QDRANT_API_KEY`, `LLM_PROVIDER`, `GEMINI_API_KEY`, `JWT_SECRET`, `INGEST_TOKEN`.
- `config/settings.py` uses helpers: `_bool()`, `_int()`, `_float()`, `_list()` for typed env reading.

### API Usage
All API calls use JSON payloads under `/api/v1/`. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ask/` | Full RAG pipeline — returns natural language answer |
| `POST` | `/api/v1/search/` | Semantic + hybrid search without LLM synthesis |
| `GET` | `/api/v1/chats/` | List chat sessions |
| `GET` | `/api/v1/chats/{session_id}/` | Retrieve chat history |
| `POST` | `/api/v1/chats/{session_id}/rename/` | Update session title |
| `DELETE` | `/api/v1/chats/{session_id}/delete/` | Delete chat session |
| `GET` | `/api/v1/health/` | Health check (MongoDB, Qdrant, embeddings, LLM) |
| `POST` | `/api/v1/ingest/` | Trigger data ingestion (requires `INGEST_TOKEN`) |

**`/api/v1/ask/` Request example:**
```json
{
  "question": "who is on leave on 13 nov",
  "session_id": "...",
  "user_id": "...",
  "email": "employee@example.com",
  "workspace_id": "...",
  "top_k": 6,
  "hops": 1
}
```

**Important**: Every request must include a valid `workspace_id`. Membership is strictly verified before data retrieval. Non-admin users have scoped access to their assigned work.

### Code Conventions
- Follow Django/DRF patterns established in the codebase.
- Use collection aliases in `config/settings.py` DEFAULT_COLLECTION_ALIASES map when referring to collections by business terms (e.g., "project" → "boards", "leave request" → "leaverequests").
- ID sanitization: `_sanitize_ids` dynamically resolves raw ObjectIds to user names via the `users` collection. `_clean_context_ids` scrubs IDs from LLM prompts.
- Structured queries use a strict operator whitelist: `eq`, `ne`, `in`, `contains`, `gt`, `gte`, `lt`, `lte`.
- Multi-hop relationship expansion respects `MAX_HOPS` setting (default: 1).
- Vector search uses RRF (Reciprocal Rank Fusion) with `RRF_K=60`.

### Testing
- Run Django test suite: `python manage.py test`
- Run specific leave query tests: `python test_specific_leave_queries.py`
- Tests verify structured query planning, leave engine logic, date boundaries, and ID sanitization.

### Running the Application
```powershell
# 1. Setup virtual environment
cd MONGO_CHATBOT
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Configure environment
copy .env.example .env
# Edit .env with your MongoDB, Qdrant, and LLM credentials

# 3. Ingest data
python manage.py ingest

# 4. Start server
python manage.py runserver
```

### Deployment Notes
- `manage.py runserver` auto-binds to `0.0.0.0:$PORT` on cloud platforms (Render).
- On cloud startup, migrations auto-run if `RENDER` env var is set.
- WhiteNoise serves static files in production.
- CORS is configured for localhost, 127.0.0.1, and onrender.com domains.

### Privacy & Access Control
- **Admin vs Non-Admin**: Admins have org-wide visibility; employees have scoped access.
- **Workspace Scoping**: All requests must include a valid `workspace_id`; membership verified before retrieval.
- **Privacy Enforcement**: Non-admins cannot retrieve private leave records or private work of other employees.
- **Raw ID Exposure**: Never display raw ObjectIds to frontend — always resolve via `_sanitize_ids`.