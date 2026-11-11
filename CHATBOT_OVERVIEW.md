# Mongo RAG Pro — Chatbot Overview

## What it is
Mongo RAG Pro is a Django-based chatbot that answers plain-language questions about any MongoDB database. It combines:
- **Structured queries** (deterministic, exact) for leave/logged-time/user profiles/projects/etc.
- **RAG** (semantic + keyword hybrid search + LLM generation) for everything else.
- **Relationship resolution** (ObjectId links between collections).
- **Role-based access** (employee sees only own records; admin sees all).

## Architecture (high level)

User (Browser/React)
  -> Django REST API (/api/v1/ask, /api/v1/search)
    -> Authorization (JWT + scope resolution)
    -> RAG Service (ask/search routing)
       -> Structured Query first (try_structured)
       -> else Retrieval (hybrid search + embeddings)
       -> LLM Service (planner + generate)
    -> Mongo Service (DB access)
  -> MongoDB (collections + rag_embeddings vector store)
  -> LLM Provider (Ollama / OpenAI / Groq)

## Main collections used
activities, boards, cards, departments, leave_entries, memberships, notes, projects, rag_embeddings, subtasks, tasks, users, workspaces

Primary collection is cards (tasks + nested loggedTime entries), used by both structured queries and RAG. users is used for identity/permissions. ag_embeddings holds the vector store.

## Key services (rag/services/)
- mongo_service — DB access, list_collections
- rag_service — ask/search routing, chat memory integration
- structured_query — deterministic answers (logged time, leave, user profiles, projects, departments, etc.)
- llm_service — query planning + answer generation
- retrieval / keyword_search — hybrid search
- embedding_service / embedding_store / vector_index — embeddings + vector search
- authorization_service — scope resolution (employee/admin)
- chat_memory — in-memory session turns
- cache_service — caching
- schema_catalog — auto-profiles collections for the LLM
- ingestion_service / document_converter / chunking — ingestion pipeline
- relationships — ObjectId link resolution

## Query flow
1. Request arrives at /api/v1/ask/ with question, optional user_id/email/role, session_id.
2. If user_id supplied, resolve account + scope from MongoDB; JWT must match; role is informational only.
3. Anonymous requests can chat but cannot retrieve indexed records.
4. rag_service.ask:
   - Try structured_query first (logged time, leave, profiles, etc.).
   - If no structured answer, retrieve via hybrid search (semantic + keyword), follow ObjectId refs, then LLM generates answer from retrieved records only.
5. Employee scope restricts records to those linked to their user_id; admin sees all.
6. Chat history kept in memory, associated with browser session, expires after 1h inactivity, retains latest 10 turns.

## API endpoints
- POST /api/v1/ask/ — search + generate answer
- POST /api/v1/search/ — return matching records only
- GET /api/v1/chats/ — list sessions
- GET /api/v1/chats/{session_id}/ — read turns
- POST /api/v1/chats/{session_id}/rename/ — rename
- DELETE /api/v1/chats/{session_id}/delete/ — delete
- GET /api/v1/health/ — service status
- GET/POST /api/v1/ingest/ — ingestion status/start (requires ingest token)

## Security
- No user account required to reach service; deploy only on trusted networks.
- Admin determined by MongoDB role, not request role field.
- Employee cannot view another user's data; cross-user name queries return "not authenticated" style response.
- JWT_SECRET must match between Django and frontend for account-scoped requests.

## Ingestion
- Documents of any schema converted to natural language.
- ObjectId relationships detected and resolved.
- Embeddings stored in MongoDB (rag_embeddings).
- Run: python manage.py ingest

## Logged-time feature (recently enhanced)
- Reads nested loggedTime entries on cards.
- Supports exact date, month, year, week scoping.
- Superlative queries: "who has highest/lowest logged time in September / 2026 / this week".
- Team-level ranking aggregates per person and names the top contributor.
- Period resolution: _extract_period handles bare months, bare years, "this week/month/year", exact dates.

## Configuration
- .env (from .env.example) for local Ollama profile.
- MONGO_URI, MONGO_DATABASE required.
- LLM_PROVIDER: ollama (default local) / openai / groq.
- CORS via DJANGO_CORS_ALLOWED_ORIGINS.

## Running locally
1. copy .env.example .env
2. set MONGO_URI, MONGO_DATABASE
3. python manage.py ingest
4. python manage.py runserver
5. (Ollama) ollama pull gemma3:4b

## Tech stack
Django, Django REST Framework, MongoDB (pymongo), sentence-transformers, Ollama/OpenAI/Groq, mongomock (tests).
