# Mongo RAG Pro: MongoDB, Relationships, Vector Search & Structured Intelligence

An enterprise-grade, retrieval-augmented intelligence system designed to answer plain-language questions across complex MongoDB schemas. The system converts raw documents into natural language, resolves relational references across collections without hardcoded schemas, embeds chunks into **Qdrant Vector DB**, performs hybrid semantic + keyword search with multi-hop graph expansion, executes deterministic structured queries for aggregate/list calculations, and synthesizes answers via LLMs (Google Gemini, OpenAI, Groq, Ollama) while enforcing strict role-based access control and data privacy.

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                            │
│  python manage.py ingest  /  POST /api/v1/ingest/                      │
├────────────────────────────────────────────────────────────────────────┤
│  MongoDB Collections (Dynamic Discovery: users, boards, cards, etc.)   │
│    │                                                                   │
│    ├──► Relationship Resolver (Probes & caches foreign ObjectId links) │
│    ├──► Universal Converter (Document -> Human text + Reference names) │
│    ├──► Smart Chunking (Fits token budget, preserves line boundaries)  │
│    ├──► Content Hash Check (Skips unchanged chunks -> incremental)     │
│    ├──► Embeddings (FastEmbed / all-MiniLM-L6-v2 or OpenAI, normalized)│
│    └──► Qdrant Vector Store (Point: vector + payload + relationships)  │
│         └──► Automatic Payload Index Creation (Self-healing schema)    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                           ASK / QUERY PIPELINE                         │
│  POST /api/v1/ask/                                                     │
├────────────────────────────────────────────────────────────────────────┤
│  User Question + Authentication Context (user_id, email, workspace_id) │
│    │                                                                   │
│    ├──► 1. Conversational & Guardrail Check (Greetings, capabilities)  │
│    │                                                                   │
│    ├──► 2. Personal Work Routing (Direct lookup for current user tasks)│
│    │                                                                   │
│    ├──► 3. Canonical Leave Requests Engine                             │
│    │      • Directly queries `leaverequests` + `leavetypes`            │
│    │      • Reconciles requester -> users, leaveType -> leavetypes     │
│    │      • Date filtering, approval status & superlative aggregates   │
│    │      • Excludes administrative template "Leave" board & cards     │
│    │                                                                   │
│    ├──► 4. Structured Query Planner (Lists & exact counts)             │
│    │      • Plans catalog query -> strictly validated JSON filter      │
│    │      • Direct MongoDB `find` / `count_documents` execution        │
│    │      • Enriches project records with exact linked members         │
│    │                                                                   │
│    ├──► 5. Hybrid Semantic Retrieval (When semantic search is needed)  │
│    │      • Qdrant cosine vector search + keyword payload match        │
│    │      • Reciprocal Rank Fusion (RRF) reranking                     │
│    │      • Multi-Hop expansion (follows forward/reverse relationships)│
│    │      • Context Sanitization (scrubs raw IDs, resolves user names) │
│    │      • LLM Answer Generation (Gemini 3.5 / OpenAI / Ollama)       │
│    │                                                                   │
│    └──► 6. Dynamic ID Sanitization & Privacy Shield                    │
│           • Resolves 24-hex ObjectIds against `users` collection       │
│           • Replaces "user with ID <id>" with real team member names   │
│           • Strips any dangling or non-user hex IDs completely         │
│           • Sanitizes punctuation artifacts (empty commas, brackets)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Capabilities & Intelligent Engines

### A. Canonical Leave Management Engine
- **Direct Collection Binding**: Employee leave records are queried directly from the **`leaverequests`** collection (joined with `users` for requester names and `leavetypes` for policy categories such as *Full Day Leave*, *Short Leave*, *Half Day Leave*).
- **Template Board Separation**: Administrative template boards named `"Leave"` and their placeholder cards (`Short leave`, `Halfday Leave`, `Full leave`) are strictly isolated from operational projects and tasks.
- **Natural Date & Status Parsing**:
  - Specific day lookups (e.g. `who is on leave on 13 nov` -> filters approved leaves across `00:00:00` to `23:59:59.999999` boundaries).
  - Status classification (`APPROVED`, `PENDING`, `REJECTED`, `CANCELLED_BY_REQUESTER`, `CANCELLED_BY_ADMIN`).
  - Superlative aggregates (e.g. `who has least leaves in oct`, `who takes most leaves`).

### B. Dynamic ID Sanitization & Privacy Shield
- **Zero Raw ID Exposure**: Raw database ObjectIds (such as `68f0d3a8445a827910e3956d`) are never displayed to the frontend.
- **Automated Name Resolution**: If a user ID appears in relationships or text, `_sanitize_ids` dynamically queries the `users` collection to resolve it to the user's full name (e.g., *"Rohit Gupta"*).
- **Context Scrubbing**: Context fed into LLM prompts is cleaned via `_clean_context_ids`, ensuring the model never sees raw hex IDs for owners, assignees, or members.
- **Clean Team Summaries**: Project members and assignees are filtered to valid user records, eliminating empty commas like `(, , Anil Bansal, ...)` and presenting only clean, human-readable name rosters.

### C. Self-Healing Vector Store (Qdrant)
- **Automatic Payload Indexing**: On startup, `embedding_store.ensure_indexes()` verifies and creates required Qdrant payload schema indexes (including integer index on `metadata.chunk_index`, keyword index on `metadata.workspace_id`, and `reference_ids`).
- **Resilient Multi-Hop Expansion**: Reverse relationship lookups (`find_referencing_batch`) run inside non-fatal handlers, allowing multi-hop queries to fall back gracefully to primary search hits without aborting user chat turns.

### D. Structured Query Planner
- **Exact Counts & Deterministic Lists**: Queries such as *"list all pending projects"*, *"how many departments are there"*, or *"unique members on Web Chat Pingly"* are translated into strict MongoDB queries rather than approximate similarity retrieval.
- **Strict Operator Whitelist**: Supported operations (`eq`, `ne`, `in`, `contains`, `gt`, `gte`, `lt`, `lte`) prevent prompt injections while ensuring 100% precision for counts and status filters.

---

## 3. Database Dependencies

The system integrates two complementary data tiers:

| Database | Technology | Primary Role in Chatbot |
|---|---|---|
| **Operational Store** | **MongoDB Atlas** | Single source of truth for business data: `users`, `boards` (projects), `cards` (tasks), `subtasks`, `leaverequests`, `leavetypes`, `workspaces`, `departments`. Powers exact lists, status checks, leave balances, logged time, and permission checks. |
| **Vector Store** | **Qdrant (Cloud / Local)** | Semantic index storing dense vector embeddings and chunk payloads (`rag_chunks` collection). Powers conceptual similarity matching, unstructured search, hybrid keyword matching, and multi-hop relationship traversals. |

---

## 4. Quick Start

### 1. Environment Setup
```powershell
# Create and activate virtual environment
cd MONGO_CHATBOT
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env` and verify your settings:
```ini
# MongoDB
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGO_DATABASE=flowtask
MONGO_MAX_POOL_SIZE=10

# Qdrant Vector DB
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION=rag_chunks

# LLM Provider (Google Gemini, OpenAI, or Ollama)
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your-gemini-api-key>
LLM_MODEL=gemini-3.5-flash-lite

# Retrieval & Hybrid Settings
STRUCTURED_QUERIES=true
HYBRID_SEARCH=true
TOP_K=6
LIST_TOP_K=30
MAX_HOPS=1
```

### 3. Run Ingestion & Server
```powershell
# Ingest data from MongoDB into Qdrant (creates points and schema indexes)
python manage.py ingest

# Start Django development server
python manage.py runserver
```

---

## 5. REST API Reference

The REST API is versioned under `/api/v1/` and accepts standard JSON payloads.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ask/` | Executes full RAG pipeline or structured query and returns natural language answer. |
| `POST` | `/api/v1/search/` | Performs semantic + hybrid search over indexed records without LLM synthesis. |
| `GET` | `/api/v1/chats/` | Lists chat sessions for the current authenticated user/session. |
| `GET` | `/api/v1/chats/{session_id}/` | Retrieves chat history turns for a specific session. |
| `POST` | `/api/v1/chats/{session_id}/rename/` | Updates session title. |
| `DELETE` | `/api/v1/chats/{session_id}/delete/` | Deletes a chat session and clears its memory. |
| `GET` | `/api/v1/health/` | Health check for MongoDB, Qdrant, embeddings, and LLM provider. |
| `POST` | `/api/v1/ingest/` | Triggers background data ingestion (requires `INGEST_TOKEN`). |

### Example: `/api/v1/ask/` Request
```json
{
  "question": "who is on leave on 13 nov",
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "user_id": "69ce2134b97c97fe612de9be",
  "email": "employee@example.com",
  "workspace_id": "6a759a7611d309503d9cee15",
  "top_k": 6,
  "hops": 1
}
```

### Example Response
```json
{
  "question": "who is on leave on 13 nov",
  "answer": "Team members on approved leave on 13 Nov 2026:\n\n- **Anil Bansal** (Anil Bansal | Short Leave (1 day(s), 13 Nov 2026) — Status: `Approved` — Reason: Going Home)",
  "sources": [
    {
      "collection": "leaverequests",
      "document_id": "6af5b77dc717f79471cd8fbb",
      "name": "Anil Bansal",
      "score": null,
      "related": false
    }
  ],
  "cached": false,
  "mode": "structured",
  "collection": "leaverequests",
  "total_matching": 1,
  "timings_ms": {
    "db": 12,
    "llm": 0,
    "total": 14
  }
}
```

---

## 6. Access Control & Permissions

- **Admin vs Non-Admin**:
  - Users with `role: "admin"` in MongoDB have organization-wide visibility across all projects, cards, and team summaries in their workspace.
  - Non-admin users (`role: "employee"`) have scoped access to their assigned work and publicly queryable status (e.g. organization out-of-office calendar).
- **Workspace Scoping**: Every request must be tied to a valid `workspace_id`. Membership is strictly verified before data retrieval begins.
- **Privacy Enforcement**: Non-admins are prohibited from retrieving sensitive private leave records or private work of other employees unless querying shared team calendar status.

---

## 7. Automated Testing & Verification

Run automated test suites to verify functionality:
```powershell
# Run Django unit test suite
python manage.py test

# Verify specific leave queries and date boundaries
python test_specific_leave_queries.py
```

---

## 8. Codebase Structure

```
MONGO_CHATBOT/
├── config/
│   ├── settings.py           # Django configuration, environment loading, logging
│   ├── urls.py               # Root URL router (/api/v1/ routes)
│   └── asgi.py / wsgi.py     # Server gateway interfaces
├── rag/
│   ├── services/
│   │   ├── rag_service.py           # Core orchestrator: question routing, context building, asking
│   │   ├── structured_query.py      # Structured planner, leave engine, project enrichers, ID sanitizer
│   │   ├── document_converter.py    # Document -> natural language text converter, field sanitizers
│   │   ├── embedding_store.py       # Qdrant client, self-healing payload indexes, vector search
│   │   ├── embedding_service.py     # Local ONNX FastEmbed & hosted embedding generation
│   │   ├── mongo_service.py         # MongoDB connection pooling, batch readers, document fetchers
│   │   ├── llm_service.py           # Model integrations (Gemini, OpenAI, Ollama), system prompts
│   │   ├── authorization_service.py # User scope verification, workspace membership checks
│   │   ├── retrieval.py             # Hybrid search (RRF fusion), multi-hop relation expansion
│   │   ├── chat_memory.py           # Multi-turn memory management & edit index tracking
│   │   └── cache_service.py         # In-memory TTL caches for query plans and answers
│   ├── models.py                    # ChatSession and ChatMessage SQLite models
│   ├── views.py                     # DRF REST API viewsets and controllers
│   ├── apps.py                      # App initialization & startup warm-up
│   └── tests.py                     # Unit and integration test suite
├── static/                          # Glass UI styling and chat scripts
├── templates/                       # Frontend HTML templates
├── manage.py                        # Django CLI entrypoint
└── requirements.txt                 # Production dependencies
```
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
  -> MongoDB (business collections)
  -> Qdrant (chunk vectors + text/metadata payloads)
  -> LLM Provider (Ollama / OpenAI / Groq)

## Main collections used
activities, boards, cards, departments, leave_entries, memberships, notes, projects, subtasks, tasks, users, workspaces

Primary collection is cards (tasks + nested loggedTime entries), used by both structured queries and RAG. users is used for identity/permissions. 
Qdrant collection `rag_chunks` holds vectors and their text/metadata payloads.

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

## Safety warning system
- The same request guard runs before chat generation and search retrieval for every account, including admins.
- It blocks natural-language requests to delete, edit, bulk-change, or alter access/security settings, and requests to expose credentials or sensitive employee data.
- Blocked responses include `blocked: true`, a warning category, and a user-facing message; no business-data query or LLM call is made for that request.
- Conceptual questions about these operations remain available. The chatbot has no business-record write tool; approved audited workflows remain the place to make authorized changes.

## Ingestion
- Documents of any schema converted to natural language.
- ObjectId relationships detected and resolved.
- Embeddings and chunk payloads stored in Qdrant; source records remain in MongoDB.
- Run: python manage.py ingest

## Logged-time feature (recently enhanced)
- Reads nested loggedTime entries on cards.
- Supports exact date, month, year, week scoping.
- Superlative queries: "who has highest/lowest logged time in September / 2026 / this week".
- Team-level ranking aggregates per person and names the top contributor.
- Period resolution: _extract_period handles bare months, bare years, "this week/month/year", exact dates.

## Configuration
- .env (from .env.example) for local Ollama profile.
- MONGO_URI, MONGO_DATABASE and QDRANT_URL/QDRANT_API_KEY required (or configure QDRANT_PATH for local embedded mode).
- LLM_PROVIDER: ollama (default local) / openai / groq.
- CORS via DJANGO_CORS_ALLOWED_ORIGINS.

## Running locally
1. copy .env.example .env
2. set MongoDB and Qdrant settings
3. python -m pip install -r requirements.txt
4. python manage.py ingest --full
5. python manage.py runserver
6. (Ollama) ollama pull gemma3:4b

## Tech stack
Django, Django REST Framework, MongoDB (pymongo), Qdrant, sentence-transformers, Ollama/OpenAI/Groq, mongomock (tests).
