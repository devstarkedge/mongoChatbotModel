# Mongo RAG Pro: System Architecture & Component Wiring

This document provides a comprehensive blueprint of how the Mongo RAG Pro Chatbot functions, including architectural wiring, execution sequence, database relationships, and component interactions.

---

## 1. End-to-End Chatbot Execution & Wiring Diagram

The diagram below maps the complete path of an incoming query from the client, through authentication, routing, deterministic engines, vector retrieval, and LLM answer generation:

```mermaid
flowchart TD
    Client["Client / User / Frontend\n(React UI, REST API, or CLI)"]
    
    subgraph Django_Layer["Django & HTTP Gateway"]
        WSGI["Gunicorn / WSGI / WhiteNoise"]
        MW["Middleware Stack\n- CorsMiddleware\n- SecurityMiddleware\n- request_log_middleware\n- AuthenticationMiddleware"]
        Urls["urls.py Router\n- / (UI)\n- /api/v1/ask/ & /ask/\n- /health/\n- /api/v1/search/"]
        AskView["AskView (api_views.py)"]
    end

    subgraph Decision_Engine["RagService & Structured Routing (rag_service.py)"]
        Guardrails{"1. Conversational?\n(Greeting / Help)"}
        PersonalCheck{"2. Personal Work?\n(My tasks / work)"}
        LeaveCheck{"3. Leave Query?\n(Who on leave / balance)"}
        TeamCheck{"4. Project / Team Query?\n(Team size / who works on X)"}
        WorkStatusCheck{"5. Employee Task Query?\n(Shubham uncompleted tasks)"}
        PlannerCheck{"6. Structured Aggregation?\n(Counts / list / totals)"}
        HybridRAG["7. Hybrid RAG Pipeline\n(Conceptual search / fallback)"]
    end

    subgraph Data_Stores["Data Storage & Indexing Tier"]
        MongoDB[("MongoDB Atlas\n- boards (Projects)\n- cards (Tasks)\n- users (Employees)\n- departments\n- leaverequests\n- leavetypes")]
        Qdrant[("Qdrant Vector DB\n- Collection: rag_chunks\n- Payload schema indexes\n- Dense embeddings (384-dim)")]
    end

    subgraph AI_Synthesis["AI Synthesis & Privacy Shield"]
        FastEmbed["Embedding Service\n(all-MiniLM-L6-v2)"]
        MultiHop["Multi-Hop Graph Resolver\n(Forward & Reverse ObjectIds)"]
        LLM["LLM Gateway (llm_service.py)\n- Primary: Google Gemini\n- Fallback: Groq (gpt-oss-120b)"]
        Sanitizer["Dynamic ID Sanitizer\n(Zero raw hex ObjectIds exposed)"]
    end

    %% Wiring connections
    Client -->|"HTTP POST /ask/"| WSGI
    WSGI --> MW
    MW --> Urls
    Urls --> AskView
    AskView --> Guardrails

    Guardrails -->|"Yes"| Sanitizer
    Guardrails -->|"No"| PersonalCheck

    PersonalCheck -->|"Yes"| MongoDB
    PersonalCheck -->|"No"| LeaveCheck

    LeaveCheck -->|"Yes"| MongoDB
    LeaveCheck -->|"No"| TeamCheck

    TeamCheck -->|"Yes"| MongoDB
    TeamCheck -->|"No"| WorkStatusCheck

    WorkStatusCheck -->|"Yes"| MongoDB
    WorkStatusCheck -->|"No"| PlannerCheck

    PlannerCheck -->|"Planned Filter"| MongoDB
    PlannerCheck -->|"No / Semantic"| HybridRAG

    HybridRAG --> FastEmbed
    FastEmbed --> Qdrant
    Qdrant -->|"Vector Hits"| MultiHop
    MongoDB -->|"Text Match ($text)"| MultiHop
    MultiHop --> LLM

    MongoDB -->|"Resolved Names"| Sanitizer
    LLM --> Sanitizer
    Sanitizer -->|"Clean Markdown Answer"| Client
```

---

## 2. Request / Response Lifecycle Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Client
    participant GW as Django Gateway (urls & AskView)
    participant SQ as Structured Query Engine (structured_query.py)
    participant DB as MongoDB Atlas (Operational Store)
    participant QD as Qdrant (Vector DB)
    participant LLM as LLM Service (Gemini / Groq)
    participant PS as Privacy Sanitizer

    User->>GW: POST /api/v1/ask/ {"question": "shubham's uncompleted tasks"}
    GW->>SQ: try_structured(question, user_context)
    
    rect rgb(240, 248, 255)
        Note over SQ: Step 1: Detect intent & extract entities
        SQ->>SQ: Name: "Shubham" (strips status "uncompleted")
        SQ->>DB: _ids_by_name("users", "Shubham")
        DB-->>SQ: [ObjectId('695b4fd0...')] (Shubham Dhatwalia)
        
        Note over SQ: Step 2: Query exact tasks
        SQ->>DB: cards.find({"assignees": uid, "status": {"$nin": ["done","completed"]}})
        DB-->>SQ: [8 matching task documents]
        
        Note over SQ: Step 3: Resolve Board / Project names
        SQ->>DB: boards.find({"_id": {"$in": board_ids}})
        DB-->>SQ: [board name mappings]
        
        SQ->>SQ: Build direct_answer markdown
    end

    alt Deterministic Match Found
        SQ-->>GW: ExactResult (direct_answer, total=8)
    else Needs Semantic RAG
        GW->>QD: Vector search (cosine >= 0.35)
        QD-->>GW: Chunks & reference ObjectIds
        GW->>DB: Multi-hop graph expansion & $text search
        DB-->>GW: Related entities & context
        GW->>LLM: generate(question, context)
        LLM-->>GW: Raw generated answer
    end

    GW->>PS: sanitize(answer)
    PS->>DB: Resolve any remaining ObjectIds to user names
    DB-->>PS: User display names
    PS-->>GW: Clean markdown without raw ObjectIds
    GW-->>User: 200 OK {"answer": "...", "sources": [...]}
```

---

## 3. Database Schema & Relational Wiring

The diagram below illustrates how data is related across collections in MongoDB and mirrored into Qdrant for semantic search:

```mermaid
erDiagram
    USERS ||--o{ BOARDS : "owns / member of"
    DEPARTMENTS ||--o{ USERS : "contains"
    DEPARTMENTS ||--o{ BOARDS : "governs"
    BOARDS ||--o{ CARDS : "contains (tasks)"
    BOARDS ||--o{ LISTS : "contains (columns)"
    CARDS }o--o{ USERS : "assigned to"
    CARDS ||--o{ LOGGED_TIME : "tracks"
    USERS ||--o{ LEAVE_REQUESTS : "submits"
    LEAVE_TYPES ||--o{ LEAVE_REQUESTS : "categorizes"
    CARDS ||--o{ QDRANT_CHUNKS : "embedded into"
    BOARDS ||--o{ QDRANT_CHUNKS : "embedded into"

    USERS {
        ObjectId _id PK
        string name
        string email
        ObjectId department FK
        string role
    }

    DEPARTMENTS {
        ObjectId _id PK
        string name
        ObjectId[] members FK
        ObjectId[] managers FK
    }

    BOARDS {
        ObjectId _id PK
        string name
        ObjectId department FK
        ObjectId owner FK
        ObjectId[] members FK
        string status
        boolean isArchived
    }

    CARDS {
        ObjectId _id PK
        string title
        ObjectId board FK
        ObjectId list FK
        ObjectId[] assignees FK
        string status
        loggedTime[] loggedTime
    }

    LEAVE_REQUESTS {
        ObjectId _id PK
        ObjectId user FK
        ObjectId leaveType FK
        date startDate
        date endDate
        string status
    }

    LEAVE_TYPES {
        ObjectId _id PK
        string name
    }

    QDRANT_CHUNKS {
        string id PK
        float384 vector
        string text
        string collection
        string document_id
        string[] reference_ids
    }
```

---

## 4. Architectural Component Reference

### 1. Gateway & Routing (`config/urls.py`, `rag/urls.py`, `rag/api_views.py`)
- **`AskView` (`/api/v1/ask/`, `/ask/`)**: The primary API entry point for user chat questions. Accepts JSON payloads with session IDs, query strings, and optional workspace tokens.
- **`HealthView` (`/health/`, `/api/v1/health/`)**: Uptime and health check endpoint returning system diagnostics (MongoDB status, vector collection count, active LLM provider).
- **`request_log_middleware` (`rag/middleware.py`)**: Intercepts requests, measures response latencies, and automatically strips body bytes on `HEAD` requests to ensure RFC 9110 compliance.

### 2. Structured Query Engine (`rag/services/structured_query.py`)
- **Deterministic Accuracy**: Direct MongoDB queries bypass LLMs for mathematical, count, status, or date-sensitive operations.
- **Canonical Leave Engine**: Resolves employee leave requests against `leaverequests` and `leavetypes` with precise calendar filtering.
- **Team & Department Linkage**: Resolves full project hierarchies: `board -> department -> user roster`, active contributors, and task assignees.
- **Employee Task Resolver**: Extracts employee names and status modifiers (`uncompleted`, `pending`, `done`) and retrieves assigned cards.

### 3. Vector Search & Graph Expansion (`rag/services/vector_index.py`, `rag/services/rag_service.py`)
- **Qdrant Vector Store**: Dense vector matching with auto-healing payload indexes.
- **Reciprocal Rank Fusion (RRF)**: Merges dense vector semantic results with MongoDB sparse text search (`$text`).
- **Multi-Hop Graph Traversals**: Follows foreign keys (`board`, `assignees`, `department`) forward and reverse to assemble complete relational context.

### 4. LLM Service (`rag/services/llm_service.py`)
- **Multi-Provider Engine**:
  - Primary: Google Gemini (`gemini-3.5-flash-lite` via OpenAI-compatible endpoint).
  - Fallback: Groq (`gpt-oss-120b`).
- **Strict Anti-Hallucination Prompts**: Returns `NOT_FOUND_ANSWER` when evidence is missing.

### 5. Privacy & Sanitization Shield (`rag/services/structured_query.py`)
- **Zero Hex ID Policy**: Raw 24-character ObjectIds (`68f0d3a8...`) are matched against `users` and replaced with human names (e.g. `Vishal Sharma`). Any unmapped IDs are removed before returning to the client.
