# Graph Report - MONGO_CHATBOT  (2026-11-06)

## Corpus Check
- 55 files · ~109,990 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: (none) 3, .css 1, .faiss 1)

## Summary
- 882 nodes · 2133 edges · 69 communities (38 shown, 31 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 87 edges (avg confidence: 0.91)
- Token cost: 6,120 input · 1,980 output

## Community Hubs (Navigation)
- Django Core
- Chat Frontend
- LLM Service
- Embedding Store
- API Auth Views
- FAISS Index
- Ask Tests
- Management Commands
- Retrieval Search
- Structured Query
- RAG Service
- Query Correction
- API Tests
- Relationship Detection
- LLM Errors
- Check LLM Command
- Exact Answers
- HTML Parsing
- Exact Answers
- Authorization
- Auth Views
- Schema Catalog
- Embeddings Store
- Relationship Resolver
- Logged Time Tests
- Test Infra
- Schema Catalog
- Ingestion Tests
- Django Config
- Logged Time
- Leave Answers
- Document Converter
- Django App Config
- MongoDB Service
- Embedding Service
- Resilience Tests
- Query Cache
- Collections Cache
- Hybrid Retrieval
- Django Views
- Relationship Detection
- Anonymous Tests
- Search Tests
- Mongo Access
- Public Access Tests
- API Serializers
- Ingest API
- Logged Time
- Chunking
- Document Conversion
- Document Walker
- System Overview
- HTML Parsing
- Sensitive Fields
- RAG Context
- Gemini Tests
- Tech Stack
- Chatbot Docs
- Django Command
- Django Command
- LLM Provider
- Retrieval Context
- Logged Time Detection
- Dev Requirements
- Gemma3 Model

## God Nodes (most connected - your core abstractions)
1. `ask()` - 77 edges
2. `ChatbotError` - 38 edges
3. `StructuredQueryTests` - 33 edges
4. `get_db()` - 27 edges
5. `run_ingestion()` - 26 edges
6. `AskTests` - 26 edges
7. `_logged_time_answer()` - 24 edges
8. `ConfigError` - 23 edges
9. `MongoServiceError` - 22 edges
10. `try_structured()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `_error_response()` --uses--> `ChatbotError`  [INFERRED]
  rag/api_views.py → rag/services/exceptions.py
- `AskView` --uses--> `ChatbotError`  [INFERRED]
  rag/api_views.py → rag/services/exceptions.py
- `SearchView` --uses--> `ChatbotError`  [INFERRED]
  rag/api_views.py → rag/services/exceptions.py
- `HealthView` --uses--> `ChatbotError`  [INFERRED]
  rag/api_views.py → rag/services/exceptions.py
- `ChatListView` --uses--> `ChatbotError`  [INFERRED]
  rag/api_views.py → rag/services/exceptions.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **RAG retrieval stack** — readme_mongodb, readme_faiss, readme_ollama, readme_hybrid_search [INFERRED 0.75]

## Communities (69 total, 31 thin omitted)

### Community 0 - "Django Core"
Cohesion: 0.07
Nodes (29): get_owner(), owner_session_qs(), _log(), request_log_middleware(), middleware(), Migration, Migration, ChatMessage (+21 more)

### Community 1 - "Chat Frontend"
Cohesion: 0.16
Nodes (33): addBot(), addTyping(), addUser(), api(), ask(), authHeaders(), beginEditMessage(), beginRename() (+25 more)

### Community 2 - "LLM Service"
Cohesion: 0.10
Nodes (10): _system_prompt(), warm_up(), fake_ollama(), factory(), OllamaClientTests, down(), handler(), handler() (+2 more)

### Community 3 - "Embedding Store"
Cohesion: 0.16
Nodes (16): collection(), count(), delete_collections(), delete_ids(), distinct_models(), ensure_indexes(), existing_hashes(), find_referencing() (+8 more)

### Community 4 - "API Auth Views"
Cohesion: 0.15
Nodes (10): RenameChatSerializer, AskView, ChatDeleteView, ChatDetailView, ChatListView, ChatRenameView, HealthView, _set_chat_owner() (+2 more)

### Community 5 - "FAISS Index"
Cohesion: 0.18
Nodes (9): VectorStoreError, _bitmap_type(), current_version(), _faiss(), get_index(), rebuild_index(), reset_for_tests(), _stamp_now() (+1 more)

### Community 6 - "Ask Tests"
Cohesion: 0.14
Nodes (4): ask(), is_list_question(), validate_question(), AskTests

### Community 7 - "Management Commands"
Cohesion: 0.17
Nodes (6): Command, ChatbotError, EmptyDatabaseError, IngestionBusyError, IngestionError, start_job()

### Community 8 - "Retrieval Search"
Cohesion: 0.15
Nodes (8): search(), blocked_collections(), is_blocked_collection(), expand(), Hit, search(), _to_hit(), BlockedActivityCollectionTests

### Community 9 - "Structured Query"
Cohesion: 0.13
Nodes (11): _assigned_work_filter(), _context(), _count(), _employee_name_from_question(), _employee_work_status(), _exec_resolver(), _fallback_assigned_work(), _person_lookup() (+3 more)

### Community 10 - "RAG Service"
Cohesion: 0.11
Nodes (9): check_company_identity(), check_conversational(), check_self_identity(), _group(), _multi_topic_leave_question(), _person_name_from_company_question(), _prior_list_reference(), _source() (+1 more)

### Community 11 - "Query Correction"
Cohesion: 0.15
Nodes (10): _best_spelling(), clear_vocabulary_cache(), correct_query(), repair_spelling(), _distance_within(), expand_query(), _load_vocabulary(), _masked_candidates() (+2 more)

### Community 13 - "Relationship Detection"
Cohesion: 0.17
Nodes (12): strip_id_suffix(), guess_collections(), _norm(), _singular(), catalog_text(), render(), sort_key(), get_aliases() (+4 more)

### Community 14 - "LLM Errors"
Cohesion: 0.19
Nodes (11): ConfigError, LLMError, _chat(), _get_client(), _history_text(), _http(), _ollama_chat(), _ollama_post() (+3 more)

### Community 15 - "Check LLM Command"
Cohesion: 0.21
Nodes (5): generate(), CheckLlmCommandTests, fake_openai_client(), GeminiTests, provider_error()

### Community 16 - "Exact Answers"
Cohesion: 0.12
Nodes (8): _exact_answer(), _field_from_text(), _format_board_answer(), add_bullet(), _meaningful_field(), _plain_answer(), _strip_blank_bullet_lines(), _unique_records()

### Community 17 - "HTML Parsing"
Cohesion: 0.20
Nodes (13): strip_html(), _assignee_board_match(), _assignee_card_match(), _assignee_count_answer(), _assignee_identity(), _assignee_ids(), _assignee_summary(), _assignee_target() (+5 more)

### Community 19 - "Authorization"
Cohesion: 0.17
Nodes (7): AccessScope, _id_values(), mentions_another_user(), mentions_current_user(), resolve_scope(), AuthorizationError, search()

### Community 21 - "Schema Catalog"
Cohesion: 0.17
Nodes (9): build_catalog(), _catalog_path(), _read(), _refresh_in_background(), work(), reset_for_tests(), _stamp(), validate_plan() (+1 more)

### Community 22 - "Embeddings Store"
Cohesion: 0.15
Nodes (9): get_db(), get_document(), _condition(), _ids_by_name(), _logged_time_by_project(), _logged_time_for_project(), _project_team_details(), _regex_exact() (+1 more)

### Community 26 - "Schema Catalog"
Cohesion: 0.20
Nodes (10): _Acc, Catalog, _collect(), CollectionProfile, FieldInfo, get_catalog(), _kind(), _profile() (+2 more)

### Community 28 - "Django Config"
Cohesion: 0.17
Nodes (4): _bool(), _float(), _int(), _list()

### Community 29 - "Logged Time"
Cohesion: 0.19
Nodes (9): _id_variants(), is_logged_time_followup(), is_logged_time_question(), _logged_time_answer(), _logged_time_project_candidate(), _logged_time_user_names(), _mentions_self(), _personal_logged_time_question() (+1 more)

### Community 30 - "Leave Answers"
Cohesion: 0.18
Nodes (9): ExactResult, _format_duration(), _leave_board_ids(), _leave_time_answer(), _leave_time_empty(), _leave_type_from_question(), _leave_types_answer(), _leave_types_empty() (+1 more)

### Community 31 - "Document Converter"
Cohesion: 0.21
Nodes (5): convert_document(), Converted, _format_datetime(), pick_display_name(), _summary()

### Community 32 - "Django App Config"
Cohesion: 0.18
Nodes (3): RagConfig, _warm_up(), warm_up()

### Community 33 - "MongoDB Service"
Cohesion: 0.23
Nodes (6): _get_client(), get_embeddings_collection(), iter_batches(), list_collections(), ping(), _require()

### Community 34 - "Embedding Service"
Cohesion: 0.27
Nodes (6): embed_query(), embed_texts(), _encode_local(), _encode_openai(), get_model(), EmbeddingError

### Community 39 - "Django Views"
Cohesion: 0.33
Nodes (4): aget_chat_owner(), get_chat_owner(), _owner_id(), index()

### Community 40 - "Relationship Detection"
Cohesion: 0.25
Nodes (3): iter_ref_candidates(), walk(), RelationshipTests

### Community 45 - "API Serializers"
Cohesion: 0.29
Nodes (4): AskRequestSerializer, IngestRequestSerializer, SearchRequestSerializer, SearchView

### Community 46 - "Ingest API"
Cohesion: 0.48
Nodes (4): _error_response(), IngestView, AuthError, job_status()

### Community 47 - "Logged Time"
Cohesion: 0.29
Nodes (5): humanize_key(), execute(), _labels_for(), _select_lines(), _without_logged_time()

### Community 51 - "System Overview"
Cohesion: 0.40
Nodes (6): Mongo RAG Pro, FAISS, Hybrid search, MongoDB, Ollama, Structured query

### Community 53 - "Sensitive Fields"
Cohesion: 0.40
Nodes (3): is_sensitive(), _build(), _scalars()

### Community 54 - "RAG Context"
Cohesion: 0.40
Nodes (3): build_context(), add(), _without_logged_time_text()

### Community 56 - "Tech Stack"
Cohesion: 0.40
Nodes (5): Django, Requirements, FAISS, PyMongo, Sentence Transformers

### Community 57 - "Chatbot Docs"
Cohesion: 0.40
Nodes (5): Chat View, Mongo RAG Chatbot, Mongo RAG Chatbot Interface, Index View, Search View

## Knowledge Gaps
- **14 isolated node(s):** `MongoDB`, `Ollama`, `Gemma3 4B`, `FAISS`, `Django` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 252 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **31 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ask()` connect `Ask Tests` to `Django Core`, `LLM Service`, `API Auth Views`, `FAISS Index`, `Retrieval Search`, `Structured Query`, `RAG Service`, `Query Correction`, `Relationship Detection`, `Check LLM Command`, `Exact Answers`, `Exact Answers`, `Authorization`, `Logged Time Tests`, `Ingestion Tests`, `Logged Time`, `Resilience Tests`, `Search Tests`, `RAG Context`, `Gemini Tests`, `Retrieval Context`, `Logged Time Detection`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `ChatbotError` (e.g. with `AskView` and `ChatDeleteView`) actually correct?**
  _`ChatbotError` has 17 INFERRED edges - model-reasoned connections that need verification._
- **What connects `MongoDB`, `Ollama`, `Gemma3 4B` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Django Core` be split into smaller, more focused modules?**
  _Cohesion score 0.07180851063829788 - nodes in this community are weakly interconnected._
- **Why does `ChatbotError` connect `Management Commands` to `Django App Config`, `Embedding Service`, `Embedding Store`, `API Auth Views`, `FAISS Index`, `Structured Query`, `RAG Service`, `API Serializers`, `Ingest API`, `LLM Errors`, `Relationship Detection`, `HTML Parsing`, `Authorization`, `Django Command`, `Django Command`, `LLM Provider`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `StructuredQueryTests` (e.g. with `LLMError` and `Hit`) actually correct?**
  _`StructuredQueryTests` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `LLM Service` be split into smaller, more focused modules?**
  _Cohesion score 0.10344827586206896 - nodes in this community are weakly interconnected._