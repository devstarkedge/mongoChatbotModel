"""
Run with:  python manage.py test
Needs no real MongoDB, internet or API key: MongoDB is mongomock, embeddings are a
deterministic hashing embedder, and the LLM is faked.
"""
import json
import re
import tempfile
import zlib
from unittest.mock import patch

import mongomock
import numpy as np
import jwt
from bson import ObjectId
from django.test import Client, SimpleTestCase, override_settings
from rest_framework.test import APIClient

from rag.management.commands.seed_sample_data import SAMPLE
from rag.services import (authorization_service, document_converter, embedding_service, embedding_store,
                          ingestion_service, llm_service, mongo_service, rag_service, relationships, retrieval,
                          structured_query, vector_index)
from rag.services.exceptions import AuthorizationError, InvalidRequestError, VectorStoreError
from rag.services.retrieval import rrf_fuse


def fake_embed_texts(texts, batch_size=64):
    """Bag-of-words hashing embedder: texts sharing words get higher cosine similarity."""
    out = np.zeros((len(texts), 128), dtype="float32")
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if re.fullmatch(r"[0-9a-f]{24}", token):      # random ObjectIds are noise: keep the fake embedder deterministic
                continue
            out[i, zlib.crc32(token.encode()) % 128] += 1
        out[i] /= np.linalg.norm(out[i]) or 1
    return out


def _mongomock_add_replace_shim():
    """Newer pymongo passes `sort=` to the bulk builder; older mongomock doesn't accept it (test-only gap)."""
    from mongomock.collection import BulkOperationBuilder

    original = BulkOperationBuilder.add_replace

    def add_replace(self, selector, replacement, upsert=False, collation=None, hint=None, sort=None):
        return original(self, selector, replacement, upsert=upsert, collation=collation, hint=hint)

    return patch.object(BulkOperationBuilder, "add_replace", add_replace)


@override_settings(SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies")
class AnonymousUiTests(SimpleTestCase):
    def test_chatbot_is_open_and_login_routes_are_removed(self):
        client = Client()
        self.assertEqual(client.get("/").status_code, 200)
        for path in ("/login/", "/logout/", "/api/v1/auth/token/", "/api/v1/auth/me/"):
            self.assertEqual(client.get(path).status_code, 404, path)


class MongoCase(SimpleTestCase):
    """Base: fresh in-memory MongoDB with the sample data, fake embedder, temp index dir."""

    structured = False     # exact-query path off by default so the similarity-search tests stay focused

    def setUp(self):
        self.db = mongomock.MongoClient()["testdb"]
        for name, docs in SAMPLE.items():
            self.db[name].insert_many([dict(d) for d in docs])
        self.db["users"].update_one({"email": "meera@example.com"}, {"$set": {"role": "admin"}})
        self.admin_id = str(self.db["users"].find_one({"email": "meera@example.com"})["_id"])
        self.real_resolve_scope = authorization_service.resolve_scope
        self.scope_patch = patch.object(
            authorization_service, "resolve_scope",
            side_effect=lambda user_id, email=None: self.real_resolve_scope(user_id or self.admin_id, email=email),
        )
        self.scope_patch.start()
        self.addCleanup(self.scope_patch.stop)
        tmp = tempfile.mkdtemp()
        patches = [
            patch("rag.services.mongo_service.get_db", return_value=self.db),
            patch("rag.services.mongo_service.get_embeddings_collection", return_value=self.db["rag_embeddings"]),
            patch("rag.services.mongo_service.ping", return_value=True),
            patch("rag.services.embedding_service.embed_texts", side_effect=fake_embed_texts),
            _mongomock_add_replace_shim(),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        overrides = override_settings(VECTOR_STORE_DIR=tmp, MIN_SCORE=0.0, EMBEDDING_PROVIDER="local",
                                      STRUCTURED_QUERIES=self.structured)
        overrides.enable()
        self.addCleanup(overrides.disable)
        vector_index.reset_for_tests()
        structured_query.reset_for_tests()
        embedding_service.clear_query_cache()
        rag_service._answer_cache.clear()

    def ingest(self, **kwargs):
        return ingestion_service.run_ingestion(log=lambda m: None, **kwargs)

    def emb(self, source, doc_filter=None):
        return self.db["rag_embeddings"].find_one({"_id": {"$regex": f"^{source}:"}, **(doc_filter or {})})


@override_settings(SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies")
class PublicAccessTests(MongoCase):
    databases = {"default"}

    def setUp(self):
        super().setUp()
        self.client = APIClient()

    def test_chat_api_accepts_anonymous_requests(self):
        response = self.client.post("/api/v1/ask/", {"question": "Hello"}, format="json")
        self.assertEqual(response.status_code, 200, response.data)
        self.assertIn("answer", response.data)

    def test_anonymous_ask_does_not_retrieve_indexed_records(self):
        self.ingest()
        response = self.client.post(
            "/api/v1/ask/", {"question": "Design landing page"}, format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(response.data["sources"], [])

    @override_settings(CORS_ALLOWED_ORIGINS=["https://frontend.example"])
    def test_cors_allows_only_configured_browser_origins(self):
        allowed = self.client.options(
            "/api/v1/ask/",
            HTTP_ORIGIN="https://frontend.example",
            HTTP_ACCESS_CONTROL_REQUEST_METHOD="POST",
            HTTP_ACCESS_CONTROL_REQUEST_HEADERS="content-type",
        )
        self.assertEqual(allowed.headers.get("Access-Control-Allow-Origin"), "https://frontend.example")
        denied = self.client.get("/api/v1/health/", HTTP_ORIGIN="https://untrusted.example")
        self.assertNotIn("Access-Control-Allow-Origin", denied.headers)

    def test_anonymous_chat_history_can_be_listed_renamed_loaded_and_deleted(self):
        session_id = "cafe-1234"
        response = self.client.post(
            "/api/v1/ask/", {"question": "hello", "session_id": session_id}, format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)

        sessions = self.client.get("/api/v1/chats/")
        self.assertEqual(sessions.status_code, 200, sessions.data)
        self.assertEqual(sessions.data["sessions"][0]["session_id"], session_id)

        detail = self.client.get(f"/api/v1/chats/{session_id}/")
        self.assertEqual(detail.status_code, 200, detail.data)
        self.assertEqual(detail.data["turns"][0]["question"], "hello")

        renamed = self.client.post(
            f"/api/v1/chats/{session_id}/rename/", {"title": "Greeting"}, format="json",
        )
        self.assertEqual(renamed.status_code, 200, renamed.data)
        self.assertEqual(renamed.data["title"], "Greeting")

        deleted = self.client.delete(f"/api/v1/chats/{session_id}/delete/")
        self.assertEqual(deleted.status_code, 200, deleted.data)
        self.assertEqual(self.client.get("/api/v1/chats/").data["sessions"], [])

    def test_anonymous_chat_history_is_scoped_to_its_browser_session(self):
        session_id = "deadcafe-1234"
        response = self.client.post(
            "/api/v1/ask/", {"question": "hello", "session_id": session_id}, format="json",
        )
        self.assertEqual(response.status_code, 200, response.data)

        other_client = APIClient()
        self.assertEqual(other_client.get("/api/v1/chats/").data["sessions"], [])
        self.assertEqual(other_client.get(f"/api/v1/chats/{session_id}/").status_code, 404)

    @override_settings(INGEST_TOKEN="ingest-test-token", DEBUG=False)
    def test_ingestion_endpoint_uses_server_side_bearer_token(self):
        response = self.client.get("/api/v1/ingest/")
        self.assertEqual(response.status_code, 401)

        with patch("rag.api_views.ingestion_service.start_job", return_value={"id": "test", "state": "running"}):
            response = self.client.post(
                "/api/v1/ingest/", {"full": False}, format="json",
                HTTP_AUTHORIZATION="Bearer ingest-test-token",
            )
        self.assertEqual(response.status_code, 202, response.data)
        self.assertEqual(response.data["id"], "test")


def _global_scope_for_test(user_id):
    """Exercise legacy global-query behavior without granting it to real accounts."""
    return authorization_service.AccessScope(
        user_id, True, None, f"{user_id}:test-global", f"{user_id}:admin",
    )


class ConverterTests(SimpleTestCase):
    def test_types_are_converted_to_readable_text(self):
        from datetime import datetime
        oid = ObjectId()
        doc = {"_id": ObjectId(), "name": "Ann", "tags": ["a", "b"], "active": True,
               "joined": datetime(2026, 8, 12, 6, 38), "owner": oid,
               "address": {"city": "Paris", "geo": {"lat": 1.5}},
               "password": "secret", "empty": "", "blob": b"x", "items": [{"sku": "A1"}, {"sku": "B2"}]}
        text = "\n".join(document_converter.convert_document("people", doc).lines)
        for expected in ("Name: Ann", "Tags: a, b", "Active: Yes", "Joined: August 12, 2026 06:38 UTC",
                         f"Owner ID: {oid}", "Address City: Paris", "Address Geo Lat: 1.5",
                         "Items 1 Sku: A1", "Items 2 Sku: B2"):
            self.assertIn(expected, text)
        for missing in ("secret", "Password", "Empty", "Blob"):
            self.assertNotIn(missing, text)

    def test_metadata_has_primary_id_collection_fields_and_references(self):
        oid, doc_id = ObjectId(), ObjectId()
        record = document_converter.to_records("things", {"_id": doc_id, "label": "X", "ref": oid})[0]
        meta = record["metadata"]
        self.assertEqual(record["_id"], f"things:{doc_id}:0")
        self.assertEqual((meta["collection"], meta["primary_id"]), ("things", str(doc_id)))
        self.assertIn("label", meta["field_names"])
        self.assertEqual(meta["reference_ids"], [str(oid)])

    def test_long_text_is_chunked_and_nothing_is_lost(self):
        words = " ".join(f"w{i}" for i in range(1500)) + " THE_END"
        records = document_converter.to_records("docs", {"_id": "d1", "name": "Report", "body": words})
        self.assertGreater(len(records), 5)
        self.assertTrue(all(len(r["text"]) < 1000 for r in records))
        self.assertIn("THE_END", records[-1]["text"])
        self.assertIn("(Report)", records[1]["text"])
        self.assertEqual(len({r["_id"] for r in records}), len(records))

    def test_empty_document_gives_no_records(self):
        self.assertEqual(document_converter.to_records("c", {"_id": ObjectId()}), [])


class RelationshipTests(MongoCase):
    def test_guess_collections(self):
        names = ["users", "workspaces", "categories", "addresses"]
        g = relationships.guess_collections
        self.assertEqual(g("userId", names), ["users"])
        self.assertEqual(g("workspace", names), ["workspaces"])
        self.assertEqual(g("category_id", names), ["categories"])
        self.assertEqual(g("address", names), ["addresses"])
        self.assertEqual(g("createdBy", names), [])

    def test_detects_targets_by_name_and_by_probing(self):
        resolver = relationships.RelationshipResolver(self.db, list(SAMPLE))
        resolver.prefetch("memberships", list(self.db["memberships"].find()))
        resolver.prefetch("projects", list(self.db["projects"].find()))
        resolver.prefetch("tasks", list(self.db["tasks"].find()))
        t = resolver._targets
        self.assertEqual(t[("memberships", "user")], "users")
        self.assertEqual(t[("memberships", "workspace")], "workspaces")
        self.assertEqual(t[("projects", "workspaceId")], "workspaces")
        self.assertEqual(t[("projects", "createdBy")], "users")       # found by probing
        self.assertEqual(t[("tasks", "assignedTo")], "users")         # found by probing
        self.assertEqual(t[("tasks", "projectId")], "projects")

    def test_string_ids_are_treated_as_references_when_they_resolve(self):
        self.db["notes"].insert_one({"text": "hello", "author": str(ObjectId("b00000000000000000000001"))})
        resolver = relationships.RelationshipResolver(self.db, list(SAMPLE) + ["notes"])
        doc = self.db["notes"].find_one()
        resolver.prefetch("notes", [doc])
        text = "\n".join(document_converter.convert_document("notes", doc, resolver).lines)
        self.assertIn("Author: Aisha Khan", text)


class IngestionTests(MongoCase):
    def test_ingest_stores_text_metadata_relationships_and_vector_together(self):
        summary = self.ingest()
        self.assertEqual(summary["mode"], "incremental")
        doc = self.emb("memberships")
        self.assertEqual(len(doc["embedding"]), 128)
        self.assertIn("User: Aisha Khan", self.db["rag_embeddings"].find_one(
            {"_id": f"memberships:{self.db['memberships'].find_one()['_id']}:0"})["text"])
        self.assertIn("Workspace: Acme Corp", doc["text"])
        self.assertEqual({r["ref_collection"] for r in doc["relationships"]}, {"users", "workspaces"})
        self.assertEqual(doc["metadata"]["model"], "local:sentence-transformers/all-MiniLM-L6-v2")
        for rel in doc["relationships"]:
            self.assertEqual(set(rel), {"field", "ref_id", "ref_collection", "ref_name"})

    def test_sensitive_fields_never_reach_embeddings(self):
        self.ingest()
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "notarealhash|Password"}}), 0)

    def test_second_run_embeds_nothing_then_only_changes_and_deletions(self):
        first = self.ingest()
        total = sum(s["embedded"] for s in first["collections"].values())
        self.assertGreater(total, 10)
        second = self.ingest()
        self.assertEqual(sum(s["embedded"] for s in second["collections"].values()), 0)

        self.db["projects"].update_one({"project": "ShopEase"}, {"$set": {"status": "completed"}})
        self.db["tasks"].delete_one({"title": "Fix login bug"})
        third = self.ingest()
        self.assertEqual(third["collections"]["projects"]["embedded"], 1)
        self.assertEqual(third["collections"]["tasks"]["removed"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "Fix login bug"}}), 0)

    def test_renaming_a_referenced_document_refreshes_the_documents_that_point_to_it(self):
        self.ingest()
        self.db["workspaces"].update_one({"name": "Acme Corp"}, {"$set": {"name": "Acme Global"}})
        summary = self.ingest()
        self.assertGreaterEqual(summary["collections"]["memberships"]["embedded"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"text": {"$regex": "Acme Global"}}) > 0, True)

    def test_full_mode_reembeds_everything(self):
        self.ingest()
        again = self.ingest(full=True)
        self.assertEqual(sum(s["unchanged"] for s in again["collections"].values()), 0)

    def test_unknown_collection_is_a_clean_error(self):
        from rag.services.exceptions import IngestionError
        with self.assertRaises(IngestionError):
            self.ingest(collections=["nope"])

    def test_embeddings_collection_is_never_ingested_itself(self):
        self.ingest()
        self.assertNotIn("rag_embeddings", ingestion_service.mongo_service.list_collections())


class SearchTests(MongoCase):
    def setUp(self):
        super().setUp()
        self.ingest()

    def test_vector_search_finds_the_right_record(self):
        hits = retrieval.search("Health4Travel completed", top_k=3, hybrid=False)
        self.assertTrue(any(h.collection == "projects" and h.display_name == "Health4Travel" for h in hits))
        self.assertTrue(all(h.vector_score is not None for h in hits))

    def test_collection_filter(self):
        hits = retrieval.search("Acme Corp", top_k=5, collections=["workspaces"], hybrid=False)
        self.assertTrue(hits and all(h.collection == "workspaces" for h in hits))

    def test_hnsw_search_uses_roaring_selector_for_collection_and_account_filters(self):
        from django.conf import settings
        from pyroaring import BitMap

        index = vector_index.get_index()
        allowed_id = next(doc_id for doc_id in index.ids if doc_id.startswith("workspaces:"))
        bitmap, (selector, mask) = index._filter_selector(["workspaces"], {allowed_id}, "test-scope")
        row = index.ids.index(allowed_id)
        self.assertIsInstance(bitmap, BitMap)
        self.assertTrue(selector.is_member(row))
        self.assertTrue(mask[row >> 3] & (1 << (row & 7)))

        query_vector = embedding_service.embed_query("Acme Corp")
        hits = index.search(query_vector, 5, ["workspaces"], {allowed_id}, "test-scope")
        self.assertEqual([doc_id for doc_id, _score in hits], [allowed_id])

        loaded = vector_index.VectorIndex()
        loaded.load(settings.VECTOR_STORE_DIR)
        self.assertEqual(loaded.ids, index.ids)
        self.assertEqual(loaded.collections, index.collections)

    def test_legacy_index_metadata_triggers_hnsw_rebuild(self):
        from django.conf import settings
        from pathlib import Path

        metadata_path = Path(settings.VECTOR_STORE_DIR) / vector_index.META_FILE
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("backend")
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        vector_index.reset_for_tests()

        rebuilt = vector_index.get_index()
        self.assertTrue(hasattr(rebuilt.index, "hnsw"))
        self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["backend"],
                         vector_index.BACKEND)

    def test_keyword_failure_falls_back_to_vector_search(self):
        hits = retrieval.search("Health4Travel completed", top_k=3, hybrid=True)  # mongomock has no $text
        self.assertTrue(hits)

    def test_rrf_fusion_rewards_agreement(self):
        fused = rrf_fuse([("a", .9), ("b", .8)], [("b", 5.0), ("c", 4.0)], k=3, keyword_weight=1.0, rrf_k=60)
        self.assertEqual(fused[0][0], "b")
        self.assertEqual({i for i, _ in fused}, {"a", "b", "c"})

    def test_multi_hop_follows_relationships_forward_and_reverse(self):
        hits = [h for h in retrieval.search("Rohan Mehta backend developer", top_k=8, hybrid=False)
                if h.collection == "users"][:1]
        self.assertEqual(hits[0].display_name, "Rohan Mehta")
        related = retrieval.expand(hits, hops=1, max_related=10)
        self.assertTrue(any(r.collection == "tasks" and r.via["direction"] == "reverse" for r in related))

        task = [h for h in retrieval.search("Fix login bug", top_k=8, hybrid=False) if h.collection == "tasks"][0]
        forward = {(r.collection, r.display_name) for r in retrieval.expand([task], hops=1, max_related=10)
                   if r.via["direction"] == "forward"}
        self.assertIn(("projects", "ShopEase"), forward)
        self.assertIn(("users", "Rohan Mehta"), forward)

    def test_two_hops_reach_the_workspace_through_the_project(self):
        task = [h for h in retrieval.search("Fix login bug", top_k=8, hybrid=False) if h.collection == "tasks"][0]
        related = retrieval.expand([task], hops=2, max_related=10)
        self.assertIn("workspaces", {r.collection for r in related})

    def test_employee_search_only_returns_records_linked_to_their_user_id(self):
        aisha_id = str(self.db["users"].find_one({"email": "aisha@example.com"})["_id"])
        result = rag_service.search("Design landing page", user_id=aisha_id, hybrid=False, expand=True)
        returned_ids = {row["id"] for row in result["results"]}
        returned_ids.update(row["id"] for row in result.get("related", []))
        scope = authorization_service.resolve_scope(aisha_id)
        self.assertTrue(returned_ids)
        self.assertTrue(returned_ids.issubset(scope.embedding_ids))
        self.assertTrue(any("Design landing page" in row["text"] for row in result["results"]))
        self.assertFalse(any("Fix login bug" in row["text"] for row in result["results"]))

    def test_employee_lookup_of_another_user_is_rejected(self):
        aisha_id = str(self.db["users"].find_one({"email": "aisha@example.com"})["_id"])
        with self.assertRaisesRegex(
            AuthorizationError,
            r"You are not authenticated , to access information",
        ):
            rag_service.ask("Who is Rohan Mehta?", user_id=aisha_id)

    def test_admin_search_can_access_all_records(self):
        self.ingest()
        result = rag_service.search("Fix login bug", user_id=self.admin_id, hybrid=False)
        self.assertTrue(any("Fix login bug" in row["text"] for row in result["results"]))
        scope = self.real_resolve_scope(self.admin_id)
        self.assertIsNone(scope.embedding_ids)
        self.assertTrue(scope.is_admin)

    def test_account_scope_uses_database_role_and_validates_email(self):
        aisha = self.db["users"].find_one({"email": "aisha@example.com"})
        aisha_id = str(aisha["_id"])

        scope = self.real_resolve_scope(aisha_id, email="AISHA@example.com")
        self.assertFalse(scope.is_admin)
        self.assertIsNotNone(scope.embedding_ids)
        with self.assertRaises(AuthorizationError):
            self.real_resolve_scope(aisha_id, email="meera@example.com")

    @override_settings(STRUCTURED_QUERIES=True)
    def test_employee_questions_skip_unscoped_structured_queries(self):
        aisha_id = str(self.db["users"].find_one({"email": "aisha@example.com"})["_id"])
        scope = self.real_resolve_scope(aisha_id)
        with patch.object(structured_query, "try_structured") as exact:
            result = rag_service.ask("List all tasks", user_id=aisha_id)

        exact.assert_not_called()
        self.assertTrue(all(
            any(doc_id.startswith(f"{source['collection']}:{source['document_id']}:")
                for doc_id in scope.embedding_ids)
            for source in result["sources"]
        ))

    @override_settings(STRUCTURED_QUERIES=True)
    def test_admin_can_look_up_any_users_profile(self):
        pooja_id = self.db["users"].insert_one({
            "name": "Pooja Bharti",
            "email": "pooja@example.com",
            "department": "Finance",
            "leaveBalance": 12,
        }).inserted_id
        self.db["leave_entries"].insert_one({
            "user": pooja_id,
            "hours": 4,
            "date": "2026-08-27",
            "type": "leave",
        })
        self.ingest()

        result = rag_service.ask("who is Pooja", user_id=self.admin_id)
        scope = self.real_resolve_scope(self.admin_id)
        self.assertIsNone(scope.embedding_ids)
        self.assertIn("Pooja", result["answer"])


class BlockedActivityCollectionTests(MongoCase):
    """The audit 'activities' collection must never be indexed, searched, planned against, or expanded into."""

    def setUp(self):
        super().setUp()
        self.db["activities"].insert_many([
            {"type": "time_logged", "description": "SECRET_ACTIVITY_MARKER logged four hours",
             "card": ObjectId(), "createdAt": "2026-08-12T06:38:00Z"},
            {"type": "card_moved", "description": "SECRET_ACTIVITY_MARKER moved a card", "board": ObjectId()},
        ])

    def _stale_activity_embedding(self):
        vector = fake_embed_texts(["SECRET_ACTIVITY_MARKER activity audit"])[0].tolist()
        return {
            "_id": "activities:old-1:0",
            "embedding": vector,
            "text": "Collection: activities\nDescription: SECRET_ACTIVITY_MARKER logged four hours",
            "metadata": {"collection": "activities", "primary_id": "old-1", "chunk_index": 0,
                         "chunk_count": 1, "field_names": ["description"], "reference_ids": [],
                         "display_name": "SECRET_ACTIVITY_MARKER", "content_hash": "stale",
                         "model": "local:sentence-transformers/all-MiniLM-L6-v2"},
            "relationships": [],
        }

    def test_activities_is_not_listed_ingested_or_kept_from_an_older_index(self):
        self.assertNotIn("activities", mongo_service.list_collections())
        self.assertTrue(mongo_service.is_blocked_collection("activities"))
        self.db["rag_embeddings"].insert_one(self._stale_activity_embedding())

        summary = self.ingest()

        self.assertNotIn("activities", summary["collections"])
        self.assertEqual(summary["blocked_removed"], 1)
        self.assertEqual(self.db["rag_embeddings"].count_documents({"metadata.collection": "activities"}), 0)

    def test_vector_search_never_returns_activities_even_if_the_index_still_contains_one(self):
        self.ingest()
        self.db["rag_embeddings"].insert_one(self._stale_activity_embedding())
        vector_index.rebuild_index()

        hits = retrieval.search("SECRET_ACTIVITY_MARKER", top_k=10, hybrid=False)
        self.assertFalse(any(hit.collection == "activities" for hit in hits))
        self.assertEqual(retrieval.search("SECRET_ACTIVITY_MARKER", top_k=10, hybrid=False,
                                          collections=["activities"]), [])

    def test_expansion_never_follows_relationships_into_activities(self):
        from rag.services.retrieval import Hit

        self.ingest()
        self.db["rag_embeddings"].insert_one(self._stale_activity_embedding())
        vector_index.rebuild_index()
        origin = Hit(id="cards:origin:0", text="Project card", collection="cards", primary_id="origin",
                     relationships=[{"field": "activity", "ref_id": "old-1", "ref_collection": "activities"}])
        related = retrieval.expand([origin], hops=1, max_related=5)
        self.assertFalse(any(hit.collection == "activities" for hit in related))

    def test_catalog_and_plan_validation_reject_activities(self):
        self.ingest()
        catalog = structured_query.build_catalog(force=True)
        self.assertNotIn("activities", catalog)

        contaminated = {"activities": {"count": 2, "fields": {}, "valid": []}}
        plan = {"collection": "activities", "operation": "list", "filters": []}
        self.assertIsNone(structured_query.validate_plan(plan, contaminated))
        self.assertIsNone(structured_query.execute(plan, contaminated, "planner"))

    def test_a_saved_catalog_from_an_older_build_is_pruned_on_load(self):
        from pathlib import Path
        from django.conf import settings

        self.ingest()
        path = Path(settings.VECTOR_STORE_DIR) / structured_query.CATALOG_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        data["catalog"]["activities"] = {"count": 2, "fields": {}, "valid": []}
        path.write_text(json.dumps(data), encoding="utf-8")

        structured_query.reset_for_tests()
        catalog = structured_query.build_catalog()
        self.assertNotIn("activities", catalog)


class AskTests(MongoCase):
    def setUp(self):
        super().setUp()
        scope_patch = patch.object(
            authorization_service, "resolve_scope",
            side_effect=lambda user_id, email=None: _global_scope_for_test(user_id or self.admin_id),
        )
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        self.ingest()

    def test_retrieval_context_uses_prior_questions_not_generated_answers(self):
        history = [{"question": "List Aisha Khan's tasks", "answer": "unrelated generated claim"}]
        query = rag_service._retrieval_query("which ones are completed?", history)
        self.assertIn("List Aisha Khan's tasks", query)
        self.assertIn("which ones are completed?", query)
        self.assertNotIn("unrelated generated claim", query)

    def test_ordinal_followup_uses_first_item_from_previous_answer(self):
        session_id = "cafe-1234"
        answer = ("There are 26 categories. Here are some examples:\n\n"
                  "- SEO (under the SEO department)\n- Hubspot (under the Frontend department)")
        chat_owner = f"{self.admin_id}:admin"
        rag_service.chat_memory.clear(session_id, owner_id=chat_owner)
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=chat_owner)
        rag_service.chat_memory.add(session_id, "How many categories are there?", answer, owner_id=chat_owner)
        with patch.object(rag_service.retrieval, "search") as search:
            result = rag_service.ask("what is first one", session_id=session_id, user_id=self.admin_id)
        self.assertEqual(result["answer"], "The first one listed was SEO (under the SEO department).")
        self.assertEqual(result["mode"], "history_reference")
        search.assert_not_called()

    def test_conversational_turn_is_stored_and_can_be_edited(self):
        session_id = "ca11c011ab"
        owner_id = f"{self.admin_id}:admin"
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=owner_id)

        rag_service.ask("hello", session_id=session_id, user_id=self.admin_id)
        rag_service.ask("goodbye", session_id=session_id, user_id=self.admin_id)
        rag_service.ask("thanks", session_id=session_id, edit_index=1, user_id=self.admin_id)

        state = rag_service.chat_memory.get_state(session_id, owner_id=owner_id)
        self.assertEqual([turn["question"] for turn in state["turns"]], ["hello", "thanks"])

    def test_edit_index_tracks_turns_after_history_window_rolls(self):
        session_id = "ca11c012ab"
        owner_id = f"{self.admin_id}:admin"
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=owner_id)
        for index in range(rag_service.chat_memory.MAX_TURNS + 2):
            rag_service.chat_memory.add(session_id, f"question {index}", f"answer {index}", owner_id=owner_id)

        result = rag_service.ask(
            "hello", session_id=session_id, edit_index=11, user_id=self.admin_id)

        state = rag_service.chat_memory.get_state(session_id, owner_id=owner_id)
        self.assertEqual(result["mode"], "conversational")
        self.assertEqual(state["offset"], 2)
        self.assertEqual(len(state["turns"]), rag_service.chat_memory.MAX_TURNS)
        self.assertEqual(state["turns"][-1]["question"], "hello")

    def test_editing_turn_outside_retained_history_has_clear_error(self):
        session_id = "ca11c013ab"
        owner_id = f"{self.admin_id}:admin"
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=owner_id)
        for index in range(rag_service.chat_memory.MAX_TURNS + 2):
            rag_service.chat_memory.add(session_id, f"question {index}", f"answer {index}", owner_id=owner_id)

        with self.assertRaisesRegex(InvalidRequestError, "outside the 10-turn edit history"):
            rag_service.ask("hello", session_id=session_id, edit_index=1, user_id=self.admin_id)

    def test_admin_cannot_reopen_legacy_global_chat_history(self):
        session_id = "ad11001234ab"
        admin_owner = f"{self.admin_id}:admin"
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=admin_owner)
        rag_service.chat_memory.add(
            session_id, "List all users", "A global list of users.", owner_id=admin_owner)

        with patch.object(authorization_service, "resolve_scope", side_effect=self.real_resolve_scope), \
                self.assertRaises(InvalidRequestError):
            rag_service.ask("what is first one", session_id=session_id, user_id=self.admin_id)

    def test_ask_sends_context_with_related_records_and_returns_sources(self):
        with patch.object(llm_service, "generate", return_value="ShopEase is in progress.") as gen:
            result = rag_service.ask("What is the status of ShopEase?", hops=1)
        self.assertEqual(result["answer"], "ShopEase is in progress.")
        context = gen.call_args[0][1]
        self.assertIn("ShopEase", context)
        self.assertIn("[1]", context)
        self.assertTrue(any(s["collection"] == "projects" for s in result["sources"]))
        self.assertIn("search", result["timings_ms"])

    def test_not_found_answer_has_no_sources(self):
        with patch.object(llm_service, "generate", return_value=llm_service.NOT_FOUND_ANSWER):
            self.assertEqual(rag_service.ask("What is the CEO's phone number?")["sources"], [])

    def test_answers_are_cached_until_the_index_changes(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("Which projects are completed?")
            second = rag_service.ask("which   projects are COMPLETED?")
            self.assertTrue(second["cached"])
            self.assertEqual(gen.call_count, 1)
            self.db["projects"].update_one({"project": "ShopEase"}, {"$set": {"status": "paused"}})
            self.ingest()
            rag_service.ask("Which projects are completed?")
            self.assertEqual(gen.call_count, 2)

    def test_repeated_question_in_same_chat_uses_cached_answer(self):
        session_id = "ca11e1234abc"
        self.addCleanup(rag_service.chat_memory.clear, session_id)
        with patch.object(llm_service, "generate", return_value="ShopEase is in progress.") as gen, \
                patch.object(rag_service.retrieval, "search", wraps=rag_service.retrieval.search) as search:
            rag_service.ask("What is the status of ShopEase?", session_id=session_id)
            second = rag_service.ask("  what is the status of ShopEase?  ", session_id=session_id)
            third = rag_service.ask("What is the status of ShopEase?", session_id=session_id)

        self.assertTrue(second["cached"])
        self.assertTrue(third["cached"])
        self.assertEqual(third["answer"], "ShopEase is in progress.")
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(search.call_count, 1)

    def test_changed_chat_context_does_not_reuse_an_old_answer(self):
        session_id = "ca11c0123abc"
        self.addCleanup(rag_service.chat_memory.clear, session_id)
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("What is the status of ShopEase?", session_id=session_id)
            rag_service.ask("Tell me about the projects.", session_id=session_id)
            repeated = rag_service.ask("What is the status of ShopEase?", session_id=session_id)

        self.assertFalse(repeated["cached"])
        self.assertEqual(gen.call_count, 3)

    @override_settings(STRUCTURED_QUERIES=True)
    def test_structured_answers_are_cached_for_repeated_questions(self):
        exact = structured_query.ExactResult(
            collection="projects", total=1,
            records=[{"document_id": "project-1", "name": "ShopEase", "text": "ShopEase"}],
            conditions="none", complete=True, source="planner", operation="list",
            context="EXACT QUERY RESULT", timings={},
        )
        with patch.object(structured_query, "try_structured", return_value=exact) as query, \
                patch.object(llm_service, "generate", return_value="ShopEase is listed.") as gen, \
                patch.object(authorization_service, "resolve_scope",
                             side_effect=lambda user_id, email=None: _global_scope_for_test(user_id or self.admin_id)):
            rag_service.ask("List all projects")
            repeated = rag_service.ask(" list all projects ")

        self.assertTrue(repeated["cached"])
        self.assertEqual(query.call_count, 1)
        self.assertEqual(gen.call_count, 1)

    @override_settings(MAX_CONTEXT_CHARS=300)
    def test_context_respects_the_character_budget(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            rag_service.ask("Tell me about tasks and projects", top_k=10)
        self.assertLess(len(gen.call_args[0][1]), 900)

    def test_list_question_detection(self):
        for yes in ("List all tasks", "which projects have been finished?", "How many tasks are there?",
                    "Show me medium priority tasks", "What are the open tasks", "every user in Acme"):
            self.assertTrue(rag_service.is_list_question(yes), yes)
        for no in ("What is the status of ShopEase?", "Who is the manager of Acme?", "which is the best project"):
            self.assertFalse(rag_service.is_list_question(no), no)

    def test_list_questions_retrieve_more_than_the_selected_mode(self):
        with patch.object(llm_service, "generate", return_value="ok") as gen:
            wide = rag_service.ask("List all records", top_k=2, hops=0)
            narrow = rag_service.ask("What is the status of ShopEase?", top_k=2, hops=0)
        self.assertGreater(len(wide["sources"]), 5)
        self.assertTrue(wide["list_mode"])
        self.assertEqual(wide["records_used"], len(wide["sources"]))
        self.assertLessEqual(len(narrow["sources"]), 2)
        self.assertFalse(narrow["list_mode"])
        self.assertEqual(gen.call_args_list[0].kwargs["max_tokens"], 2048)
        self.assertIsNone(gen.call_args_list[1].kwargs["max_tokens"])

    @override_settings(LIST_TOP_K=0)
    def test_list_widening_can_be_turned_off(self):
        with patch.object(llm_service, "generate", return_value="ok"):
            self.assertLessEqual(len(rag_service.ask("List all records", top_k=2, hops=0)["sources"]), 2)

    def test_validation(self):
        for bad in ("  ", "x" * 501, 5):
            with self.assertRaises(InvalidRequestError):
                rag_service.validate_question(bad)

    def test_missing_index_is_a_clean_error(self):
        self.db["rag_embeddings"].delete_many({})
        vector_index.reset_for_tests()
        import shutil
        from django.conf import settings
        shutil.rmtree(settings.VECTOR_STORE_DIR, ignore_errors=True)
        with self.assertRaises(VectorStoreError):
            rag_service.ask("anything at all")

    @override_settings(LLM_API_KEY="")
    def test_missing_llm_key_is_a_clean_error(self):
        from rag.services.exceptions import ConfigError
        llm_service._client = None
        with self.assertRaises(ConfigError):
            rag_service.ask("Which projects are completed?")


@override_settings(SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies")
class ApiTests(MongoCase):
    databases = {"default"}

    def setUp(self):
        super().setUp()
        self.client = Client()
        self.client.get("/")
        self.csrf_token = self.client.cookies["csrftoken"].value

    def test_search_api_is_accessible_without_login(self):
        self.ingest()

        response = self.post("/api/search/", {"query": "Design landing page", "hybrid": False, "expand": True})
        self.assertEqual(response.status_code, 200, response.content)

    def test_search_api_works_without_anonymous_session_cookie(self):
        from django.conf import settings
        self.client.cookies.pop(settings.SESSION_COOKIE_NAME, None)
        self.ingest()
        response = self.post("/api/search/", {"query": "private user data"})
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()["results"], [])

    def post(self, url, payload, **extra):
        extra.setdefault("HTTP_X_CSRFTOKEN", self.csrf_token)
        return self.client.post(url, json.dumps(payload), content_type="application/json", **extra)

    def test_index_page(self):
        self.assertContains(self.client.get("/"), "Ask anything about your MongoDB data.")

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_ask_endpoint_end_to_end(self):
        self.ingest()
        admin = self.db["users"].find_one({"_id": ObjectId(self.admin_id)})
        secret = "api-test-secret-that-is-at-least-32-bytes"
        token = jwt.encode({"id": self.admin_id}, secret, algorithm="HS256")
        with patch.object(llm_service, "generate", return_value="Health4Travel is completed."):
            response = self.post("/api/ask/", {
                "question": "Which projects have been finished?",
                "user_id": self.admin_id,
                "email": admin["email"],
                "role": "admin",
            }, HTTP_AUTHORIZATION=f"Bearer {token}")
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["answer"], "Health4Travel is completed.")
        self.assertTrue(data["sources"])

    def test_ask_api_persists_chat_and_reuses_history(self):
        answers = [
            {"answer": "There are four projects.", "sources": [{"name": "Project data"}]},
            {"answer": "The first is Health4Travel.", "sources": []},
        ]
        with patch("rag.api_views.rag_service.ask", side_effect=answers) as ask:
            first = self.post("/api/v1/ask/", {"question": "How many projects are there?"})
            self.assertEqual(first.status_code, 200, first.content)
            session_id = first.json()["session_id"]

            second = self.post("/api/v1/ask/", {
                "question": "Which is the first?",
                "session_id": session_id,
            })
            self.assertEqual(second.status_code, 200, second.content)

        self.assertEqual(ask.call_args_list[0].kwargs["history"], [])
        self.assertEqual(ask.call_args_list[1].kwargs["history"], [{
            "question": "How many projects are there?",
            "answer": "There are four projects.",
        }])
        detail = self.client.get(f"/api/v1/chats/{session_id}/")
        self.assertEqual(detail.status_code, 200, detail.content)
        self.assertEqual(len(detail.json()["messages"]), 4)

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_ask_api_accepts_frontend_identity_fields(self):
        user_id = self.admin_id
        email = self.db["users"].find_one({"_id": ObjectId(user_id)})["email"]
        secret = "api-test-secret-that-is-at-least-32-bytes"
        token = jwt.encode({"id": user_id}, secret, algorithm="HS256")
        with patch(
            "rag.api_views.rag_service.ask",
            return_value={"answer": "Welcome.", "sources": []},
        ) as ask:
            response = self.post("/api/v1/ask/", {
                "question": "Hello there",
                "session_id": "beef-1234",
                "user_id": user_id,
                "email": email,
                "role": "admin",
            }, HTTP_AUTHORIZATION=f"Bearer {token}")

        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()["session_id"], "beef-1234")
        self.assertEqual(ask.call_args.kwargs["user_id"], user_id)

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_account_chat_survives_missing_django_session_cookie(self):
        from django.conf import settings

        user_id = self.admin_id
        email = self.db["users"].find_one({"_id": ObjectId(user_id)})["email"]
        secret = "api-test-secret-that-is-at-least-32-bytes"
        token = jwt.encode({"id": user_id, "email": email}, secret, algorithm="HS256")
        headers = {"HTTP_AUTHORIZATION": f"Bearer {token}"}

        with patch(
            "rag.api_views.rag_service.ask",
            return_value={"answer": "First answer.", "sources": []},
        ):
            first = self.post("/api/v1/ask/", {
                "question": "First question",
                "user_id": user_id,
                "email": email,
                "role": "admin",
            }, **headers)
        self.assertEqual(first.status_code, 200, first.content)
        session_id = first.json()["session_id"]

        self.client.cookies.pop(settings.SESSION_COOKIE_NAME, None)
        with patch(
            "rag.api_views.rag_service.ask",
            return_value={"answer": "Second answer.", "sources": []},
        ):
            second = self.post("/api/v1/ask/", {
                "question": "Second question",
                "session_id": session_id,
                "user_id": user_id,
                "email": email,
                "role": "admin",
            }, **headers)
        self.assertEqual(second.status_code, 200, second.content)

        self.client.cookies.pop(settings.SESSION_COOKIE_NAME, None)
        sessions = self.client.get("/api/v1/chats/", **headers)
        self.assertEqual(sessions.status_code, 200, sessions.content)
        self.assertEqual(sessions.json()["sessions"][0]["session_id"], session_id)

        self.client.cookies.pop(settings.SESSION_COOKIE_NAME, None)
        detail = self.client.get(f"/api/v1/chats/{session_id}/", **headers)
        self.assertEqual(detail.status_code, 200, detail.content)
        self.assertEqual(len(detail.json()["turns"]), 2)

        anonymous_detail = self.client.get(f"/api/v1/chats/{session_id}/")
        self.assertEqual(anonymous_detail.status_code, 404)

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_ask_api_rejects_user_id_that_does_not_match_token(self):
        secret = "api-test-secret-that-is-at-least-32-bytes"
        token = jwt.encode({"id": self.admin_id}, secret, algorithm="HS256")
        with patch("rag.api_views.rag_service.ask") as ask:
            response = self.post("/api/v1/ask/", {
                "question": "Hello there",
                "user_id": "507f1f77bcf86cd799439011",
                "email": "person@example.com",
                "role": "admin",
            }, HTTP_AUTHORIZATION=f"Bearer {token}")

        self.assertEqual(response.status_code, 401, response.content)
        ask.assert_not_called()

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_employee_cannot_expand_search_by_claiming_admin_role(self):
        self.ingest()
        employee = self.db["users"].find_one({"email": "aisha@example.com"})
        employee_id = str(employee["_id"])
        scope = self.real_resolve_scope(employee_id)
        secret = "api-test-secret-that-is-at-least-32-bytes"
        token = jwt.encode({"id": employee_id}, secret, algorithm="HS256")

        with patch.object(llm_service, "generate", return_value="Here are the matching records."):
            response = self.post("/api/v1/ask/", {
                "question": "Design landing page",
                "user_id": employee_id,
                "email": employee["email"],
                "role": "admin",
            }, HTTP_AUTHORIZATION=f"Bearer {token}")

        self.assertEqual(response.status_code, 200, response.content)
        self.assertTrue(response.data["sources"])
        self.assertTrue(all(
            any(doc_id.startswith(f"{source['collection']}:{source['document_id']}:")
                for doc_id in scope.embedding_ids)
            for source in response.data["sources"]
        ))

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_api_returns_authentication_message_for_another_employee_lookup(self):
        employee = self.db["users"].find_one({"email": "aisha@example.com"})
        employee_id = str(employee["_id"])
        secret = "api-test-secret-that-is-at-least-32-bytes"
        token = jwt.encode({"id": employee_id, "email": employee["email"]}, secret, algorithm="HS256")

        response = self.post("/api/v1/ask/", {
            "question": "Who is Rohan Mehta?",
            "user_id": employee_id,
            "email": employee["email"],
            "role": "employee",
        }, HTTP_AUTHORIZATION=f"Bearer {token}")

        self.assertEqual(response.status_code, 403, response.content)
        self.assertEqual(
            response.json()["error"],
            "You are not authenticated , to access information",
        )

    def test_ask_endpoint_accepts_absolute_edit_index_above_nine(self):
        with patch("rag.api_views.rag_service.ask", return_value={"answer": "stored answer", "sources": []}):
            first = self.post("/api/v1/ask/", {"question": "question 0"})
            session_id = first.json()["session_id"]
            for index in range(1, rag_service.chat_memory.MAX_TURNS + 2):
                self.post("/api/v1/ask/", {
                    "question": f"question {index}",
                    "session_id": session_id,
                })

        existing_history = self.client.get(f"/api/v1/chats/{session_id}/").json()
        self.assertEqual(existing_history["offset"], 2)
        self.assertEqual(len(existing_history["turns"]), rag_service.chat_memory.MAX_TURNS)

        response = self.post("/api/ask/", {
            "question": "hello", "session_id": session_id, "edit_index": 11,
        })

        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()["mode"], "conversational")
        history = self.client.get(f"/api/chats/{session_id}/").json()
        self.assertEqual(history["offset"], 2)
        self.assertEqual(history["turns"][-1]["question"], "hello")

    @override_settings(CHAT_JWT_SECRET="api-test-secret-that-is-at-least-32-bytes")
    def test_search_endpoint(self):
        self.ingest()
        token = jwt.encode(
            {"id": self.admin_id},
            "api-test-secret-that-is-at-least-32-bytes",
            algorithm="HS256",
        )
        data = self.post("/api/search/", {
            "query": "Acme Corp workspace",
            "top_k": 3,
            "hybrid": False,
            "expand": True,
            "user_id": self.admin_id,
            "email": self.db["users"].find_one({"_id": ObjectId(self.admin_id)})["email"],
            "role": "admin",
        }, HTTP_AUTHORIZATION=f"Bearer {token}").json()
        self.assertLessEqual(len(data["results"]), 3)
        self.assertIn("related", data)
        self.assertIn("relationships", data["results"][0])

    def test_bad_input_returns_json_errors(self):
        self.assertEqual(
            self.client.post(
                "/api/ask/", "nope", content_type="application/json",
                HTTP_X_CSRFTOKEN=self.csrf_token,
            ).status_code,
            400,
        )
        self.assertEqual(self.post("/api/ask/", {"question": "hello", "top_k": 999}).status_code, 400)
        self.assertEqual(self.post("/api/search/", {"query": "x", "collections": "users"}).status_code, 400)
        self.assertEqual(self.client.get("/api/ask/").status_code, 405)

    def test_ask_before_ingest_explains_what_to_do(self):
        response = self.post("/api/ask/", {"question": "Which projects are completed?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("ingest", response.json()["error"])

    @override_settings(INGEST_TOKEN="s3cret")
    def test_ingest_requires_the_token(self):
        self.assertEqual(self.post("/api/ingest/", {}).status_code, 401)
        self.assertEqual(self.client.get("/api/ingest/", HTTP_AUTHORIZATION="Bearer wrong").status_code, 401)
        with patch("rag.api_views.ingestion_service.start_job", return_value={"id": "x", "state": "running"}):
            response = self.post("/api/ingest/", {"full": True}, HTTP_AUTHORIZATION="Bearer s3cret")
        self.assertEqual(response.status_code, 202)

    @override_settings(INGEST_TOKEN="", DEBUG=False)
    def test_ingest_is_disabled_without_a_token_in_production(self):
        self.assertEqual(self.post("/api/ingest/", {}).status_code, 401)

    def test_health(self):
        self.ingest()
        data = self.client.get("/api/health/").json()
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["embeddings"], 0)
        self.assertTrue(data["index"]["ready"])


DEPARTMENTS = ["Engineering", "Human Resources", "Finance", "Marketing", "Sales", "Legal", "Operations", "Support",
               "Design", "Data Science", "Security", "Procurement", "Facilities", "Training", "Research",
               "Quality Assurance", "Logistics", "Public Relations", "Compliance", "Administration"]


@override_settings(STRUCT_PLANNER="on")
class StructuredQueryTests(MongoCase):
    """Exact answers for list/count questions (the '20 departments but I only see 6, 10, 8, 13' problem)."""

    structured = True

    def setUp(self):
        super().setUp()
        scope_patch = patch.object(
            authorization_service, "resolve_scope",
            side_effect=lambda user_id, email=None: _global_scope_for_test(user_id or self.admin_id),
        )
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])

    def ask(self, question, plan=None, **kw):
        with patch.object(llm_service, "plan_query", return_value=plan), \
             patch.object(llm_service, "generate", return_value="ok") as gen:
            result = rag_service.ask(question, **kw)
        return result, gen

    def test_all_20_departments_are_listed_every_time_even_without_the_planner(self):
        for question in ("List all departments", "Show me the department names", "What are our departments?",
                         "how many departments do we have"):
            result, gen = self.ask(question)                       # planner returns nothing -> name-match fallback
            self.assertEqual(result["mode"], "structured", question)
            self.assertEqual(result["total_matching"], 20, question)
            context = gen.call_args[0][1]
            self.assertIn("Exact number of records in the collection: 20", context)
            self.assertIn("the list is complete", context)
            for name in DEPARTMENTS:
                self.assertIn(name, context)
            self.assertNotIn("Document ID", context)
            self.assertEqual(len(result["sources"]), 20)

    def test_planner_filters_give_exact_counts_and_lists(self):
        plan = {"collection": "tasks", "operation": "list",
                "filters": [{"field": "priority", "op": "eq", "value": "MEDIUM"}]}      # case-insensitive
        result, gen = self.ask("Show me all medium priority tasks", plan)
        self.assertEqual(result["total_matching"], 2)
        self.assertEqual(result["conditions"], "priority = MEDIUM")
        context = gen.call_args[0][1]
        self.assertIn("Design landing page", context)
        self.assertIn("Fix login bug", context)
        self.assertNotIn("Set up payment gateway", context)
        self.assertIn("Exact number of matching records in the database: 2", context)

    def test_structured_followup_passes_recent_turns_to_planner(self):
        plan = {"collection": "tasks", "operation": "list",
            "filters": [{"field": "assignedTo", "op": "eq", "value": "Aisha Khan"},
                    {"field": "status", "op": "eq", "value": "done"}],
            "show": ["title", "status"]}
        session_id = "f00d-1234"
        history = [{"question": "List Aisha Khan's tasks", "answer": "She has a design task and a release-notes task."}]
        chat_owner = f"{self.admin_id}:admin"
        rag_service.chat_memory.clear(session_id, owner_id=chat_owner)
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=chat_owner)
        rag_service.chat_memory.add(
            session_id, history[0]["question"], history[0]["answer"], owner_id=chat_owner)
        with patch.object(llm_service, "plan_query", return_value=plan) as planner, \
             patch.object(llm_service, "generate", return_value="ok"):
            result = rag_service.ask("which ones are completed?", session_id=session_id, user_id=self.admin_id)
        self.assertEqual(result["collection"], "tasks")
        self.assertEqual(planner.call_args.kwargs["history"], history)

    def test_planner_failure_does_not_guess_collection_for_followup(self):
        history = [{"question": "List all people in Acme Corp", "answer": "Aisha Khan is a member."}]
        with patch.object(llm_service, "plan_query", return_value=None):
            result = structured_query.try_structured("department wise?", history=history)
        self.assertIsNone(result)

    def test_filter_by_a_persons_name_on_a_reference_field(self):
        plan = {"collection": "tasks", "operation": "list",
                "filters": [{"field": "assignedTo", "op": "eq", "value": "Rohan Mehta"}]}
        result, gen = self.ask("What tasks are assigned to Rohan Mehta?", plan)
        self.assertEqual(result["total_matching"], 2)
        context = gen.call_args[0][1]
        self.assertIn("Fix login bug", context)
        self.assertIn("Assigned To: Rohan Mehta", context)            # names are shown, not ids
        self.assertNotIn("Write release notes", context)

    def test_count_operation_sends_only_a_few_examples(self):
        plan = {"collection": "departments", "operation": "count", "filters": []}
        result, gen = self.ask("How many departments are there?", plan)
        self.assertEqual(result["total_matching"], 20)
        context = gen.call_args[0][1]
        self.assertIn("Exact number of matching records in the database: 20", context)
        self.assertIn("Only 10 examples are listed", context)
        self.assertEqual(result["records_used"], 10)

    @override_settings(STRUCT_MAX_RECORDS=3)
    def test_big_results_are_cut_but_the_total_stays_exact(self):
        plan = {"collection": "departments", "operation": "list", "filters": []}
        result, gen = self.ask("List all departments", plan)
        context = gen.call_args[0][1]
        self.assertEqual(result["total_matching"], 20)
        self.assertIn("Only the first 3 of 20", context)
        self.assertIn("NOT complete", context)

    def test_a_condition_is_never_silently_dropped(self):
        bad_field = {"collection": "tasks", "operation": "list",
                     "filters": [{"field": "colour", "op": "eq", "value": "red"}]}
        self.assertIsNone(structured_query.validate_plan(bad_field, structured_query.build_catalog()))
        bad_op = {"collection": "tasks", "filters": [{"field": "priority", "op": "$where", "value": "1"}]}
        self.assertIsNone(structured_query.validate_plan(bad_op, structured_query.build_catalog()))
        bad_num = {"collection": "tasks", "filters": [{"field": "priority", "op": "gt", "value": "abc"}]}
        self.assertIsNone(structured_query.validate_plan(bad_num, structured_query.build_catalog()))
        nested = {"collection": "tasks", "filters": [{"field": "priority", "op": "eq", "value": {"$ne": 1}}]}
        self.assertIsNone(structured_query.validate_plan(nested, structured_query.build_catalog()))
        unknown = {"collection": "nope", "filters": []}
        self.assertIsNone(structured_query.validate_plan(unknown, structured_query.build_catalog()))

    def test_invalid_plan_on_a_small_collection_reads_everything_and_says_conditions_were_not_applied(self):
        bad = {"collection": "tasks", "operation": "list", "filters": [{"field": "colour", "op": "eq", "value": "red"}]}
        result, gen = self.ask("Show me the tasks with a red label", bad)
        context = gen.call_args[0][1]
        self.assertEqual(result["mode"], "structured")
        self.assertIn("Conditions were NOT applied by the system", context)

    def test_zero_matches_fall_back_to_similarity_search(self):
        self.ingest()
        plan = {"collection": "tasks", "operation": "list", "filters": [{"field": "priority", "op": "eq", "value": "urgent"}]}
        result, _ = self.ask("Show me all urgent tasks", plan)
        self.assertNotEqual(result.get("mode"), "structured")

    def test_sensitive_fields_never_reach_the_llm(self):
        plan = {"collection": "users", "operation": "list", "filters": []}
        _, gen = self.ask("List all users", plan)
        context = gen.call_args[0][1]
        self.assertIn("Aisha Khan", context)
        self.assertNotIn("notarealhash", context)

    def test_ordinary_questions_do_not_use_the_exact_path(self):
        self.ingest()
        with patch.object(structured_query, "try_structured") as spy, \
             patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("What is the status of ShopEase?")
        spy.assert_not_called()

    @override_settings(STRUCTURED_QUERIES=False)
    def test_can_be_turned_off(self):
        self.ingest()
        with patch.object(structured_query, "try_structured") as spy, patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("List all departments")
        spy.assert_not_called()

    def test_collection_filter_from_the_api_skips_the_exact_path(self):
        self.ingest()
        with patch.object(structured_query, "try_structured") as spy, patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("List all departments", collections=["departments"])
        spy.assert_not_called()

    def test_numeric_comparison_and_in_operator(self):
        plan = {"collection": "departments", "operation": "list", "filters": [{"field": "floor", "op": "gte", "value": 3}]}
        result, _ = self.ask("Which departments are on floor 3 or above?", plan)
        self.assertEqual(result["total_matching"], 8)                 # floors 3 and 4: 4 + 4
        plan = {"collection": "departments", "operation": "list",
                "filters": [{"field": "name", "op": "in", "value": ["Finance", "legal"]}]}
        result, _ = self.ask("List the Finance and Legal departments", plan)
        self.assertEqual(result["total_matching"], 2)

    def test_planner_output_parsing(self):
        parse = llm_service._parse_json
        self.assertEqual(parse('Sure! {"collection": "tasks"} done'), {"collection": "tasks"})
        self.assertIsNone(parse("no json here"))
        self.assertIsNone(parse("{broken"))

    def test_llm_failure_in_planner_still_gives_the_complete_list(self):
        from rag.services.exceptions import LLMError
        with patch.object(llm_service, "plan_query", side_effect=LLMError()), \
             patch.object(llm_service, "generate", return_value="ok") as gen:
            result = rag_service.ask("List all departments")
        self.assertEqual(result["total_matching"], 20)
        self.assertIn("Finance", gen.call_args[0][1])


class LoggedTimeTests(MongoCase):
    """Logged-time questions always read nested `loggedTime` entries on 'cards', completely."""

    def setUp(self):
        super().setUp()
        scope_patch = patch.object(
            authorization_service, "resolve_scope",
            side_effect=lambda user_id, email=None: _global_scope_for_test(user_id or self.admin_id),
        )
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        self.aisha = self.db["users"].find_one({"email": "aisha@example.com"})
        self.rohan = self.db["users"].find_one({"email": "rohan@example.com"})
        self.shop = ObjectId()
        self.blog = ObjectId()
        self.db["boards"].insert_many([
            {"_id": self.shop, "name": "ShopEase"},
            {"_id": self.blog, "name": "Blog"},
        ])
        self.long_description = "Architecture notes " + ("segment " * 400).strip()
        self.db["cards"].insert_many([
            {"title": "Design landing page", "board": self.shop, "status": "in progress",
             "loggedTime": [
                 {"user": self.aisha["_id"], "hours": 2, "minutes": 30, "date": "2026-01-02",
                  "description": self.long_description, "billable": True},
                 {"user": self.rohan["_id"], "hours": 1, "minutes": 15, "date": "2026-01-03",
                  "description": "Review pass"},
             ]},
            {"title": "Write release notes", "board": self.blog, "status": "todo",
             "loggedTime": [
                 {"user": self.rohan["_id"], "hours": 4, "minutes": 0, "date": "2026-01-04",
                  "description": "BLOG-ONLY"},
                 {"user": self.aisha["_id"], "hours": 3, "minutes": 0, "date": "2026-01-05",
                  "description": "AISHA-BLOG"},
             ]},
            {"title": "Bulk tracked task", "board": self.shop, "status": "done",
             "loggedTime": [{"user": self.aisha["_id"], "hours": 0, "minutes": 15,
                             "date": f"2026-02-{index + 1:02d}", "description": f"ENTRY-{index:02d}"}
                            for index in range(25)]},
        ])
        self.db["subtasks"].insert_one({
            "title": "Ignored subtask", "loggedTime": [
                {"user": self.aisha["_id"], "hours": 99, "description": "SUBTASK-ONLY"}]})
        self.db["activities"].insert_one({
            "title": "Ignored activity", "loggedTime": [
                {"user": self.aisha["_id"], "hours": 99, "description": "ACTIVITY-ONLY"}]})

    def ask(self, question, **kwargs):
        with patch.object(rag_service.retrieval, "search") as search, \
             patch.object(llm_service, "plan_query") as planner, \
             patch.object(llm_service, "generate", return_value="short answer") as generate:
            result = rag_service.ask(question, user_id=kwargs.pop("user_id", self.admin_id), **kwargs)
        return result, search, planner, generate

    @override_settings(STRUCT_MAX_RECORDS=1, LIST_MAX_TOKENS=5)
    def test_every_card_entry_is_returned_without_llm_or_truncation(self):
        result, search, planner, generate = self.ask("Show all logged time", collections=["activities"])

        self.assertEqual(result["mode"], "structured")
        self.assertEqual(result["collection"], "cards")
        self.assertEqual(result["total_matching"], 29)
        search.assert_not_called()
        planner.assert_not_called()
        generate.assert_not_called()
        self.assertIn(self.long_description, result["answer"])
        for index in range(25):
            self.assertIn(f"ENTRY-{index:02d}", result["answer"])
        self.assertIn("Design landing page", result["answer"])
        self.assertIn("Write release notes", result["answer"])
        self.assertNotIn("SUBTASK-ONLY", result["answer"])
        self.assertNotIn("ACTIVITY-ONLY", result["answer"])

    def test_project_scope_only_returns_that_board(self):
        result, search, _, generate = self.ask("Show logged time for ShopEase")

        self.assertEqual(result["collection"], "cards")
        self.assertIn("Design landing page", result["answer"])
        self.assertIn("Bulk tracked task", result["answer"])
        self.assertNotIn("Write release notes", result["answer"])
        self.assertNotIn("BLOG-ONLY", result["answer"])
        search.assert_not_called()
        generate.assert_not_called()

    def test_person_scope_only_returns_that_persons_entries(self):
        result, _, _, _ = self.ask("Show logged time of Aisha Khan")

        self.assertIn("Aisha Khan", result["answer"])
        self.assertIn(self.long_description, result["answer"])
        self.assertNotIn("Rohan Mehta", result["answer"])
        self.assertNotIn("Review pass", result["answer"])

    def test_personal_question_uses_the_authenticated_account(self):
        result, _, _, _ = self.ask("Show my logged time", user_id=str(self.aisha["_id"]))

        self.assertIn("Aisha Khan", result["answer"])
        self.assertIn(self.long_description, result["answer"])
        self.assertNotIn("Rohan Mehta", result["answer"])

    def test_calculation_followup_keeps_the_previous_person_and_project(self):
        session_id = "10ed1234"
        owner_id = f"{self.admin_id}:admin"
        self.addCleanup(rag_service.chat_memory.clear, session_id, owner_id=owner_id)
        self.ask("Show logged time of Aisha Khan in ShopEase", session_id=session_id)
        state = rag_service.chat_memory.get_state(session_id, owner_id=owner_id)
        self.assertEqual(state["turns"][0]["question"], "Show logged time of Aisha Khan in ShopEase")

        result, search, _, _ = self.ask("calculate total", session_id=session_id)

        self.assertEqual(result["mode"], "structured")
        self.assertIn("Total: 8h 45m", result["answer"])
        self.assertIn("Aisha Khan", result["answer"])
        self.assertIn("ShopEase", result["answer"])
        self.assertNotIn("Rohan Mehta", result["answer"])
        self.assertNotIn("BLOG-ONLY", result["answer"])
        self.assertNotIn("AISHA-BLOG", result["answer"])
        search.assert_not_called()

    def test_no_matching_entries_never_falls_back_to_similarity_search(self):
        self.db["cards"].delete_many({})
        result, search, _, _ = self.ask("Show all logged time")

        self.assertEqual(result["mode"], "structured")
        self.assertEqual(result["total_matching"], 0)
        self.assertIn("No logged-time entries were found", result["answer"])
        search.assert_not_called()


@override_settings(STRUCT_PLANNER="on")
class SpeedAndResilienceTests(MongoCase):
    """The exact path must stay fast and must never fail just because the language model does."""

    structured = True

    def setUp(self):
        super().setUp()
        scope_patch = patch.object(
            authorization_service, "resolve_scope",
            side_effect=lambda user_id, email=None: _global_scope_for_test(user_id or self.admin_id),
        )
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])

    def ask(self, question, plan=None, generate="ok", **kw):
        with patch.object(llm_service, "plan_query", return_value=plan) as pq, \
             patch.object(llm_service, "generate", return_value=generate) as gen:
            return rag_service.ask(question, **kw), pq, gen

    # ---- the reported failure: "give me list of all departments" -> LLM error
    def test_llm_failure_still_returns_the_complete_list(self):
        from rag.services.exceptions import LLMError
        with patch.object(llm_service, "plan_query", return_value=None), \
             patch.object(llm_service, "generate", side_effect=LLMError()):
            result = rag_service.ask("give me list of all departments")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["total_matching"], 20)
        for name in DEPARTMENTS:
            self.assertIn(f"- {name}", result["answer"])
        self.assertIn("20 matching record(s)", result["answer"])
        self.assertEqual(len(result["sources"]), 20)

    def test_degraded_count_and_truncated_list(self):
        from rag.services.exceptions import LLMError
        plan = {"collection": "departments", "operation": "count", "filters": []}
        with patch.object(llm_service, "plan_query", return_value=plan), patch.object(llm_service, "generate", side_effect=LLMError()):
            self.assertIn("20 matching record(s)", rag_service.ask("How many departments are there?")["answer"])
        with override_settings(STRUCT_MAX_RECORDS=3), patch.object(llm_service, "plan_query", return_value=None), \
             patch.object(llm_service, "generate", side_effect=LLMError()):
            answer = rag_service.ask("List all departments")["answer"]
        self.assertIn("Showing the first 3 of 20", answer)

    def test_configuration_errors_are_not_hidden_by_the_fallback(self):
        from rag.services.exceptions import ConfigError
        with patch.object(llm_service, "plan_query", return_value=None), \
             patch.object(llm_service, "generate", side_effect=ConfigError("bad key")):
            with self.assertRaises(ConfigError):
                rag_service.ask("List all departments")

    # ---- speed
    def test_catalog_is_saved_and_reloaded_without_probing(self):
        from pathlib import Path
        from django.conf import settings
        structured_query.build_catalog()
        self.assertTrue((Path(settings.VECTOR_STORE_DIR) / "catalog.json").exists())
        structured_query._memory.clear()
        with patch.object(relationships.RelationshipResolver, "_infer", side_effect=AssertionError("probed")):
            catalog = structured_query.build_catalog()
        self.assertEqual(catalog["tasks"]["fields"]["priority"]["types"], {"str"})
        self.assertEqual(catalog["tasks"]["fields"]["assignedTo"]["ref"], "users")
        self.assertIn("departments", catalog)

    def test_ingest_writes_the_catalog(self):
        from pathlib import Path
        from django.conf import settings
        self.ingest()
        self.assertTrue((Path(settings.VECTOR_STORE_DIR) / "catalog.json").exists())

    def test_reference_filters_need_no_probing_at_question_time(self):
        structured_query.build_catalog()
        structured_query.reset_for_tests()       # simulate a server restart: only the saved file remains
        plan = {"collection": "tasks", "operation": "list",
                "filters": [{"field": "assignedTo", "op": "eq", "value": "Rohan Mehta"}]}
        with patch.object(relationships.RelationshipResolver, "_infer", side_effect=AssertionError("probed")):
            result, _, gen = self.ask("What tasks are assigned to Rohan Mehta?", plan)
        self.assertEqual(result["total_matching"], 2)
        self.assertIn("Assigned To: Rohan Mehta", gen.call_args[0][1])

    def test_the_plan_is_cached_so_the_planner_runs_once(self):
        plan = {"collection": "departments", "operation": "list", "filters": []}
        _, pq, _ = self.ask("List all departments", plan)
        with patch.object(llm_service, "plan_query", side_effect=AssertionError("planner called again")), \
             patch.object(llm_service, "generate", return_value="ok"):
            again = rag_service.ask("list   ALL departments")
        self.assertEqual(again["total_matching"], 20)
        self.assertEqual(pq.call_count, 1)

    def test_a_failed_planner_call_is_not_cached(self):
        from rag.services.exceptions import LLMError
        with patch.object(llm_service, "plan_query", side_effect=LLMError()), patch.object(llm_service, "generate", return_value="ok"):
            rag_service.ask("List all departments")
        plan = {"collection": "departments", "operation": "count", "filters": []}
        result, pq, _ = self.ask("List all departments", plan)
        self.assertEqual(pq.call_count, 1)
        self.assertEqual(result["records_used"], 10)

    def test_show_limits_the_fields_sent_to_the_llm(self):
        plan = {"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}
        _, _, gen = self.ask("Give me the names of all departments", plan)
        context = gen.call_args[0][1]
        self.assertIn("[1] Name: Engineering", context)
        self.assertNotIn("Floor", context)
        plan["show"] = ["nonexistent"]
        structured_query._plans.clear()
        _, _, gen = self.ask("Give me the names of all departments again", plan)
        self.assertIn("Floor", gen.call_args[0][1])         # unknown "show" fields are ignored, not fatal

    def test_timings_show_where_the_time_goes(self):
        result, _, _ = self.ask("List all departments", {"collection": "departments", "filters": []})
        self.assertEqual(set(result["timings_ms"]), {"plan", "db", "llm"})


class LlmErrorMessageTests(SimpleTestCase):
    """Every failure of the provider becomes a specific, actionable message (and never a traceback)."""

    def raised_by(self, name, status=None, message="boom"):
        error = type(name, (Exception,), {})(message)
        error.status_code = status
        client = type("C", (), {})()
        client.chat = type("Ch", (), {})()
        client.chat.completions = type("Co", (), {"create": lambda self, **kw: (_ for _ in ()).throw(error)})()
        with patch.object(llm_service, "_get_client", return_value=client):
            try:
                llm_service._chat([{"role": "user", "content": "x"}], 10)
            except Exception as exc:
                return exc
        raise AssertionError("no error raised")

    def test_messages(self):
        from rag.services.exceptions import ConfigError, LLMError
        cases = [("AuthenticationError", 401, ConfigError, "API key was rejected"),
                 ("PermissionDeniedError", 403, ConfigError, "denied"),
                 ("NotFoundError", 404, ConfigError, "was not found"),
                 ("RateLimitError", 429, LLMError, "rate-limited"),
                 ("APIStatusError", 413, LLMError, "too large"),
                 ("APITimeoutError", None, LLMError, "took longer"),
                 ("APIConnectionError", None, LLMError, "Could not connect"),
                 ("InternalServerError", 503, LLMError, "temporary problem (HTTP 503)"),
                 ("BadRequestError", 400, LLMError, "rejected the request: boom"),
                 ("WeirdError", None, LLMError, "unavailable right now")]
        for name, status, kind, text in cases:
            error = self.raised_by(name, status)
            self.assertIsInstance(error, kind, name)
            self.assertIn(text, error.message, name)

    def test_reasoning_effort_is_dropped_if_the_provider_rejects_it(self):
        from types import SimpleNamespace
        calls = []

        class BadRequestError(Exception):
            pass

        class Completions:
            def create(self, **kw):
                calls.append(dict(kw))
                if "reasoning_effort" in kw:
                    raise BadRequestError("unknown parameter: reasoning_effort")
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch.object(llm_service, "_get_client", return_value=client), override_settings(LLM_REASONING_EFFORT="low"):
            self.assertEqual(llm_service._chat([{"role": "user", "content": "x"}], 10), "hi")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", calls[1])

    def test_planner_uses_its_own_model_when_configured(self):
        from types import SimpleNamespace
        seen = []

        class Completions:
            def create(self, **kw):
                seen.append(kw["model"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"collection": null}'))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch.object(llm_service, "_get_client", return_value=client), \
             override_settings(PLANNER_MODEL="small-fast", LLM_MODEL="big-slow"):
            llm_service.plan_query("q", "catalog")
            llm_service._chat([{"role": "user", "content": "x"}], 5)
        self.assertEqual(seen, ["small-fast", "big-slow"])


# =============================================================== Ollama (local model)
import httpx  # noqa: E402

OLLAMA = dict(LLM_PROVIDER="ollama", LLM_MODEL="gemma3:4b", OLLAMA_HOST="http://ollama.test:11434",
              OLLAMA_NUM_CTX=8192, OLLAMA_KEEP_ALIVE="30m", LLM_TIMEOUT=30, LLM_TEMPERATURE="0",
              LLM_MAX_TOKENS=600, LLM_API_KEY="")


def fake_ollama(handler):
    """Make llm_service talk to an in-process fake Ollama instead of the network."""
    def factory():
        return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)
    return patch.object(llm_service, "_http", factory)


def reply(content, **extra):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}, "done": True, **extra})


@override_settings(**OLLAMA)
class OllamaClientTests(SimpleTestCase):
    def test_request_uses_native_api_with_context_window_and_no_api_key(self):
        seen = {}

        def handler(request):
            seen["url"], seen["json"] = str(request.url), json.loads(request.content)
            return reply("  There are 20 departments.  ")

        with fake_ollama(handler):
            answer = llm_service.generate("How many?", "EXACT QUERY RESULT ...")
        body = seen["json"]
        self.assertEqual(answer, "There are 20 departments.")
        self.assertEqual(seen["url"], "http://ollama.test:11434/api/chat")
        self.assertEqual(body["model"], "gemma3:4b")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["keep_alive"], "30m")
        self.assertEqual(body["options"], {"num_ctx": 8192, "num_predict": 600, "temperature": 0.0})
        self.assertNotIn("format", body)
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertIn("EXACT QUERY RESULT", body["messages"][0]["content"])
        self.assertLess(len(body["messages"][0]["content"]), 2200)        # compact prompt for a small model
        self.assertIn("Question: How many?", body["messages"][1]["content"])

    def test_max_tokens_override_is_passed_as_num_predict(self):
        seen = {}
        with fake_ollama(lambda r: (seen.update(json.loads(r.content)), reply("ok"))[1]):
            llm_service.generate("q", "ctx", max_tokens=900)
        self.assertEqual(seen["options"]["num_predict"], 900)

    def test_planner_asks_for_json_mode_and_parses_the_plan(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}')

        with fake_ollama(handler):
            plan = llm_service.plan_query("list departments", "- departments (20 records): name")
        self.assertEqual(seen["format"], "json")
        self.assertEqual(plan["collection"], "departments")
        self.assertIn("Catalog:", seen["messages"][1]["content"])

    def test_hosted_prompt_is_unchanged_for_other_providers(self):
        with override_settings(LLM_PROVIDER="openai"):
            self.assertIs(llm_service._system_prompt(), llm_service.SYSTEM_PROMPT)
        self.assertIs(llm_service._system_prompt(), llm_service.SYSTEM_PROMPT_COMPACT)
        self.assertIn(llm_service.NOT_FOUND_ANSWER, llm_service.SYSTEM_PROMPT_COMPACT)

    def test_failures_become_actionable_messages(self):
        from rag.services.exceptions import ConfigError, LLMError

        def down(request):
            raise httpx.ConnectError("refused")

        with fake_ollama(down), self.assertRaises(LLMError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("Could not reach Ollama at http://ollama.test:11434", cm.exception.message)
        self.assertIn("ollama serve", cm.exception.message)

        missing = lambda r: httpx.Response(404, json={"error": 'model "gemma3:4b" not found, try pulling it first'})
        with fake_ollama(missing), self.assertRaises(ConfigError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("ollama pull gemma3:4b", cm.exception.message)

        def slow(request):
            raise httpx.ReadTimeout("slow")

        with fake_ollama(slow), self.assertRaises(LLMError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("took longer than 30s", cm.exception.message)

        broken = lambda r: httpx.Response(500, json={"error": "model runner has unexpectedly stopped"})
        with fake_ollama(broken), self.assertRaises(LLMError) as cm:
            llm_service.generate("q", "c")
        self.assertIn("HTTP 500", cm.exception.message)
        self.assertIn("unexpectedly stopped", cm.exception.message)

        with fake_ollama(lambda r: httpx.Response(200, text="not json")), self.assertRaises(LLMError):
            llm_service.generate("q", "c")

    def test_status_reports_running_and_model_installed(self):
        def tags(models):
            return patch("httpx.get", return_value=httpx.Response(200, json={"models": [{"name": m} for m in models]}))

        with tags(["gemma3:4b", "llama3:8b"]):
            self.assertEqual(llm_service.status(), {"provider": "ollama", "model": "gemma3:4b",
                                                    "reachable": True, "model_available": True})
        with tags(["llama3:8b"]):
            self.assertEqual(llm_service.status()["model_available"], False)
        with patch("httpx.get", side_effect=httpx.ConnectError("down")):
            info = llm_service.status()
        self.assertEqual((info["reachable"], info["model_available"]), (False, False))
        with override_settings(LLM_PROVIDER="openai"):
            self.assertEqual(llm_service.status(), {"provider": "openai", "model": "gemma3:4b"})

    def test_warm_up_loads_the_model_with_an_empty_message_list(self):
        seen = {}
        with fake_ollama(lambda r: (seen.update(json.loads(r.content)), reply(""))[1]):
            llm_service.warm_up()
        self.assertEqual((seen["model"], seen["messages"], seen["keep_alive"]), ("gemma3:4b", [], "30m"))
        with override_settings(LLM_PROVIDER="openai"), patch.object(llm_service, "_ollama_post") as post:
            llm_service.warm_up()
        post.assert_not_called()


@override_settings(STRUCT_PLANNER="on")
class OllamaEndToEndTests(MongoCase):
    """Whole pipeline with a fake Ollama: planner (JSON mode) then answer, on the 20-departments case."""

    structured = True

    def setUp(self):
        super().setUp()
        scope_patch = patch.object(
            authorization_service, "resolve_scope",
            side_effect=lambda user_id, email=None: _global_scope_for_test(user_id or self.admin_id),
        )
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])
        overrides = override_settings(**OLLAMA)
        overrides.enable()
        self.addCleanup(overrides.disable)
        self.requests = []

    def handler(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("format") == "json":
            return reply('{"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}')
        return reply("There are 20 departments, listed below.")

    def test_list_question_runs_planner_then_answer_with_all_names(self):
        with fake_ollama(self.handler):
            result = rag_service.ask("give me list of all departments")
        self.assertEqual(result["mode"], "structured")
        self.assertEqual(result["total_matching"], 20)
        self.assertFalse(result["degraded"])
        planner, answer = self.requests
        self.assertEqual(planner["format"], "json")
        user_prompt = answer["messages"][1]["content"]
        for name in DEPARTMENTS:
            self.assertIn(name, user_prompt)
        self.assertNotIn("Floor", user_prompt)                         # only the requested field was sent
        self.assertEqual(answer["options"]["num_ctx"], 8192)

    def test_when_ollama_is_down_the_complete_list_is_still_returned(self):
        def down(request):
            raise httpx.ConnectError("refused")

        with fake_ollama(down):
            result = rag_service.ask("give me list of all departments")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["total_matching"], 20)
        for name in DEPARTMENTS:
            self.assertIn(f"- {name}", result["answer"])

    @override_settings(STRUCT_PLANNER="off")
    def test_planner_can_be_switched_off_for_speed(self):
        with fake_ollama(self.handler):
            result = rag_service.ask("give me list of all departments")
        self.assertEqual(result["total_matching"], 20)
        self.assertEqual(len(self.requests), 1)                        # only the answer call, no planning call
        self.assertNotIn("format", self.requests[0])

    def test_health_reports_the_local_model(self):
        with patch("httpx.get", return_value=httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})):
            health = rag_service.health()
        self.assertEqual(health["llm"]["model_available"], True)
        self.assertEqual(health["llm_model"], "gemma3:4b")

    def test_catalog_text_shrinks_for_small_models(self):
        catalog = structured_query.build_catalog()
        full = structured_query.catalog_text(catalog, limit=100000)
        tiny = structured_query.catalog_text(catalog, limit=len(full) // 2)
        self.assertLessEqual(len(tiny), len(full) // 2)
        no_values = structured_query.catalog_text(catalog, limit=len(full) - 1)
        self.assertNotIn("[Engineering", no_values)


# =============================================================== Gemini (Google, OpenAI-compatible endpoint)
from io import StringIO  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from django.core.management import call_command  # noqa: E402
from django.core.management.base import CommandError  # noqa: E402

GEMINI = dict(LLM_PROVIDER="gemini", LLM_MODEL="gemini-3.5-flash-lite", LLM_API_KEY="SECRET-KEY-123",
              LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/", LLM_TEMPERATURE="0",
              LLM_MAX_TOKENS_PARAM="max_tokens", LLM_REASONING_EFFORT="", LLM_MAX_TOKENS=600, STRUCT_PLANNER="on")


def fake_openai_client(content="ok", finish_reason="stop", error=None, seen=None):
    """Stands in for the openai client; records the request fields."""
    class Completions:
        def create(self, **kw):
            if seen is not None:
                seen.append(kw)
            if error is not None:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                                                            message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def provider_error(name, status, message="boom"):
    error = type(name, (Exception,), {})(message)
    error.status_code = status
    return error


@override_settings(**GEMINI)
class GeminiTests(SimpleTestCase):
    def test_request_never_sends_temperature_and_uses_max_tokens(self):
        seen = []
        with patch.object(llm_service, "_get_client", return_value=fake_openai_client("Hello", seen=seen)):
            self.assertEqual(llm_service.generate("q", "ctx", max_tokens=900), "Hello")
            llm_service.plan_query("q", "catalog")
        for request in seen:
            self.assertNotIn("temperature", request)            # deprecated on Gemini 3.x, even though LLM_TEMPERATURE=0
            self.assertNotIn("max_completion_tokens", request)
            self.assertNotIn("reasoning_effort", request)       # default thinking level is the fastest
            self.assertEqual(request["model"], "gemini-3.5-flash-lite")
            self.assertEqual(request["messages"][-1]["role"], "user")   # a trailing model turn is rejected by Gemini 3.5
        self.assertEqual(seen[0]["max_tokens"], 900)

    def test_other_providers_still_send_temperature_and_max_completion_tokens(self):
        seen = []
        with override_settings(LLM_PROVIDER="openai", LLM_MAX_TOKENS_PARAM="max_completion_tokens"), \
             patch.object(llm_service, "_get_client", return_value=fake_openai_client("Hi", seen=seen)):
            llm_service.generate("q", "ctx")
        self.assertEqual(seen[0]["temperature"], 0.0)
        self.assertIn("max_completion_tokens", seen[0])

    def test_empty_replies_are_reported_not_shown_as_not_found(self):
        from rag.services.exceptions import LLMError
        cases = [("length", "ran out of output tokens"), ("content_filter", "safety filter"), ("stop", "empty answer")]
        for reason, text in cases:
            client = fake_openai_client(content=None, finish_reason=reason)
            with patch.object(llm_service, "_get_client", return_value=client), self.assertRaises(LLMError) as cm:
                llm_service.generate("q", "ctx")
            self.assertIn(text, cm.exception.message, reason)

    def test_misconfiguration_messages_name_the_endpoint(self):
        from rag.services.exceptions import ConfigError
        cases = [(provider_error("NotFoundError", 404), "was not found at generativelanguage.googleapis.com"),
                 (provider_error("BadRequestError", 400, "API key not valid. Please pass a valid API key."),
                  "rejected by generativelanguage.googleapis.com"),
                 (provider_error("AuthenticationError", 401), "rejected by generativelanguage.googleapis.com")]
        for error, text in cases:
            with patch.object(llm_service, "_get_client", return_value=fake_openai_client(error=error)), \
                 self.assertRaises(ConfigError) as cm:
                llm_service.generate("q", "ctx")
            self.assertIn(text, cm.exception.message)
            self.assertNotIn("SECRET-KEY-123", cm.exception.message)

    def test_a_groq_url_with_a_gemini_model_is_explained(self):
        from rag.services.exceptions import ConfigError
        error = provider_error("NotFoundError", 404, "The model `gemini-3.5-flash-lite` does not exist")
        with override_settings(LLM_BASE_URL="https://api.groq.com/openai/v1"), \
             patch.object(llm_service, "_get_client", return_value=fake_openai_client(error=error)), \
             self.assertRaises(ConfigError) as cm:
            llm_service.generate("q", "ctx")
        self.assertIn("api.groq.com", cm.exception.message)
        self.assertIn("LLM_BASE_URL belongs to the same provider", cm.exception.message)

    @override_settings(LLM_API_KEY="")
    def test_missing_key_message_mentions_gemini_key_name(self):
        from rag.services.exceptions import ConfigError
        llm_service._client = None
        with self.assertRaises(ConfigError) as cm:
            llm_service.generate("q", "ctx")
        self.assertIn("GEMINI_API_KEY", cm.exception.message)


@override_settings(**GEMINI)
class CheckLlmCommandTests(SimpleTestCase):
    def run_command(self, client):
        out = StringIO()
        with patch.object(llm_service, "_get_client", return_value=client):
            try:
                call_command("check_llm", stdout=out)
            except CommandError as exc:
                return out.getvalue(), exc
        return out.getvalue(), None

    def test_success_report_never_prints_the_key(self):
        client = fake_openai_client('{"collection": "departments", "operation": "list", "filters": []}')
        output, error = self.run_command(client)
        self.assertIsNone(error)
        self.assertIn("Provider        : gemini", output)
        self.assertIn("Model           : gemini-3.5-flash-lite", output)
        self.assertIn("generativelanguage.googleapis.com", output)
        self.assertIn("Temperature     : not sent", output)
        self.assertIn("OK      1/2", output)
        self.assertIn("OK      2/2", output)
        self.assertIn("configured correctly", output)
        self.assertNotIn("SECRET-KEY-123", output)

    def test_failure_report_says_what_is_wrong(self):
        client = fake_openai_client(error=provider_error("NotFoundError", 404))
        output, error = self.run_command(client)
        self.assertIsNotNone(error)
        self.assertIn("FAILED  1/2", output)
        self.assertIn("was not found at generativelanguage.googleapis.com", output)
        self.assertNotIn("SECRET-KEY-123", output)

    def test_plain_text_plan_is_a_warning_not_a_failure(self):
        output, error = self.run_command(fake_openai_client("sure, here you go"))
        self.assertIsNone(error)
        self.assertIn("WARNING 2/2", output)


class GeminiEndToEndTests(MongoCase):
    """Whole pipeline through the (fake) Gemini endpoint, on the 20-departments case."""

    structured = True

    def setUp(self):
        super().setUp()
        self.db["departments"].insert_many([{"name": n, "floor": i % 5} for i, n in enumerate(DEPARTMENTS)])
        overrides = override_settings(**GEMINI)
        overrides.enable()
        self.addCleanup(overrides.disable)

    def test_list_question_end_to_end(self):
        seen = []

        class Completions:
            def create(self, **kw):
                seen.append(kw)
                if "JSON query plan" in kw["messages"][0]["content"]:
                    content = '{"collection": "departments", "operation": "list", "filters": [], "show": ["name"]}'
                else:
                    content = "There are 20 departments."
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch.object(llm_service, "_get_client", return_value=client):
            result = rag_service.ask("give me list of all departments")
        self.assertEqual(result["answer"], "There are 20 departments.")
        self.assertEqual(result["total_matching"], 20)
        self.assertEqual(len(seen), 2)                                   # planner + answer
        for request in seen:
            self.assertNotIn("temperature", request)
            self.assertIn("max_tokens", request)
        for name in DEPARTMENTS:
            self.assertIn(name, seen[1]["messages"][1]["content"])

    def test_gemini_outage_still_returns_the_complete_list(self):
        client = fake_openai_client(error=provider_error("InternalServerError", 503))
        with patch.object(llm_service, "_get_client", return_value=client):
            result = rag_service.ask("give me list of all departments")
        self.assertTrue(result["degraded"])
        for name in DEPARTMENTS:
            self.assertIn(f"- {name}", result["answer"])

    def test_an_invalid_key_is_not_hidden_by_the_fallback(self):
        from rag.services.exceptions import ConfigError
        client = fake_openai_client(error=provider_error("BadRequestError", 400, "API key not valid"))
        with patch.object(llm_service, "_get_client", return_value=client), self.assertRaises(ConfigError) as cm:
            rag_service.ask("give me list of all departments")
        self.assertIn("rejected by generativelanguage.googleapis.com", cm.exception.message)
