"""
Exact answers for LIST and COUNT questions.

Similarity search returns the nearest few records, so it can never reliably list or count "all" of something.
For such questions this module queries MongoDB directly:

  1. a CATALOG of collections and fields is loaded (built once at ingest time, saved to vector_store/catalog.json)
  2. the LLM turns the question into a small JSON PLAN: {collection, operation, filters, show}   (plans are cached)
  3. the plan is VALIDATED strictly (known collection and fields, whitelisted operators). Any doubt rejects the
     whole plan: a condition is never silently dropped, because that would make the "exact" count wrong
  4. count_documents() gives the exact total, find() gives the records (up to STRUCT_MAX_RECORDS)
  5. the LLM phrases the answer. If it is unavailable, the caller still shows the plain result

The LLM never writes MongoDB syntax. Filters on reference fields accept NAMES ("assigned to Rohan Mehta"):
the name is resolved to ids in the referenced collection first.

Speed notes: the slow parts (probing which collection each id field points to, sampling field values, reading
the field list) happen once, at ingest, and are saved. A question then costs one planner call (cached for
repeated questions), one count, one find and the answer call. Time is reported as plan / db / llm.
"""
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from bson import ObjectId
from django.conf import settings

from . import authorization_service, cache_service, document_converter, embedding_store, llm_service, mongo_service, relationships
from .document_converter import EXTRA_NAME_FIELDS, HEX24, NAME_FIELDS, humanize_key, is_sensitive, strip_id_suffix, strip_html
from .exceptions import ChatbotError
from .relationships import _norm, _singular

logger = logging.getLogger(__name__)

CATALOG_FILE = "catalog.json"
OPS = {"eq", "ne", "in", "contains", "gt", "gte", "lt", "lte"}
MAX_FILTERS = 5
SAMPLE_DOCS = 25
MAX_FIELDS = 40
MAX_SHOW = 8
MAX_RECORD_CHARS = 400

_memory = cache_service.TTLCache(maxsize=4, ttl=3600)      # "catalog" -> (file_stamp, data); "names"
_plans = cache_service.TTLCache(maxsize=500, ttl=3600)                # normalised question -> {"plan": ...}
_name_ids = cache_service.TTLCache(maxsize=256, ttl=300)              # (collection, name) -> ids
_resolver = None
_lock = threading.Lock()
_refreshing = False


@dataclass
class ExactResult:
    collection: str
    total: int
    records: list                      # [{document_id, name, text}]
    conditions: str
    complete: bool
    source: str                        # "planner" | "name-match"
    operation: str = "list"
    context: str = ""
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    direct_answer: str = ""          # complete answer produced from the database, never shortened by an LLM


def reset_for_tests():
    global _resolver
    _memory.clear()
    _plans.clear()
    _name_ids.clear()
    _resolver = None


# ----------------------------------------------------------- collections
def known_collections() -> list:
    names = _memory.get("names")
    if names is None:
        names = mongo_service.list_collections()
        _memory.set("names", names)
    return names


def _exec_resolver(catalog: dict):
    """Resolver for question time: targets come from the catalog, so no probing queries are needed."""
    global _resolver
    names = known_collections()
    with _lock:
        if _resolver is None or _resolver.collections != names:
            _resolver = relationships.RelationshipResolver(mongo_service.get_db(), names)
        for cname, info in catalog.items():
            for path, meta in info["fields"].items():
                if meta.get("ref"):
                    _resolver.seed_target(cname, path, meta["ref"])
        _resolver.reset_names()
        return _resolver


# --------------------------------------------------------------- catalog
def _scalars(doc, path="", depth=0):
    """Yield (field_path, scalar_value) for the leaves of a document (sensitive keys skipped)."""
    if isinstance(doc, dict):
        if depth >= 3:
            return
        for key, value in doc.items():
            key = str(key)
            if (depth == 0 and key in ("_id", "__v")) or is_sensitive(key):
                continue
            yield from _scalars(value, f"{path}.{key}" if path else key, depth + 1)
    elif isinstance(doc, (list, tuple)):
        for item in doc[:5]:
            yield from _scalars(item, path, depth + 1)
    elif doc is not None and not isinstance(doc, (bytes, bytearray)):
        yield path, doc


def _build(resolver=None) -> dict:
    """Sample every collection. This is the slow part (it probes id fields), so it runs at ingest, not per question."""
    db = mongo_service.get_db()
    names = [name for name in mongo_service.list_collections()
             if not mongo_service.is_blocked_collection(name)]
    _memory.set("names", names)
    resolver = resolver or relationships.RelationshipResolver(db, names)
    catalog = {}
    for name in names:
        try:
            total = db[name].estimated_document_count()
            samples = list(db[name].find({}).limit(SAMPLE_DOCS))
        except Exception as exc:
            logger.warning("Catalog: skipping %s (%s)", name, type(exc).__name__)
            continue
        if not samples:
            continue
        resolver.prefetch(name, samples)
        fields = {}
        for doc in samples:
            for path, value in _scalars(doc):
                is_id = isinstance(value, ObjectId) or (isinstance(value, str) and HEX24.match(value))
                target = resolver.target_of(name, path) if is_id else None
                if is_id and not target:
                    continue                      # an id we cannot translate to names: not filterable
                if path not in fields and len(fields) >= MAX_FIELDS:
                    continue
                meta = fields.setdefault(path, {"values": [], "types": set(), "ref": None})
                if target:
                    meta["ref"] = target
                    meta["types"].add("ref")
                    continue
                kind = "bool" if isinstance(value, bool) else "num" if isinstance(value, (int, float)) else "str"
                meta["types"].add(kind)
                text = " ".join(str(value).split())[:30]
                if kind == "str" and text and text not in meta["values"] and len(meta["values"]) < 6:
                    meta["values"].append(text)
        valid = set(fields)
        try:   # field names seen on ALL ingested documents, not just the sample
            valid |= embedding_store.distinct_field_names(name)
        except Exception:
            pass
        catalog[name] = {"count": total, "fields": fields, "valid": sorted(valid)}
    return catalog


def _catalog_path() -> Path:
    return Path(settings.VECTOR_STORE_DIR) / CATALOG_FILE


def _stamp(path: Path):
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _read(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))["catalog"]
        data = {name: info for name, info in data.items()
                if not mongo_service.is_blocked_collection(name)}
        for info in data.values():
            for meta in info["fields"].values():
                meta["types"] = set(meta["types"])
        return data
    except Exception as exc:
        logger.warning("Saved catalog unreadable (%s); rebuilding.", type(exc).__name__)
        return None


def _write(path: Path, catalog: dict):
    plain = {n: {**info, "fields": {p: {**m, "types": sorted(m["types"])} for p, m in info["fields"].items()}}
             for n, info in catalog.items()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps({"catalog": plain}), encoding="utf-8")
        tmp.replace(path)
        return _stamp(path)
    except OSError as exc:
        logger.warning("Could not save the catalog (%s); keeping it in memory only.", type(exc).__name__)
        return None


def _refresh_in_background():
    global _refreshing
    if _refreshing:
        return
    _refreshing = True

    def work():
        global _refreshing
        try:
            build_catalog(force=True)
        except Exception:
            logger.exception("Background catalog refresh failed")
        finally:
            _refreshing = False

    threading.Thread(target=work, daemon=True, name="catalog-refresh").start()


def build_catalog(force: bool = False, resolver=None) -> dict:
    """
    Return the catalog: memory -> saved file -> build (slow, once). `ingest` calls this with force=True.
    A file older than CATALOG_MAX_AGE_HOURS is still used right away and refreshed in the background.
    """
    path = _catalog_path()
    stamp = _stamp(path)
    mem = _memory.get("catalog")
    if not force and mem is not None and mem[0] == stamp:
        return mem[1]
    if not force and stamp is not None:
        data = _read(path)
        if data is not None:
            _memory.set("catalog", (stamp, data))
            _plans.clear()
            if time.time() - path.stat().st_mtime > settings.CATALOG_MAX_AGE_HOURS * 3600:
                _refresh_in_background()
            return data
    started = time.perf_counter()
    catalog = _build(resolver)
    new_stamp = _write(path, catalog)
    _memory.set("catalog", (new_stamp, catalog))
    _plans.clear()
    logger.info("Catalog built for %d collections in %.1fs", len(catalog), time.perf_counter() - started)
    return catalog


CORE_COLLECTIONS = (
    "workspaces", "users", "workspacememberships", "departments",
    "boards", "cards", "roles", "lists", "tasks", "projects",
    "salesrows", "announcements", "milestones", "subtasks", "subtasknanos", "comments",
    "attachments", "holidays", "notifications"
)

NOISY_FIELD_PREFIXES = (
    "pushSubscription", "pushSubscriptions", "settings.notifications",
    "avatarPublicId", "avatarMetadata", "coverImage.thumbnailUrl",
    "coverImage.publicId", "icon.publicId", "icon.smallUrl", "icon.mediumUrl", "icon.largeUrl"
)


def _is_noisy_field(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in NOISY_FIELD_PREFIXES)


def catalog_text(catalog: dict, limit: int = None, question: str = "") -> str:
    """Compact description of collections and fields for the planner. Core collections and
    question-relevant collections are prioritized so they are never truncated."""
    limit = limit or settings.CATALOG_CHARS
    q_words = set(re.findall(r"[a-z0-9]+", question.lower())) if question else set()
    aliases = get_aliases()
    for w in list(q_words):
        if w in aliases:
            q_words.add(aliases[w])

    def sort_key(item):
        name, info = item
        norm_name = _norm(name)
        if any(w in norm_name or _singular(norm_name) == w for w in q_words):
            return (0, -info.get("count", 0))
        if name in CORE_COLLECTIONS:
            return (1, CORE_COLLECTIONS.index(name))
        if name.startswith("_") or info.get("count", 0) == 0:
            return (9, 0)
        return (2, -info.get("count", 0))

    sorted_items = sorted(catalog.items(), key=sort_key)
    prio0 = [item for item in sorted_items if sort_key(item)[0] == 0]
    prio1 = [item for item in sorted_items if sort_key(item)[0] == 1][:6]
    prio2 = [item for item in sorted_items if sort_key(item)[0] == 2][:2]
    pruned_items = prio0 + [item for item in prio1 if item not in prio0]
    if len(pruned_items) < 8:
        pruned_items += [item for item in prio2 if item not in pruned_items][:8 - len(pruned_items)]
    ordered_catalog = OrderedDict(pruned_items)

    def render(with_values: bool) -> str:
        lines = []
        if aliases:
            active_cols = set(ordered_catalog)
            distinct_aliases = sorted({f"'{k}' -> '{v}'" for k, v in aliases.items() if k != v and (v in active_cols or k in q_words)})
            if distinct_aliases:
                lines.append(f"Domain Aliases: {'; '.join(distinct_aliases)}")
        for name, info in ordered_catalog.items():
            if mongo_service.is_blocked_collection(name):
                continue
            if name.startswith("_") and not any(w in name for w in q_words):
                continue
            parts = []
            for path, meta in info["fields"].items():
                if _is_noisy_field(path):
                    continue
                if meta["ref"]:
                    parts.append(f"{path} (reference to {meta['ref']}; filter it by that record's NAME)")
                elif meta["values"] and with_values:
                    parts.append(f"{path} [{', '.join(meta['values'])}]")
                else:
                    parts.append(path)
            if parts:
                lines.append(f"- {name} ({info['count']} records): " + "; ".join(parts))
        return "\n".join(lines)

    text = render(True)
    if len(text) <= limit:
        return text
    text_no_vals = render(False)
    return text_no_vals[:limit]


# ------------------------------------------------------------------ plans
def _to_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def get_aliases() -> dict:
    return getattr(settings, "COLLECTION_ALIASES", {})


def validate_plan(plan, catalog: dict):
    """Return a clean plan or None. Strict on purpose: reject rather than guess."""
    if not isinstance(plan, dict):
        return None
    action = plan.get("action")
    if action in {"logged_time", "employee_work", "project_team", "leave_records"}:
        clean_action = {"action": action}
        for k in ("person", "project", "date", "extremum", "status", "leave_type"):
            val = plan.get(k)
            if isinstance(val, str) and val.strip():
                clean_action[k] = val.strip()
            elif isinstance(val, (int, float)):
                clean_action[k] = str(val)
        if "verify" in plan:
            clean_action["verify"] = bool(plan.get("verify"))
        return clean_action

    collection = plan.get("collection")
    if not isinstance(collection, str):
        return None
    # Resolve aliases (e.g. 'projects' -> 'boards', 'tasks' -> 'cards')
    aliases = get_aliases()
    col_lower = collection.strip().lower()
    if col_lower in aliases and aliases[col_lower] in catalog:
        collection = aliases[col_lower]
    elif collection not in catalog:
        return None
    if mongo_service.is_blocked_collection(collection):
        return None
    operation = plan.get("operation", "list")
    operation = operation if operation in ("list", "count") else "list"
    info = catalog[collection]
    valid = set(info.get("valid") or info["fields"])
    clean = []
    for item in (plan.get("filters") or [])[:MAX_FILTERS + 1]:
        if len(clean) >= MAX_FILTERS or not isinstance(item, dict):
            return None
        fname, op, value = item.get("field"), item.get("op", "eq"), item.get("value")
        if fname not in valid or op not in OPS or value is None or isinstance(value, dict):
            return None
        if op == "in" and not (isinstance(value, list) and value and all(isinstance(v, (str, int, float)) for v in value)):
            return None
        if op != "in" and isinstance(value, list):
            return None
        if op in ("gt", "gte", "lt", "lte") and _to_number(value) is None:
            return None
        clean.append({"field": fname, "op": op, "value": value})
    show = [f for f in (plan.get("show") or []) if isinstance(f, str) and f in valid][:MAX_SHOW]   # display only
    if collection == "cards" and show and "board" not in show and "board" in valid:
        show.append("board")
    return {"collection": collection, "operation": operation, "filters": clean, "show": show}


def _regex_exact(text):
    return re.compile(f"^{re.escape(str(text).strip())}$", re.IGNORECASE)


def _ids_by_name(target: str, value: str) -> list:
    """Ids of records in `target` whose name matches every word of `value` (handles first/last name fields)."""
    key = (target, " ".join(value.lower().split()))
    cached = _name_ids.get(key)
    if cached is not None:
        return cached
    tokens = re.findall(r"\w+", value)[:4]
    if not tokens:
        return []
    fields = NAME_FIELDS + EXTRA_NAME_FIELDS
    cond = {"$and": [{"$or": [{f: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)} for f in fields]} for t in tokens]}
    ids = []
    for doc in mongo_service.get_db()[target].find(cond, {"_id": 1}).limit(50):
        ids.extend([doc["_id"], str(doc["_id"])])
    if not ids and len(tokens) == 1 and len(tokens[0]) >= 4:
        raw_token = tokens[0]
        norm_token = _norm(raw_token)
        clean_norm = norm_token[:-1] if norm_token.endswith("s") and not norm_token.endswith("ss") and len(norm_token) > 3 else norm_token
        projection = {"_id": 1, **{f: 1 for f in fields}}
        for doc in mongo_service.get_db()[target].find({}, projection).limit(100):
            matched = False
            for f in fields:
                val = doc.get(f)
                if isinstance(val, str):
                    nv = _norm(val)
                    if nv and (norm_token in nv or clean_norm in nv or (len(nv) >= 4 and (nv in norm_token or nv in clean_norm))):
                        ids.extend([doc["_id"], str(doc["_id"])])
                        matched = True
                        break
    _name_ids.set(key, ids)
    return ids


def _condition(info: dict, fname: str, op: str, value, notes: list):
    """-> (mongo_condition, readable_text)"""
    meta = info["fields"].get(fname, {})
    target = meta.get("ref")
    names = value if isinstance(value, list) else [value]

    if target and any(isinstance(v, str) and not HEX24.match(v) for v in names) and op in ("eq", "ne", "contains", "in"):
        ids = []
        for v in names:
            found = _ids_by_name(target, str(v))
            if not found:
                for alt_fname, alt_meta in info["fields"].items():
                    if alt_fname != fname and alt_meta.get("ref") and alt_meta.get("ref") != target:
                        alt_target = alt_meta.get("ref")
                        alt_found = _ids_by_name(alt_target, str(v))
                        if alt_found:
                            logger.info("Auto-corrected reference field %s (%s) -> %s (%s) for '%s'",
                                        fname, target, alt_fname, alt_target, v)
                            fname = alt_fname
                            target = alt_target
                            found = alt_found
                            notes.append(f"Interpreted '{v}' as {alt_fname} in '{alt_target}' instead of {fname}.")
                            break
            if not found:
                notes.append(f"No record in '{target}' matches the name '{v}'.")
            ids.extend(found)
        text = f"{fname} = {', '.join(map(str, names))} (resolved in '{target}')"
        if op == "ne":
            return {fname: {"$nin": ids}}, text.replace("=", "≠")
        return {fname: {"$in": ids}}, text            # empty ids -> matches nothing

    # Status wording varies across boards and task systems. Interpret common
    # user-facing buckets as sets of stored status values so "pending" also
    # finds todo/in-progress records and "completed" finds done records.
    if fname.lower().endswith("status") and isinstance(value, str):
        status = _norm(value)
        done = [re.compile(r"^(done|completed?|finished|closed)$", re.IGNORECASE)]
        terminal = done + [re.compile(r"^(cancelled?|canceled?)$", re.IGNORECASE)]
        if status in {"completed", "complete", "done", "finished"} and op in ("eq", "contains"):
            return {fname: {"$in": done}}, f"{fname} is completed"
        if status in {"pending", "open", "active", "notstarted"} and op in ("eq", "contains"):
            return {fname: {"$nin": terminal}}, f"{fname} is pending/open"
        if status in {"uncompleted", "incomplete", "unfinished", "notcomplete", "notcompleted", "incomplet"} and op in ("eq", "contains", "ne"):
            return {fname: {"$in": [re.compile(r"^(planning|in[- ]?progress)$", re.IGNORECASE)]}}, f"{fname} is uncompleted (planning or in-progress)"
        if status in {"done", "completed", "complete", "finished"} and op == "ne":
            return {fname: {"$not": done[0]}}, f"{fname} is not completed"
        if status in {"onhold", "hold"} and op in ("eq", "contains"):
            return {fname: re.compile(r"^on[- ]?hold$", re.IGNORECASE)}, f"{fname} is on-hold"
        if status in {"todo", "to-do"} and op in ("eq", "contains"):
            return {fname: re.compile(r"^to[- ]?do$", re.IGNORECASE)}, f"{fname} is to-do"
        if status in {"inprogress", "in-progress"} and op in ("eq", "contains"):
            return {fname: re.compile(r"^in[- ]?progress$", re.IGNORECASE)}, f"{fname} is in-progress"
        if status in {"review", "inreview"} and op in ("eq", "contains"):
            return {fname: re.compile(r"^(in[- ]?)?review$", re.IGNORECASE)}, f"{fname} is review"
        if any(w in status for w in ("qa", "testing")) and "review" in status and op in ("eq", "contains"):
            return {fname: re.compile(r"qa.*testing.*review", re.IGNORECASE)}, f"{fname} is qa-testing-&-review"

    numeric = meta.get("types") == {"num"}
    if op in ("gt", "gte", "lt", "lte"):
        return {fname: {f"${op}": _to_number(value)}}, f"{fname} {dict(gt='>', gte='>=', lt='<', lte='<=')[op]} {value}"
    if op == "in":
        conds = [_condition(info, fname, "eq", v, notes)[0] for v in value]
        return {"$or": conds}, f"{fname} in {value}"

    if isinstance(value, bool):
        return ({fname: value} if op == "eq" else {fname: {"$ne": value}}), f"{fname} {'=' if op == 'eq' else '≠'} {value}"
    if numeric and _to_number(value) is not None:
        number = _to_number(value)
        return ({fname: number} if op == "eq" else {fname: {"$ne": number}}), f"{fname} {'=' if op == 'eq' else '≠'} {number}"
    if op == "contains":
        return {fname: re.compile(re.escape(str(value).strip()), re.IGNORECASE)}, f"{fname} contains '{value}'"
    pattern = _regex_exact(value)
    if op == "eq":
        return {fname: pattern}, f"{fname} = {value}"
    return {fname: {"$not": pattern}}, f"{fname} ≠ {value}"


def match_collection(question: str, names: list):
    """No-LLM fallback: the one collection whose name or alias appears in the question (singular or plural)."""
    if re.search(r"\bleave\s+(?:requests?|applications?)\b", question, re.IGNORECASE):
        return "leaverequests" if "leaverequests" in names else None
    words = {_singular(w) for w in re.findall(r"[a-z0-9]+", question.lower())}
    aliases = get_aliases()
    alias_matches = {aliases[w] for w in words if w in aliases and aliases[w] in names}
    if len(alias_matches) == 1:
        return next(iter(alias_matches))
    matches = [n for n in names if _singular(_norm(n)) in words]
    return matches[0] if len(matches) == 1 else None


_UNCOMPLETED_PROJECT_QUESTION = re.compile(
    r"\b(?:un[\s-]?completed?|incomplete|incomplet|not\s+(?:yet\s+)?(?:completed?|done|finished)|unfinished|not\s+complete|notcomplete)\b"
    r".*\b(?:projects?|boards?)\b|\b(?:projects?|boards?)\b"
    r".*\b(?:un[\s-]?completed?|incomplete|incomplet|not\s+(?:yet\s+)?(?:completed?|done|finished)|unfinished|not\s+complete|notcomplete)\b",
    re.IGNORECASE,
)


def _uncompleted_projects_answer(question: str, workspace_id=None):
    """List active projects in Planning or In-Progress status."""
    if not _UNCOMPLETED_PROJECT_QUESTION.search(question or ""):
        return None

    db = mongo_service.get_db()
    flt = _and_merge(
        _workspace_filter("boards", workspace_id, db),
        {"status": re.compile(r"^(planning|in[- ]?progress)$", re.IGNORECASE)},
        {"isArchived": {"$ne": True}},
        {"name": {"$nin": [re.compile(r"^leaves?$", re.IGNORECASE)]}},
    )
    collection = db["boards"]
    if project_type is None:
        values = sorted(
            {str(value).strip() for value in collection.distinct("projectType", flt)
             if value is not None and str(value).strip()},
            key=str.casefold,
        )
        answer = ("Project types: " + ", ".join(values) + ".") if values else "No project types were found."
        return ExactResult(
            collection="boards", total=len(values), records=[],
            conditions="distinct projectType values in the requested workspace",
            complete=True, source="project-type-rule", operation="list",
            direct_answer=answer, timings={"db": 0},
        )

    total = _count(collection, flt)
    docs = list(collection.find(flt).sort("name", 1).limit(settings.STRUCT_MAX_RECORDS))
    records = []
    for doc in docs:
        name = doc.get("name") or doc.get("title") or str(doc.get("_id", "Project"))
        status = str(doc.get("status") or "").strip()
        description = doc.get("description") or doc.get("scope") or ""
        records.append({
            "document_id": str(doc.get("_id", "")),
            "name": str(name),
            "text": f"Name: {name}; Status: {status}; Description: {description}",
        })

    return ExactResult(
        collection="boards", total=total, records=records,
        conditions="active projects with status Planning or In-Progress",
        complete=total <= len(records), source="status-rule", operation="list",
        timings={"db": 0},
    )


_PROJECT_TYPE_INTENT = re.compile(
    r"\b(?:project\s+types?|types?\s+of\s+projects?|in[- ]?house|hired\s+client)\b",
    re.IGNORECASE,
)


def is_project_type_question(question: str) -> bool:
    """Whether a project query asks for a project type or projects of a known type."""
    return bool(_PROJECT_TYPE_INTENT.search(question or "") and _PROJECT_INTENT.search(question or ""))


def _project_type_answer(question: str, workspace_id=None, restrict_unscoped=False,
                         is_admin=False):
    """Apply project type and explicit status constraints directly to boards."""
    if not is_project_type_question(question):
        return None
    if restrict_unscoped and not is_admin and not workspace_id:
        return None

    db = mongo_service.get_db()
    project_type = None
    if re.search(r"\bin[- ]?house\b", question, re.IGNORECASE):
        project_type = re.compile(r"^in[- ]?house$", re.IGNORECASE)
    elif re.search(r"\bhired\s+client\b", question, re.IGNORECASE):
        project_type = re.compile(r"^hired\s+client$", re.IGNORECASE)

    flt = _and_merge(
        _workspace_filter("boards", workspace_id, db),
        {"name": {"$nin": [re.compile(r"^leaves?$", re.IGNORECASE)]}},
        {"isArchived": {"$ne": True}},
    )
    conditions = []
    if project_type is not None:
        flt = _and_merge(flt, {"projectType": project_type})
        conditions.append(f"projectType matches {project_type.pattern}")
    if re.search(r"\b(?:completed?|done|finished|closed)\b", question, re.IGNORECASE):
        status = re.compile(r"^(done|completed?|finished|closed)$", re.IGNORECASE)
        flt = _and_merge(flt, {"status": status})
        conditions.append("status is completed")
    elif re.search(r"\b(?:pending|open|active|in[- ]progress|planning)\b", question, re.IGNORECASE):
        status = re.compile(r"^(planning|in[- ]?progress|to[- ]?do|open|pending)$", re.IGNORECASE)
        flt = _and_merge(flt, {"status": status})
        conditions.append("status is planning or in-progress")

    collection = db["boards"]
    total = _count(collection, flt)
    docs = list(collection.find(flt).sort("name", 1).limit(settings.STRUCT_MAX_RECORDS))
    records = []
    for doc in docs:
        name = doc.get("name") or doc.get("title") or "Project"
        status_value = str(doc.get("status") or "").strip()
        type_value = str(doc.get("projectType") or "").strip()
        description = strip_html(doc.get("description") or "")
        records.append({
            "document_id": str(doc.get("_id", "")),
            "name": str(name),
            "text": f"Name: {name}; Type: {type_value}; Status: {status_value}; Description: {description}",
        })

    qualifier = "in-house " if project_type is not None else ""
    if re.search(r"\b(?:completed?|done|finished|closed)\b", question, re.IGNORECASE):
        qualifier += "completed "
    direct_answer = ""
    if total == 0:
        direct_answer = f"No {qualifier}projects were found matching the requested criteria."
    return ExactResult(
        collection="boards", total=total, records=records,
        conditions="; ".join(conditions) or "project type query",
        complete=total <= len(records), source="project-type-rule", operation="list",
        direct_answer=direct_answer,
        timings={"db": 0},
    )


def mentions_collection(question: str) -> bool:
    """True when the question uses a collection name or alias (e.g. 'projects', 'boards', 'tasks')."""
    if re.search(r"\bleave\s+(?:requests?|applications?)\b", question, re.IGNORECASE):
        return True
    try:
        names = known_collections()
    except ChatbotError:
        return False
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    singular_words = {_singular(w) for w in words}
    aliases = get_aliases()
    if any(w in aliases for w in (words | singular_words)):
        return True
    return any(_norm(n) in words or _singular(_norm(n)) in singular_words for n in names)


_USER_ATTR_EMAIL = re.compile(r"\b(?:emails?|email\s+address(?:es)?|mail|emailid|email[- ]id)\b", re.IGNORECASE)
_USER_ATTR_WORKSPACE = re.compile(r"\b(?:workspace[s]?|company|companies|organization[s]?|team[s]?)\b", re.IGNORECASE)
_USER_ATTR_ROLE = re.compile(r"\b(?:roles?|job\s*titles?|designations?|positions?)\b", re.IGNORECASE)
_USER_ATTR_PROFILE = re.compile(r"\b(?:who\s+is|tell\s+me\s+about|profile\s+of|details\s+of|information\s+about|info\s+of)\b", re.IGNORECASE)
_PRONOUN_PATTERN = re.compile(r"\b(?:he|him|his|she|her|they|them|their|this\s+person|this\s+employee|that\s+person|that\s+employee)\b", re.IGNORECASE)


def is_person_attribute_question(question: str) -> bool:
    """Check if the question asks for a specific person's email, workspace, role, or profile."""
    q = question or ""
    if re.search(r"\b(?:my|mine|myself|who\s+am\s+i|am\s+i)\b", q, re.IGNORECASE):
        return False
    if _USER_ATTR_EMAIL.search(q):
        return bool(_PRONOUN_PATTERN.search(q)
                    or re.search(r"['’]s\s+email|\bemail\s+(?:of|for)\b|\bwhat\s+is\s+email\b", q, re.IGNORECASE)
                    or re.search(r"\b[A-Za-z][A-Za-z .'-]*?\s+email\b", q, re.IGNORECASE))
    if _USER_ATTR_WORKSPACE.search(q):
        return bool(_PRONOUN_PATTERN.search(q)
                    or re.search(r"['’]s\s+(?:workspace|company)|\b(?:workspace|company)\s+(?:of|for)\b", q, re.IGNORECASE)
                    or re.search(r"\bwhat\s+is\s+([A-Za-z][A-Za-z .'-]*?)['’]s\s+workspace\b", q, re.IGNORECASE))
    if _USER_ATTR_ROLE.search(q):
        return bool(_PRONOUN_PATTERN.search(q)
                    or re.search(r"['’]s\s+role|\brole\s+(?:of|for)\b", q, re.IGNORECASE))
    if _USER_ATTR_PROFILE.search(q):
        if re.search(r"\b(?:working\s+on|assigned\s+to|responsible\s+for|allocated\s+to|involved\s+in|project|board|task|card)\b", q, re.IGNORECASE):
            return False
        return True
    return False


def _ambiguous_user_result(matches: list, person: str, started) -> ExactResult:
    """Several users share the queried name: ask which one instead of answering for the first."""
    lines, records = [], []
    for user in matches:
        c_name = user.get("name") or user.get("displayName") or "Unknown"
        details = [str(user[field]) for field in ("role", "email") if user.get(field)]
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- {c_name}{suffix}")
        records.append({
            "document_id": str(user.get("_id", "")),
            "name": c_name,
            "text": f"Name: {c_name}; Email: {user.get('email') or 'N/A'}; Role: {user.get('role') or 'N/A'}",
        })
    direct_answer = (f"I found {len(matches)} people matching '{person}'. "
                     "Please specify which one you mean:\n" + "\n".join(lines))
    return ExactResult(
        collection="users", total=len(matches), records=records,
        conditions=f"user name matches '{person}'", complete=True, source="planner",
        operation="lookup", direct_answer=direct_answer,
        timings={"db": round((time.perf_counter() - started) * 1000)},
    )


def _user_attribute_answer(question: str, history: list = None, workspace_id=None):
    """Answer questions about a user's email, workspace, role, or profile with exact DB records."""
    if not is_person_attribute_question(question):
        return None

    attr = None
    if _USER_ATTR_EMAIL.search(question):
        attr = "email"
    elif _USER_ATTR_WORKSPACE.search(question):
        attr = "workspace"
    elif _USER_ATTR_ROLE.search(question):
        attr = "role"
    elif _USER_ATTR_PROFILE.search(question):
        attr = "profile"

    if not attr:
        return None

    has_pronoun = bool(_PRONOUN_PATTERN.search(question))
    person = None
    if not has_pronoun:
        m = re.search(r"\b(?:who\s+is|tell\s+me\s+about|profile\s+of|details\s+of)\s+(?:the\s+)?([A-Za-z][A-Za-z .\'’-]*?)(?=[?.!,]|$)", question, re.IGNORECASE)
        if m:
            person = m.group(1).strip()
        if not person:
            m = re.search(r"\b([A-Za-z][A-Za-z .\'’-]*?)[\'’]s\s+(?:email|workspace|company|role|profile|job)", question, re.IGNORECASE)
            if m:
                person = m.group(1).strip()
        if not person:
            m = re.search(r"\b(?:email|workspace|company|role|profile|details)\s+(?:of|for)\s+([A-Za-z][A-Za-z .\'’-]*?)(?=[?.!,]|$)", question, re.IGNORECASE)
            if m:
                person = m.group(1).strip()
        if not person:
            m = re.search(r"\b([A-Za-z][A-Za-z .\'’-]{2,30}?)\s+(?:email|workspace|company|role)\b", question, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                words = candidate.split()
                stops = {"what", "which", "who", "tell", "show", "give", "is", "are", "was", "the", "a", "an"}
                while words and words[0].lower() in stops:
                    words.pop(0)
                if words:
                    person = " ".join(words)

    if person:
        person = re.sub(r"[\'’]s?$", "", person).strip(" .'-")
        if _PRONOUN_PATTERN.fullmatch(person):
            has_pronoun = True
            person = None

    if (has_pronoun or not person) and history:
        for turn in reversed(history):
            ans = turn.get("answer", "")
            m_ans = re.search(r"^(?:Yes|No),\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3}?)\s+(?:has|have|works|worked|is|was|did|logged)\b", ans)
            if not m_ans:
                m_ans = re.search(r"^([A-Za-z]+(?:\s+[A-Za-z]+){0,3}?)[\'’]s\s+(?:email|workspace|company|role|tasks?|logged)", ans)
            if not m_ans:
                m_ans = re.search(r"^([A-Za-z]+(?:\s+[A-Za-z]+){0,3}?)\s+is\s+a\b", ans)
            if m_ans:
                person = m_ans.group(1).strip()
                break
            q = turn.get("question", "")
            m_q = re.search(r"\b(?:email|workspace|company|role|tasks?)\s+(?:of|for)\s+([A-Za-z][A-Za-z .\'’-]*?)(?=[?.!,]|$)", q, re.IGNORECASE)
            if not m_q:
                m_q = re.search(r"\b([A-Za-z][A-Za-z .\'’-]*?)[\'’]s", q)
            if not m_q:
                m_q = re.search(r"\b(?:did|does|do|has|have|is|was)\s+([A-Za-z][A-Za-z .\'’-]*?)\s+(?:work|worked|do|logged|assigned)\b", q, re.IGNORECASE)
            if not m_q:
                m_q = re.search(r"\b(?:who\s+is|tell\s+me\s+about)\s+([A-Za-z][A-Za-z .\'’-]*?)(?=[?.!,]|$)", q, re.IGNORECASE)
            if m_q:
                cand = re.sub(r"[\'’]s?$", "", m_q.group(1)).strip(" .'-")
                if not _PRONOUN_PATTERN.fullmatch(cand):
                    person = cand
                    break

    if not person:
        return None

    from . import query_correction
    corrected_person = query_correction.correct_query(person)
    tokens = re.findall(r"[A-Za-z0-9]+", corrected_person)
    if not tokens:
        tokens = re.findall(r"[A-Za-z0-9]+", person)
    if not tokens:
        return None

    db = mongo_service.get_db()
    started = time.perf_counter()
    name_fields = ("name", "fullName", "firstName", "lastName", "username", "displayName", "email")
    limit = min(settings.STRUCT_MAX_RECORDS, 10)
    ws_flt = _workspace_filter("users", workspace_id, db)

    def _find_users(patterns):
        condition = {"$and": [{"$or": [{fld: re.compile(re.escape(t), re.IGNORECASE)} for fld in name_fields]}
                              for t in patterns]}
        return list(db["users"].find(_and_merge(ws_flt, condition)).limit(limit))

    matches = _find_users(tokens)                      # most precise: every name token must match
    if not matches and len(tokens) > 1:
        matches = _find_users([tokens[0]])            # fall back to the first token
    if not matches and len(tokens) > 1:
        matches = _find_users([tokens[-1]])           # then the last token
    user = matches[0] if matches else None

    if not user:
        direct_answer = f"I couldn't find a user named '{person}'."
        return ExactResult(
            collection="users", total=0, records=[], conditions=f"user = {person}",
            complete=True, source="planner", operation="lookup",
            direct_answer=direct_answer, timings={"db": round((time.perf_counter() - started) * 1000)}
        )

    if len(matches) > 1:
        return _ambiguous_user_result(matches, person, started)

    canonical_name = user.get("name") or user.get("displayName") or f"{user.get('firstName', '')} {user.get('lastName', '')}".strip() or person
    email = user.get("email")
    role = user.get("role") or user.get("title") or user.get("jobTitle") or "employee"
    ws_name = None
    if user.get("lastActiveWorkspace"):
        ws_id = user["lastActiveWorkspace"]
        try:
            ws = db["workspaces"].find_one({"_id": ws_id})
            if not ws:
                ws = db["workspaces"].find_one({"_id": ObjectId(str(ws_id))})
            if ws:
                ws_name = ws.get("name") or ws.get("title")
        except Exception:
            pass

    if attr == "email":
        direct_answer = f"{canonical_name}'s email address is {email}." if email else f"{canonical_name} does not have an email address recorded."
    elif attr == "workspace":
        direct_answer = f"{canonical_name}'s workspace is {ws_name}." if ws_name else f"{canonical_name} does not have a workspace recorded."
    elif attr == "role":
        direct_answer = f"{canonical_name}'s role is {role}."
    elif attr == "profile":
        parts = [f"{canonical_name} is a {role}"]
        if ws_name:
            parts.append(f"at {ws_name}")
        ans_str = " ".join(parts) + "."
        if email:
            ans_str += f" Email: {email}."
        direct_answer = ans_str

    record = {
        "document_id": str(user.get("_id", "")),
        "name": canonical_name,
        "text": f"Name: {canonical_name}; Email: {email or 'N/A'}; Role: {role}; Workspace: {ws_name or 'N/A'}"
    }
    return ExactResult(
        collection="users", total=1, records=[record], conditions=f"user = {canonical_name}",
        complete=True, source="planner", operation="lookup",
        direct_answer=direct_answer, timings={"db": round((time.perf_counter() - started) * 1000)}
    )


_PERSON_LOOKUP_PATTERN = re.compile(
    r"^(?:who\s+is|tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\s+"
    r"(?:the\s+)?([A-Za-z][A-Za-z.'’\s-]*?)[?.!]*$",
    re.IGNORECASE,
)

_SHORT_ROLE_LOOKUP_PATTERN = re.compile(
    r"^\s*(?:the\s+)?(admin(?:istrator)?|manager|owner)\s+of\s+(.+?)[?.!]*\s*$",
    re.IGNORECASE,
)
_WORKSPACE_ADMIN_PATTERN = re.compile(
    r"^\s*(?:who\s+(?:is|are)\s+)?(?:the\s+)?(admin(?:istrator)?|manager|owner)\s+(?:of|at|for|in)\s+(.+?)[?.!]*\s*$",
    re.IGNORECASE,
)


def normalize_role_lookup_question(question: str) -> str:
    """Turn short role lookups (for example, 'admin of Acme?') into explicit person questions."""
    match = _SHORT_ROLE_LOOKUP_PATTERN.fullmatch(question or "")
    if not match:
        return question
    role, organization = match.groups()
    return f"who is {role} of {organization.strip()}?"


def is_workspace_role_question(question: str) -> bool:
    return bool(_WORKSPACE_ADMIN_PATTERN.fullmatch(question or ""))


def _workspace_admin_answer(question: str, workspace_id=None):
    """Resolve a workspace manager/admin from users.lastActiveWorkspace, or its owner field."""
    match = _WORKSPACE_ADMIN_PATTERN.fullmatch(question or "")
    if not match:
        return None

    role, workspace_name = match.groups()
    workspace_name = workspace_name.strip(" .'-")
    if not workspace_name:
        return None

    db = mongo_service.get_db()
    tokens = re.findall(r"[A-Za-z0-9]+", workspace_name)
    name_filter = {"$and": [
        {"name": {"$regex": re.escape(token), "$options": "i"}} for token in tokens
    ]}
    filters = [name_filter, {"isActive": {"$ne": False}}, {"status": {"$ne": "inactive"}}]
    if workspace_id:
        filters.append({"_id": {"$in": _id_variants([workspace_id])}})
    workspaces = list(db["workspaces"].find(_and_merge(*filters)).limit(10))
    if not workspaces:
        return ExactResult(
            collection="workspaces", total=0, records=[], conditions=f"workspace = {workspace_name}",
            complete=True, source="workspace-owner", operation="lookup",
            direct_answer=f"I couldn't find an active workspace matching '{workspace_name}'.",
        )

    if len(workspaces) > 1:
        names = [str(ws.get("name") or ws.get("title") or ws.get("_id")) for ws in workspaces]
        return ExactResult(
            collection="workspaces", total=len(workspaces), records=[],
            conditions=f"workspace name contains '{workspace_name}'", complete=True,
            source="workspace-owner", operation="lookup",
            direct_answer=(f"I found multiple workspaces matching '{workspace_name}': "
                           + ", ".join(names) + ". Please specify the workspace name."),
        )

    workspace = workspaces[0]
    ws_name = workspace.get("name") or workspace.get("title") or workspace_name
    role_key = "admin" if role.lower().startswith("admin") else role.lower()
    users = list(db["users"].find({
        "lastActiveWorkspace": {"$in": _id_variants([str(workspace["_id"])] )},
        "role": {"$regex": re.escape(role_key), "$options": "i"},
    }))
    if not users and role_key == "owner" and workspace.get("owner") is not None:
        owner = db["users"].find_one({"_id": {"$in": _id_variants([workspace["owner"]])}})
        users = [owner] if owner else []

    records = []
    names = []
    for user in users:
        name = (user.get("name") or user.get("displayName") or user.get("fullName")
                or user.get("full_name") or user.get("email") or str(user.get("_id")))
        names.append(name)
        records.append({"document_id": str(user.get("_id", "")), "name": name,
                        "text": f"Name: {name}; Role: {user.get('role', '')}; Workspace: {ws_name}"})
    if names:
        answer = f"{', '.join(names)} {('is' if len(names) == 1 else 'are')} the {role.lower()}{'' if len(names) == 1 else 's'} of {ws_name}."
    else:
        answer = f"I found the {ws_name} workspace, but couldn't find a user with the {role.lower()} role linked through lastActiveWorkspace."

    return ExactResult(
        collection="workspaces + users", total=len(records), records=records,
        conditions=f"workspace owner for '{ws_name}'", complete=True,
        source="workspace-owner", operation="lookup", direct_answer=answer,
    )


_WORKSPACE_EMPLOYEE_COUNT_PATTERN = re.compile(
    r"\b(?:how\s+many|count(?:\s+of)?|number\s+of|total\s+number\s+of)\s+"
    r"(?:the\s+)?(?:employees|staff|users|members)\s+(?:are\s+)?"
    r"(?:in|at|within|of)\s+(?:the\s+)?(.+?)\s*[?.!]*$",
    re.IGNORECASE,
)
_WORKSPACE_EMPLOYEE_LIST_PATTERN = re.compile(
    r"\b(?:list|show|give\s+me)\s+(?:the\s+)?(?:all\s+)?(?:employees|staff|users|members)\s+"
    r"(?:in|at|within|of)\s+(?:the\s+)?(.+?)\s*[?.!]*$",
    re.IGNORECASE,
)


def _workspace_employee_list_answer(question: str, is_admin: bool):
    """List every user linked directly or through membership to a named workspace."""
    if not is_admin:
        return None
    match = _WORKSPACE_EMPLOYEE_LIST_PATTERN.search((question or "").strip())
    if not match:
        return None
    workspace_name = match.group(1).strip(" .,'\"!?-")
    tokens = re.findall(r"[A-Za-z0-9]+", workspace_name)
    if not tokens:
        return None
    db = mongo_service.get_db()
    name_filter = {"$and": [{"$or": [{field: {"$regex": re.escape(token), "$options": "i"}}
                                          for field in ("name", "title")]} for token in tokens]}
    workspaces = list(db["workspaces"].find(name_filter).limit(10))
    if len(workspaces) != 1:
        return None
    workspace = workspaces[0]
    ws_id = str(workspace["_id"])
    ids = _user_ids_in_workspace(db, _id_variants([ws_id]))
    docs = list(db["users"].find({"_id": {"$in": ids}}).sort("name", 1)) if ids else []
    names = []
    for doc in docs:
        converted = document_converter.convert_document("users", doc)
        name = converted.display_name or doc.get("name") or doc.get("displayName")
        if name and name not in names:
            names.append(name)
    resolved_name = workspace.get("name") or workspace.get("title") or workspace_name
    answer = (f"There are {len(names)} employees in {resolved_name}:\n" +
              "\n".join(f"- {name}" for name in names)) if names else f"No employees are recorded for {resolved_name}."
    return ExactResult(collection="users", total=len(names), records=[],
                       conditions=f"users linked to workspace '{resolved_name}'", complete=True,
                       source="workspace-employee-list", operation="list", direct_answer=answer)


def _workspace_employee_count_answer(question: str, is_admin: bool):
    """Count users linked directly or through a membership record to a named workspace."""
    if not is_admin:
        return None
    match = _WORKSPACE_EMPLOYEE_COUNT_PATTERN.search((question or "").strip())
    if not match:
        return None

    workspace_name = match.group(1).strip(" .,'\"!?-")
    tokens = re.findall(r"[A-Za-z0-9]+", workspace_name)
    if not tokens:
        return None

    db = mongo_service.get_db()
    name_filter = {"$and": [
        {"$or": [
            {field: {"$regex": re.escape(token), "$options": "i"}}
            for field in ("name", "title")
        ]}
        for token in tokens
    ]}
    workspaces = list(db["workspaces"].find(name_filter).limit(10))
    if not workspaces:
        return ExactResult(
            collection="users", total=0, records=[], conditions=f"workspace = {workspace_name}",
            complete=True, source="workspace-employee-count", operation="count",
            direct_answer=f"I couldn't find a workspace matching '{workspace_name}'.",
        )
    if len(workspaces) > 1:
        names = [str(ws.get("name") or ws.get("title") or ws.get("_id")) for ws in workspaces]
        return ExactResult(
            collection="users", total=0, records=[], conditions=f"workspace = {workspace_name}",
            complete=True, source="workspace-employee-count", operation="count",
            direct_answer=(f"I found multiple workspaces matching '{workspace_name}': "
                           + ", ".join(names) + ". Please specify the workspace name."),
        )

    workspace = workspaces[0]
    workspace_id = str(workspace["_id"])
    user_ids = _user_ids_in_workspace(db, _id_variants([workspace_id]))
    total = _count(db["users"], {"_id": {"$in": user_ids}}) if user_ids else 0
    resolved_name = workspace.get("name") or workspace.get("title") or workspace_name
    return ExactResult(
        collection="users", total=total, records=[],
        conditions=f"users linked to workspace '{resolved_name}'", complete=True,
        source="workspace-employee-count", operation="count",
        direct_answer=f"There are {total} employees in the {resolved_name} workspace.",
    )


def is_person_lookup_question(question: str) -> bool:
    """Whether the question asks for a person's profile by name."""
    # The broad "tell me about <name>" form can capture project requests such as
    # "tell me about in house projects". Let those go through the project query path.
    if re.search(r"\b(?:projects?|boards?)\b", question or "", re.IGNORECASE):
        return False
    return bool(_PERSON_LOOKUP_PATTERN.fullmatch(question.strip()))


def _person_lookup(question: str, workspace_id=None):
    """Resolve a simple person question directly against user profile name fields."""
    if re.search(r"\b(?:projects?|boards?)\b", question or "", re.IGNORECASE):
        return None
    match = _PERSON_LOOKUP_PATTERN.fullmatch(question.strip())
    if not match:
        return None
    raw_name = match.group(1).strip()
    name = re.sub(r"['’]s?$", "", raw_name).strip(" .'-")
    tokens = re.findall(r"[A-Za-z0-9]+", name)
    if not tokens:
        return None

    fields = NAME_FIELDS + EXTRA_NAME_FIELDS
    condition = {"$and": [
        {"$or": [{field: re.compile(re.escape(token), re.IGNORECASE)} for field in fields]}
        for token in tokens
    ]}
    db = mongo_service.get_db()
    started = time.perf_counter()
    users = db["users"]
    flt = _and_merge(_workspace_filter("users", workspace_id, db), condition)
    total = _count(users, flt)
    if not total:
        return None

    docs = list(users.find(flt).limit(min(settings.STRUCT_MAX_RECORDS, 10)))
    resolver = _exec_resolver(build_catalog())
    resolver.prefetch("users", docs, infer=False)
    records = []
    for doc in docs:
        converted = document_converter.convert_document("users", doc, resolver)
        body = converted.lines[2:] if len(converted.lines) > 2 else converted.lines
        records.append({"document_id": converted.primary_id,
                        "name": converted.display_name,
                        "text": "; ".join(body)[:MAX_RECORD_CHARS]})

    direct_ans = ""
    if total > 1:
        # The name matches several people: never answer for just one of them.
        lines = []
        for doc in docs:
            c_name = doc.get("name") or doc.get("displayName") or name
            details = [str(doc[field]) for field in ("role", "email") if doc.get(field)]
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(f"- {c_name}{suffix}")
        direct_ans = (f"I found {total} people matching '{name}'. "
                      "Please specify which one you mean:\n" + "\n".join(lines))
    elif docs:
        first = docs[0]
        c_name = first.get("name") or first.get("displayName") or name
        role = first.get("role") or first.get("jobTitle") or "employee"
        ws_name = None
        if first.get("lastActiveWorkspace"):
            try:
                ws = db["workspaces"].find_one({"_id": first["lastActiveWorkspace"]})
                if not ws:
                    ws = db["workspaces"].find_one({"_id": ObjectId(str(first["lastActiveWorkspace"]))})
                if ws:
                    ws_name = ws.get("name") or ws.get("title")
            except Exception:
                pass
        parts = [f"{c_name} is a {role}"]
        if ws_name:
            parts.append(f"at {ws_name}")
        ans_s = " ".join(parts) + "."
        if first.get("email"):
            ans_s += f" Email: {first['email']}."
        direct_ans = ans_s

    result = ExactResult(
        collection="users", total=total, records=records,
        conditions=f"profile name matches '{name}'", complete=total <= len(records),
        source="planner", operation="list",
        direct_answer=direct_ans,
        timings={"db": round((time.perf_counter() - started) * 1000)},
    )
    result.context = _context(result)
    return result



# -------------------------------------------------------------- execution
def _count(collection, flt) -> int:
    try:
        return collection.count_documents(flt, maxTimeMS=settings.STRUCT_TIMEOUT_MS)
    except (TypeError, NotImplementedError):      # drivers/mocks without maxTimeMS support
        return collection.count_documents(flt)


def _labels_for(show: list) -> list:
    labels = []
    for path in show:
        parts = path.split(".")
        parts[-1] = strip_id_suffix(parts[-1]) or parts[-1]
        labels.append(" ".join(humanize_key(p) for p in parts))
    return labels


def _select_lines(lines: list, labels: list) -> list:
    kept = [l for l in lines if any(l.startswith(lab + ":") or l.startswith(lab + " ID") for lab in labels)]
    return kept or lines


def _without_logged_time(doc: dict) -> dict:
    """Drop logged-time fields so task/project details do not leak time entries."""
    dropped = {"loggedtime", "logged_time", "loggedtimes", "logged_times",
               "timelog", "time_log", "timelogs", "time_logs"}
    return {key: value for key, value in doc.items() if str(key).lower() not in dropped}


def _execute_tool_action(plan: dict, catalog: dict, source: str, include_logged_time: bool = False,
                         current_user_id: str = None, is_admin: bool = False, workspace_id=None):
    """Execute a predefined multi-collection aggregation pipeline tool."""
    action = plan.get("action")
    db = mongo_service.get_db()

    if action == "logged_time":
        person = plan.get("person")
        project = plan.get("project")
        date_str = plan.get("date")
        extremum = plan.get("extremum")
        parts = []
        if person:
            parts.append(f"{person}'s")
        if extremum:
            parts.append(extremum)
        parts.append("logged time")
        if project:
            parts.append(f"in {project}")
        if date_str:
            parts.append(f"on {date_str}")
        synthetic_q = " ".join(parts)
        res = _logged_time_answer(synthetic_q, current_user_id=current_user_id, is_admin=is_admin,
                                  workspace_id=workspace_id)
        if res is not None:
            return res

    elif action == "employee_work":
        person = plan.get("person")
        project = plan.get("project")
        status = plan.get("status")
        verify = plan.get("verify", False)
        parts = []
        if verify:
            parts.append("does")
        parts.append(person or "employee")
        parts.append("work on")
        if project:
            parts.append(f"{project} project")
        if status:
            parts.append(status)
        synthetic_q = " ".join(parts)
        res = _employee_work_status(synthetic_q, workspace_id=workspace_id)
        if res is not None:
            return res

    elif action == "project_team":
        project = plan.get("project")
        board_ids = _ids_by_name("boards", project) if project else []
        if board_ids:
            cursor = db["boards"].find(_and_merge(
                _workspace_filter("boards", workspace_id, db), {"_id": {"$in": _id_variants(board_ids)}}))
            docs = list(cursor)
            if docs:
                records = [{"document_id": str(d["_id"]), "name": d.get("name") or "Project", "text": f"Name: {d.get('name')}"} for d in docs]
                res = ExactResult(
                    collection="boards", total=len(docs), records=records, conditions=f"project = {project}",
                    complete=True, source="planner", operation="list"
                )
                _project_team_details(res)
                return res

    elif action == "leave_records":
        person = plan.get("person")
        leave_type = plan.get("leave_type")
        parts = ["leave"]
        if leave_type:
            parts.append(leave_type)
        if person:
            parts.append(f"for {person}")
        synthetic_q = " ".join(parts)
        res = _leave_time_answer(synthetic_q, current_user_id=current_user_id, is_admin=is_admin,
                                 workspace_id=workspace_id)
        if res is not None:
            return res
        res = _leave_types_answer(synthetic_q, workspace_id=workspace_id)
        if res is not None:
            return res

    return None


def execute(plan: dict, catalog: dict, source: str, include_logged_time: bool = False,
            current_user_id: str = None, is_admin: bool = False, workspace_id=None):
    started = time.perf_counter()
    action = plan.get("action")
    if action:
        res = _execute_tool_action(plan, catalog, source, include_logged_time,
                                   current_user_id=current_user_id, is_admin=is_admin,
                                   workspace_id=workspace_id)
        if res is not None:
            res.timings = {"db": round((time.perf_counter() - started) * 1000)}
            return res
        return None

    name = plan.get("collection")
    if not name or mongo_service.is_blocked_collection(name) or name not in catalog:
        return None
    info = catalog[name]
    notes, conds, texts = [], [], []
    for f in plan["filters"]:
        cond, text = _condition(info, f["field"], f["op"], f["value"], notes)
        conds.append(cond)
        texts.append(text)
    if workspace_id:
        notes.append("Workspace scope applied: only records of the requested workspace are read.")
    flt = _and_merge(_workspace_filter(name, workspace_id), {"$and": conds} if conds else {})
    if name == "boards" and not any(f.get("field") == "name" for f in plan.get("filters", [])):
        flt = _and_merge(flt, {"name": {"$nin": [re.compile(r"^leaves?$", re.IGNORECASE)]}})
    collection = mongo_service.get_db()[name]
    total = _count(collection, flt)
    if total == 0 and conds:
        # If querying 'boards' by 'members', fall back to finding boards via assigned 'cards'
        if name == "boards" and any(f.get("field") == "members" for f in plan.get("filters", [])):
            member_conds = [c for c in conds if "members" in c]
            user_ids = []
            for mc in member_conds:
                if "$in" in mc.get("members", {}):
                    user_ids.extend(mc["members"]["$in"])
            if user_ids:
                card_filter = {"assignees": {"$in": user_ids}}
                status_f = next((f for f in plan.get("filters", []) if f.get("field") == "status"), None)
                if status_f:
                    val = str(status_f.get("value", "")).lower()
                    if val in ("completed", "done"):
                        card_filter["status"] = {"$in": ["done", "completed"]}
                    elif val in ("pending", "in-progress", "open"):
                        card_filter["status"] = {"$nin": ["done", "completed"]}
                card_filter = _and_merge(_workspace_filter("cards", workspace_id), card_filter)
                cards = list(mongo_service.get_db()["cards"].find(card_filter, {"board": 1}))
                board_ids = list({c["board"] for c in cards if c.get("board")})
                if board_ids:
                    other_conds = [c for c in conds if "members" not in c and "status" not in c]
                    new_conds = [{"_id": {"$in": board_ids}}] + other_conds
                    flt = _and_merge(_workspace_filter(name, workspace_id),
                                     {"$and": new_conds} if len(new_conds) > 1 else new_conds[0])
                    total = _count(collection, flt)
                    notes.append("Resolved projects via tasks assigned to team members in 'cards'.")

        # If querying 'boards' for task-level workflow status (e.g. 'review', 'to-do', 'qa-testing-&-review')
        if total == 0 and name == "boards" and any(f.get("field") == "status" for f in plan.get("filters", [])):
            status_f = next((f for f in plan.get("filters", []) if f.get("field") == "status"), None)
            if status_f:
                s_val = str(status_f.get("value", "")).strip().lower()
                card_filter = _and_merge(_workspace_filter("cards", workspace_id),
                                         {"status": re.compile(rf"{re.escape(s_val)}", re.IGNORECASE)})
                cards = list(mongo_service.get_db()["cards"].find(card_filter, {"board": 1}))
                board_ids = list({c["board"] for c in cards if c.get("board")})
                if board_ids:
                    other_conds = [c for c in conds if "status" not in c]
                    new_conds = [{"_id": {"$in": board_ids}}] + other_conds
                    flt = _and_merge(_workspace_filter(name, workspace_id),
                                     {"$and": new_conds} if len(new_conds) > 1 else new_conds[0])
                    total = _count(collection, flt)
                    notes.append(f"Resolved {total} projects containing tasks with status '{s_val}'.")
        if total == 0:
            if plan.get("operation") == "count":
                cond_text = "; ".join(texts) if texts else "no conditions"
                return ExactResult(
                    collection=name, total=0, records=[],
                    conditions=cond_text, complete=True, source=source, operation="count",
                    direct_answer=f"There are 0 {name} matching the criteria ({cond_text}).",
                    context=f"EXACT QUERY RESULT from the '{name}' collection: 0 matching records found in the database. Conditions: {cond_text}",
                )
            return None                       # probably a mis-read question: let similarity search try instead
    wanted = 10 if plan["operation"] == "count" else settings.STRUCT_MAX_RECORDS
    cursor = collection.find(flt)
    if "createdAt" in info.get("fields", {}):
        cursor = cursor.sort("createdAt", -1)
    else:
        cursor = cursor.sort("_id", 1)
    cursor = cursor.limit(wanted)
    try:
        cursor = cursor.max_time_ms(settings.STRUCT_TIMEOUT_MS)
    except Exception:
        pass
    docs = list(cursor)
    resolver = _exec_resolver(catalog)
    resolver.prefetch(name, docs, infer=False)      # names only: targets are already known
    labels = _labels_for(plan.get("show") or [])
    records = []
    for doc in docs:
        if not include_logged_time:
            doc = _without_logged_time(doc)
        conv = document_converter.convert_document(name, doc, resolver)
        body = conv.lines[2:] if len(conv.lines) > 2 else conv.lines       # drop "Document ID" + summary lines
        if labels:
            body = _select_lines(body, labels)
        records.append({"document_id": conv.primary_id, "name": conv.display_name,
                        "text": "; ".join(body)[:MAX_RECORD_CHARS]})
    return ExactResult(collection=name, total=total, records=records, conditions="; ".join(texts) or "none",
                       complete=total <= len(records), source=source, operation=plan["operation"], notes=notes,
                       timings={"db": round((time.perf_counter() - started) * 1000)})


def _context(result: ExactResult) -> str:
    lines = [f"EXACT QUERY RESULT from the '{result.collection}' collection (read live from the database, not a similarity search).",
             f"Conditions applied: {result.conditions}"]
    if result.source == "name-match":
        lines.append("Conditions were NOT applied by the system. These are ALL records of the collection: "
                     "read them and apply the question's conditions yourself.")
        lines.append(f"Exact number of records in the collection: {result.total}")
    else:
        lines.append(f"Exact number of matching records in the database: {result.total}")
    if result.complete:
        lines.append(f"All {len(result.records)} records are listed below: the list is complete.")
    elif result.operation == "count":
        lines.append(f"Only {len(result.records)} examples are listed below. The total above is exact.")
    else:
        lines.append(f"Only the first {len(result.records)} of {result.total} are listed below: the list is NOT complete. "
                     "The total above is exact.")
    lines.extend(f"Note: {n}" for n in result.notes)
    lines.append("")
    lines.extend(f"[{i}] {r['text']}" for i, r in enumerate(result.records, 1))
    return "\n".join(lines)


_WORK_INTENT = re.compile(
    r"\b(?:tasks?|projects?|work|stuff|assignments?|deliverables?|status|completed?|complete|done|finish(?:es|ed)?|finished|pending|open|active|to[- ]?do|in[- ]progress|in[- ]review|review|assigned)\b",
    re.IGNORECASE,
)
_PROJECT_INTENT = re.compile(r"\bprojects?\b", re.IGNORECASE)
_TASK_INTENT = re.compile(r"\b(?:tasks?|work|stuff|assignments?|deliverables?)\b", re.IGNORECASE)
_TASK_COLLECTION_INTENT = re.compile(r"\b(?:tasks?|assignments?|deliverables?)\b", re.IGNORECASE)
_NAME_STOPS = {"what", "which", "who", "whom", "whose", "where", "how", "many", "show", "list", "tell", "give", "me", "the", "is", "are", "was", "were",
               "has", "have", "had", "did", "does", "do", "all", "every", "any", "employee", "employees", "user", "users",
               "member", "members", "completed", "complete", "done",
               "finished", "pending", "open", "active", "current", "task", "tasks", "project", "projects", "work",
               "stuff", "status", "of", "for", "by", "assigned", "him", "her", "them",
               "time", "times", "logged", "log", "logs", "hour", "hours", "timesheet", "timesheets",
               "least", "most", "less", "more", "fewer", "fewest", "highest", "lowest",
               "on", "in", "at", "to", "from", "with", "about", "into", "through", "during", "a", "an", "whether", "if"}


def _employee_name_from_question(question: str):
    """Extract an explicit employee name from common possessive and preposition forms."""
    patterns = (
        r"\b([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})['’]s\s+(?:(?:completed?|done|finished|pending|open|active|current|all|to[- ]?do|in[- ]progress|in[- ]review|review)\s+)*(?=(?:tasks?|projects?|work|stuff|status|assignments?|deliverables?)\b)",
        r"\b(?:what(?:\s+(?:did|has|have|does))?|show|list|find|check|tell\s+me)\s+([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2}?)(?=\s+(?:do|did|has|have|finish(?:ed)?|complete(?:d)?|tasks?|projects?|work|stuff|assignments?|deliverables?)\b)",
        r"\b([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})\s+(?=(?:worked|works|working|completed?|finished|assigned|did|has|have)\b)",
        r"\b([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})\s+(?=(?:tasks?|projects?|work|stuff|status|assignments?|deliverables?)\b)",
        r"\b(?:assigned\s+to|for|of|by)\s+([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3}?)(?=\s+(?:(?:are|is|has|have|did|does|do)\s+)?(?:tasks?|projects?|work|stuff|status|assignments?|deliverables?|completed?|done|finished|pending|in[- ]progress)\b|['’]s\b|[?.!,]|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            candidate = re.sub(r"['’]s$", "", match.group(1), flags=re.IGNORECASE)
            words = candidate.split()
            while words and words[0].lower() in _NAME_STOPS:
                words.pop(0)
            while words and words[-1].lower() in _NAME_STOPS:
                words.pop()
            if words:
                return " ".join(words[:4]).strip(" .'-")
    return None


def _target_project_from_question(question: str, person_name: str = None):
    """Extract a target project or board name if specified in the question."""
    patterns = (
        r"\b(?:in|on|at|for)\s+(?:the\s+)?(.+?)\s+(?:project|board|task|card)\b",
        r"\b(?:in|on|at|for)\s+(?:the\s+)?(?:project|board|task|card)\s+(.+?)(?=[?.!,]|$)\b",
        r"\b(?:project|board|task|card)\s+(.+?)(?=[?.!,]|$)\b",
        r"\b(?:in|on|at|for)\s+(?:the\s+)?([A-Za-z0-9_ -]+?)(?=[?.!,]|$)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).strip(" .'-")
        candidate = re.sub(r"^(?:the\s+)?(?:project|board|task|card)\s+", "", raw, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+(?:project|board|task|card)$", "", candidate, flags=re.IGNORECASE).strip(" .'-")
        if person_name:
            candidate = re.sub(rf"^{re.escape(person_name)}\s*", "", candidate, flags=re.IGNORECASE).strip(" .'-")
        words = candidate.split()
        while words and words[0].lower() in _NAME_STOPS:
            words.pop(0)
        while words and words[-1].lower() in _NAME_STOPS:
            words.pop()
        candidate = " ".join(words).strip(" .'-")
        if not candidate:
            continue
        if re.search(r"\b(?:which|what|who|whom|whose|where|how|konsa|konse|kisme)\b", candidate, re.IGNORECASE):
            continue
        if person_name and _norm(candidate) == _norm(person_name):
            continue
        return candidate
    return None


def _id_variants(values):
    """Make both BSON ObjectId and string forms available for mixed-schema references."""
    variants = []
    for value in values:
        variants.append(value)
        text = str(value)
        if HEX24.match(text):
            try:
                variants.extend((text, ObjectId(text)))
            except Exception:
                pass
    return list(dict.fromkeys(variants))


# ---------------------------------------------------------------- workspace scoping
_WS_DIRECT_FIELDS = ("workspaceId", "workspace_id", "workspace", "lastActiveWorkspace")
_WS_USER_FIELDS = ("user", "userId", "memberId")


def _and_merge(*filters) -> dict:
    parts = [f for f in filters if f]
    if not parts:
        return {}
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _direct_ws_condition(ws_variants: list) -> dict:
    return {"$or": [{field: {"$in": ws_variants}} for field in _WS_DIRECT_FIELDS]}


def _ids_in_workspace(db, name: str, ws_variants: list, workspace_id) -> list:
    """Document ids of `name` that belong to the workspace (direct fields only, no recursion)."""
    if not workspace_id:
        return None        # signals "no restriction"
    return [doc["_id"] for doc in db[name].find(_direct_ws_condition(ws_variants), {"_id": 1})]


def _user_ids_in_workspace(db, ws_variants: list) -> list:
    ids = [doc["_id"] for doc in db["users"].find(_direct_ws_condition(ws_variants), {"_id": 1})]
    for coll in ("workspacememberships", "memberships"):
        for doc in db[coll].find(_direct_ws_condition(ws_variants), {"user": 1, "userId": 1, "memberId": 1}):
            for field in _WS_USER_FIELDS:
                value = doc.get(field)
                if value is None:
                    continue
                ids.extend(_id_variants([value]))
    return list(dict.fromkeys(ids))


def _workspace_filter(name: str, workspace_id, db=None):
    """Mongo filter restricting reads of `name` to one workspace, or None when unrestricted.

    Collections with no mappable workspace link match nothing (fail closed).
    """
    if not workspace_id:
        return None
    db = mongo_service.get_db() if db is None else db
    ws_variants = _id_variants([str(workspace_id)])
    if name == "workspaces":
        return {"_id": {"$in": ws_variants}}
    if name == "users":
        user_ids = _user_ids_in_workspace(db, ws_variants)
        return {"_id": {"$in": user_ids}} if user_ids else {"_id": None}
    if name == "tasks":
        project_ids = _ids_in_workspace(db, "projects", ws_variants, workspace_id)
        return {"projectId": {"$in": project_ids}} if project_ids else {"_id": None}
    if name == "boards":
        project_ids = _ids_in_workspace(db, "projects", ws_variants, workspace_id)
        return {"$or": [{"workspaceId": {"$in": ws_variants}}, {"workspace": {"$in": ws_variants}},
                        {"project": {"$in": project_ids}}]}
    if name == "cards":
        board_ids = _ids_in_workspace(db, "boards", ws_variants, workspace_id)
        return {"$or": [{"workspaceId": {"$in": ws_variants}}, {"workspace": {"$in": ws_variants}},
                        {"board": {"$in": board_ids}}]}
    return _direct_ws_condition(ws_variants)


def _assigned_work_filter(ids, person_name, completed_only, open_only, collection_name):
    """Build assignment/status filters for older and newer work-item schemas."""
    fields = {
        "tasks": ("assignees", "assignedTo", "assigned_to", "assignee", "user", "userId"),
        "subtasks": ("assignees", "assignedTo", "assigned_to", "assignee", "user", "userId"),
        "subtasknanos": ("assignees", "assignedTo", "assigned_to", "assignee", "user", "userId"),
    }.get(collection_name, ("assignees",))
    clauses = []
    name_pattern = re.compile(re.escape(person_name), re.IGNORECASE)
    for field_name in fields:
        clauses.append({field_name: {"$in": ids}})
        # Some legacy collections store the assignee's name instead of a user id;
        # these dotted paths also cover arrays of embedded assignee objects.
        for suffix in ("name", "user", "userId", "_id"):
            clauses.append({f"{field_name}.{suffix}": {"$in": ids}})
            clauses.append({f"{field_name}.{suffix}": name_pattern})
        clauses.append({field_name: name_pattern})

    query = {"$or": clauses}
    if completed_only:
        query["status"] = {"$in": [re.compile(r"^(done|completed?|finished|closed)$", re.IGNORECASE)]}
    elif open_only:
        query["status"] = {"$nin": [re.compile(r"^(done|completed?|finished|closed|cancelled?|canceled?)$", re.IGNORECASE)]}
        query["isArchived"] = {"$ne": True}
    return query


def _fallback_assigned_work(db, existing, ids, person_name, completed_only, open_only, count_only,
                            personal=False, include_logged_time=False, workspace_id=None):
    """If cards have no matches, try legacy task and subtask collections in order."""
    for collection_name in ("tasks", "subtasks", "subtasknanos"):
        if collection_name not in existing:
            continue
        query = _assigned_work_filter(ids, person_name, completed_only, open_only, collection_name)
        query = _and_merge(_workspace_filter(collection_name, workspace_id, db), query)
        collection = db[collection_name]
        total = _count(collection, query)
        if not total:
            continue
        cursor = collection.find(query).sort("_id", -1)
        if not personal:
            cursor = cursor.limit(10 if count_only else settings.STRUCT_MAX_RECORDS)
        docs = list(cursor)
        if not include_logged_time:
            docs = [_without_logged_time(doc) for doc in docs]
        resolver = _exec_resolver(build_catalog())
        resolver.prefetch(collection_name, docs, infer=False)
        records = []
        for doc in docs:
            converted = document_converter.convert_document(collection_name, doc, resolver)
            body = converted.lines[2:] if len(converted.lines) > 2 else converted.lines
            title = doc.get("title") or doc.get("name") or converted.display_name
            records.append({"document_id": converted.primary_id, "name": title,
                            "text": "; ".join(body)[:MAX_RECORD_CHARS]})
        return collection_name, total, records, docs
    return None


def _employee_work_status(question: str, history: list = None, current_user_id: str = None,
                          workspace_id=None):
    """Resolve an employee to their user id, then query board membership or assigned cards."""
    work_subject = bool(_TASK_INTENT.search(question) or _PROJECT_INTENT.search(question)
                        or re.search(r"\bwhat\s+(?:did|has|have|does)\b.*\b(?:do|done|finish(?:ed)?)\b",
                                     question, re.IGNORECASE))
    prior_work_subject = any(
        _TASK_INTENT.search(turn.get("question", "")) or _PROJECT_INTENT.search(turn.get("question", ""))
        for turn in history or []
    )
    if not work_subject and not prior_work_subject:
        return None
    scoped_question = question
    personal = bool(current_user_id and re.search(r"\b(?:my|mine|myself)\b", question, re.IGNORECASE))
    person_name = None if personal else _employee_name_from_question(question)
    if not person_name:
        for turn in reversed(history or []):
            prior = turn.get("question", "")
            if _WORK_INTENT.search(prior) and (person_name := _employee_name_from_question(prior)):
                scoped_question = f"{prior} {question}"
                break
    if not person_name and not personal:
        return None
    if personal:
        user_ids = _id_variants([current_user_id])
        person = mongo_service.get_db()["users"].find_one(
            {"_id": {"$in": user_ids}}, {"name": 1, "firstName": 1, "lastName": 1}
        )
        person_name = (person or {}).get("name") or " ".join(filter(None, (
            (person or {}).get("firstName"), (person or {}).get("lastName")
        ))) or "you"
    else:
        user_ids = _ids_by_name("users", person_name)
    if not user_ids:
        return None

    db = mongo_service.get_db()
    existing = set(db.list_collection_names())
    ids = _id_variants(user_ids)
    ws_cards = _workspace_filter("cards", workspace_id, db)
    ws_boards = _workspace_filter("boards", workspace_id, db)
    ws_tasks = _workspace_filter("tasks", workspace_id, db)
    ws_users = _workspace_filter("users", workspace_id, db)
    include_logged_time = wants_logged_time(question)
    started = time.perf_counter()
    count_only = bool(re.search(r"\bhow\s+many\b|\bcount\b|\btotal\b", question, re.IGNORECASE))
    completed_only = bool(
        (_TASK_INTENT.search(question) or _PROJECT_INTENT.search(question))
        and re.search(r"\b(?:completed?|done|finished)\b", question, re.IGNORECASE)
        and not re.search(r"\b(?:not|never|incomplete|pending|open|or\s+not|status|whether)\b", question, re.IGNORECASE)
    )
    open_only = bool(re.search(r"\b(?:not\s+(?:completed?|done)|incomplete|pending|open|active|to[- ]?do|in[- ]progress|in[- ]review|review)\b",
                               question, re.IGNORECASE))
    target_project = _target_project_from_question(question, person_name) or _target_project_from_question(scoped_question, person_name)
    if target_project:
        target_board_ids = _ids_by_name("boards", target_project)
        matching_boards = list(db["boards"].find({"_id": {"$in": _id_variants(target_board_ids)}})) if target_board_ids else []
        board_title = matching_boards[0].get("name") if matching_boards else target_project

        is_member = any(any(uid in (b.get("members") or []) for uid in ids) for b in matching_boards)
        b_var = _id_variants(target_board_ids)
        card_filter = {"board": {"$in": b_var}, "assignees": {"$in": ids}} if b_var else {"_id": {"$in": []}}
        cards = list(db["cards"].find(card_filter, {"title": 1, "status": 1, "isArchived": 1, "board": 1, "loggedTime": 1}))
        if not cards and b_var:
            cards = list(db["cards"].find({"board": {"$in": b_var}, "loggedTime.user": {"$in": ids}},
                                          {"title": 1, "status": 1, "isArchived": 1, "board": 1, "loggedTime": 1}))
        if not cards and not is_member and "tasks" in existing and b_var:
            for coll in ("tasks", "subtasks"):
                if coll in existing:
                    found_tasks = list(db[coll].find({"projectId": {"$in": b_var}, "assignedTo": {"$in": ids}}, {"title": 1, "name": 1, "status": 1}))
                    if found_tasks:
                        cards = [{"title": t.get("title") or t.get("name"), "status": t.get("status")} for t in found_tasks]
                        break

        is_verification = bool(re.search(r"\b(?:does|did|is|has|have|was|were|whether|if)\b", question, re.IGNORECASE))
        has_worked = bool(cards or is_member)
        u_doc = db["users"].find_one({"_id": {"$in": ids}}, {"name": 1, "firstName": 1, "lastName": 1})
        display_person = (u_doc or {}).get("name") or " ".join(filter(None, (
            (u_doc or {}).get("firstName"), (u_doc or {}).get("lastName")
        ))) or person_name.title()

        if not matching_boards and not has_worked:
            # Check if target refers to a specific task / card
            matching_task_cards = list(db["cards"].find({
                "title": re.compile(rf"^{re.escape(target_project)}$", re.IGNORECASE)
            }))
            if not matching_task_cards:
                matching_task_cards = list(db["cards"].find({
                    "title": re.compile(re.escape(target_project), re.IGNORECASE)
                }))
            if matching_task_cards:
                matched_c = matching_task_cards[0]
                b_doc = db["boards"].find_one({"_id": matched_c.get("board")})
                if not b_doc:
                    try:
                        b_doc = db["boards"].find_one({"_id": ObjectId(str(matched_c.get("board")))})
                    except Exception:
                        pass
                b_title = (b_doc.get("name") if b_doc else None) or (b_doc.get("title") if b_doc else None) or "Unknown project"
                t_user_entries = [e for e in matched_c.get("loggedTime", []) if str(e.get("user")) in [str(i) for i in ids]]
                t_assigned = any(str(a) in [str(i) for i in ids] for a in matched_c.get("assignees", []))
                if t_user_entries or t_assigned:
                    lt_info = f" Records show {len(t_user_entries)} logged-time entries." if t_user_entries else ""
                    direct_answer = f"Yes, {display_person} works on / has worked on task '{matched_c.get('title')}' under project '{b_title}'.{lt_info}"
                    return ExactResult(
                        collection="cards", total=1, records=[{
                            "document_id": str(matched_c.get("_id", "")),
                            "name": matched_c.get("title", ""),
                            "text": f"Project: {b_title}; Task: {matched_c.get('title')}; Status: {matched_c.get('status')}"
                        }],
                        conditions=f"user = {person_name}; task = {matched_c.get('title')}",
                        complete=True, source="planner", operation="lookup",
                        direct_answer=direct_answer, timings={"db": round((time.perf_counter() - started) * 1000)}
                    )
                else:
                    direct_answer = f"No, {display_person} has not worked on task '{matched_c.get('title')}' under project '{b_title}' (no assigned tasks or logged time found)."
                    return ExactResult(
                        collection="cards", total=0, records=[],
                        conditions=f"user = {person_name}; task = {matched_c.get('title')}",
                        complete=True, source="planner", operation="lookup",
                        direct_answer=direct_answer, timings={"db": round((time.perf_counter() - started) * 1000)}
                    )

        if has_worked:
            card_lines = []
            for c in cards:
                st = str(c.get("status") or "").strip()
                st_suffix = f" [{st}]" if st else ""
                card_lines.append(f"- {c.get('title') or 'Untitled'}{st_suffix}")
            if is_verification:
                if cards:
                    direct_answer = f"Yes, {display_person} works on / has worked on project '{board_title}'.\n\nAssigned tasks ({len(cards)}):\n" + "\n".join(card_lines)
                else:
                    direct_answer = f"Yes, {display_person} is a member of project '{board_title}' (no specific tasks currently assigned)."
            else:
                direct_answer = f"{display_person}'s tasks in project '{board_title}' ({len(cards)}):\n" + "\n".join(card_lines)
        else:
            if not matching_boards:
                direct_answer = f"No project matching '{target_project}' was found (and {display_person} has no assignments in it)."
            else:
                direct_answer = f"No, {display_person} has not worked on project '{board_title}' (no assigned tasks or membership found)."

        records = []
        for c in cards:
            st = str(c.get("status") or "").strip()
            records.append({
                "document_id": str(c.get("_id", "")),
                "name": c.get("title") or "Untitled task",
                "text": f"Project: {board_title}; Task: {c.get('title') or 'Untitled'}; Status: {st}"
            })
        conditions = f"user = {person_name}; project = {board_title}"
        return ExactResult(
            collection="cards",
            total=len(cards),
            records=records,
            conditions=conditions,
            complete=True,
            source="employee-id",
            operation="list",
            notes=[f"Checked work and assignments for {person_name} in project '{board_title}'."],
            timings={"db": round((time.perf_counter() - started) * 1000)},
            direct_answer=direct_answer,
        )

    is_project_query = bool(_PROJECT_INTENT.search(scoped_question) and not _TASK_COLLECTION_INTENT.search(scoped_question))
    records, total, collection_name = [], 0, "boards" if is_project_query else "cards"
    docs = []
    board_names = {}

    if is_project_query and "boards" in existing:
        leave_bids = _leave_board_ids(db)
        member_clause = {"members": {"$in": ids}}
        assigned_board_ids = []
        if "cards" in existing:
            assigned_board_ids = _id_variants([
                card.get("board") for card in db["cards"].find(
                    {"assignees": {"$in": ids}, "board": {"$exists": True}}, {"board": 1}
                ) if card.get("board") is not None and str(card.get("board")) not in leave_bids
            ])
        base_clause = {"$or": [member_clause, {"_id": {"$in": assigned_board_ids}}]} if assigned_board_ids else member_clause
        board_filter = _and_merge(
            base_clause,
            {"name": {"$nin": [re.compile(r"^leaves?$", re.IGNORECASE)]}}
        )
        if completed_only:
            board_filter["status"] = {"$in": [re.compile(r"^(done|completed?|finished|closed)$", re.IGNORECASE)]}
        elif open_only:
            board_filter["status"] = {"$nin": [re.compile(r"^(done|completed?|finished|closed|cancelled?|canceled?)$", re.IGNORECASE)]}
            board_filter["isArchived"] = {"$ne": True}
        total = _count(db["boards"], board_filter)
        cursor = db["boards"].find(board_filter, {
            "name": 1, "description": 1, "status": 1, "isArchived": 1, "dueDate": 1,
        }).sort("_id", -1)
        if not personal:
            cursor = cursor.limit(10 if count_only else settings.STRUCT_MAX_RECORDS)
        docs = list(cursor)
        for doc in docs:
            conv = document_converter.convert_document("boards", doc)
            body = conv.lines[2:] if len(conv.lines) > 2 else conv.lines
            records.append({"document_id": conv.primary_id, "name": doc.get("name") or conv.display_name,
                            "text": "; ".join(body)})
    elif "cards" in existing:
        leave_bids = _leave_board_ids(db)
        card_filter = {"assignees": {"$in": ids}}
        if leave_bids:
            card_filter["board"] = {"$nin": _id_variants(list(leave_bids))}
        if completed_only:
            card_filter["status"] = {"$in": [re.compile(r"^(done|completed?|finished|closed)$", re.IGNORECASE)]}
        elif open_only:
            card_filter["status"] = {"$nin": [re.compile(r"^(done|completed?|finished|closed|cancelled?|canceled?)$", re.IGNORECASE)]}
            card_filter["isArchived"] = {"$ne": True}
        total = _count(db["cards"], card_filter)
        projection = {"title": 1, "description": 1, "board": 1, "status": 1, "isArchived": 1,
                      "dueDate": 1, "createdAt": 1, "assignees": 1}
        if include_logged_time:
            projection["loggedTime"] = 1
        cursor = db["cards"].find(card_filter, projection).sort("_id", -1)
        if not personal:
            cursor = cursor.limit(10 if count_only else settings.STRUCT_MAX_RECORDS)
        docs = list(cursor)
        board_ids = _id_variants([doc.get("board") for doc in docs if doc.get("board") is not None])
        board_names = {}
        if board_ids and "boards" in existing:
            for board in db["boards"].find({"_id": {"$in": board_ids}}, {"name": 1, "project": 1}):
                board_names[str(board["_id"])] = board.get("name") or board.get("project")
        for doc in docs:
            conv = document_converter.convert_document("cards", doc)
            body = conv.lines[2:] if len(conv.lines) > 2 else conv.lines
            project = board_names.get(str(doc.get("board")), "Unknown project")
            body.append(f"Project: {project}")
            records.append({"document_id": conv.primary_id, "name": doc.get("title") or conv.display_name,
                            "text": "; ".join(body)})
    else:
        collection_name = "cards"

    # Cards are the primary source. Legacy imports or older app versions may
    # still hold assigned work in these collections, so use them only if cards
    # returned no matches.
    if not is_project_query and total == 0:
        fallback = _fallback_assigned_work(
            db, existing, ids, person_name, completed_only, open_only, count_only, personal,
            include_logged_time=include_logged_time,
        )
        if fallback is not None:
            collection_name, total, records, fallback_docs = fallback
            if personal:
                docs = fallback_docs

    direct_answer = ""
    if personal and records:
        lines = [f"All {person_name}'s {collection_name} records ({total} total):", ""]
        for doc in docs:
            if collection_name == "boards":
                title = doc.get("name") or doc.get("title") or "Untitled project"
                project = ""
                status = str(doc.get("status") or "").strip()
                description = doc.get("description") or ""
                due = doc.get("dueDate")
                created = doc.get("createdAt")
                if hasattr(due, "isoformat"):
                    due = due.isoformat()
                if hasattr(created, "isoformat"):
                    created = created.isoformat()
                lines.append(f"Board: {title}")
                if project:
                    lines.append(f"  Project: {project}")
                if status:
                    lines.append(f"  Status: {status}")
                if description:
                    lines.append(f"  Description: {description}")
                if due:
                    lines.append(f"  Due: {due}")
                if created:
                    lines.append(f"  Created: {created}")
                lines.append("")
            else:
                title = doc.get("title") or doc.get("name") or "Untitled task"
                board_id = doc.get("board")
                project = board_names.get(str(board_id), "Unknown project") if board_id is not None else "Unknown project"
                status = str(doc.get("status") or "").strip()
                description = doc.get("description") or ""
                due = doc.get("dueDate")
                created = doc.get("createdAt")
                if hasattr(due, "isoformat"):
                    due = due.isoformat()
                if hasattr(created, "isoformat"):
                    created = created.isoformat()
                assignees = doc.get("assignees") or []
                if isinstance(assignees, list):
                    assignee_names = []
                    for a in assignees:
                        a_str = str(a)
                        if _HEX24_STANDALONE.fullmatch(a_str):
                            continue   # never expose raw ObjectIds
                        assignee_names.append(a_str)
                    assignees = ", ".join(assignee_names)
                elif assignees:
                    assignees = _sanitize_ids(str(assignees))
                lines.append(f"Card: {title}")
                lines.append(f"  Project: {project}")
                if status:
                    lines.append(f"  Status: {status}")
                if description:
                    lines.append(f"  Description: {description}")
                if due:
                    lines.append(f"  Due: {due}")
                if created:
                    lines.append(f"  Created: {created}")
                if assignees:
                    lines.append(f"  Assignees: {assignees}")
                logged = doc.get("loggedTime") or []
                if isinstance(logged, dict):
                    logged = [logged]
                if include_logged_time and logged:
                    lines.append("  Logged time entries:")
                    for entry in logged:
                        if not isinstance(entry, dict):
                            continue
                        hours = _to_number(entry.get("hours")) or 0
                        minutes = _to_number(entry.get("minutes")) or 0
                        entry_date = entry.get("date") or entry.get("createdAt")
                        if hasattr(entry_date, "isoformat"):
                            entry_date = entry_date.isoformat()
                        entry_desc = strip_html(entry.get("description") or "")
                        lines.append(f"    - {entry_date}: {hours}h {minutes}m" + (f" — {entry_desc}" if entry_desc else ""))
                        for key, value in entry.items():
                            if key in {"_id", "user", "userId", "user_id", "hours", "minutes", "date",
                                       "createdAt", "description"}:
                                continue
                            if isinstance(value, bool):
                                lines.append(f"      {key}: {'Yes' if value else 'No'}")
                            elif isinstance(value, (str, int, float)) and str(value).strip():
                                lines.append(f"      {key}: {' '.join(str(value).split())}")
                lines.append("")
        direct_answer = "\n".join(lines).strip()

    complete = total <= len(records)
    result = ExactResult(
        collection=collection_name, total=total, records=records,
        conditions=f"user = {person_name}" + ("; completed work" if completed_only else "; open work" if open_only else ""),
        complete=complete, source="employee-id", operation="count" if count_only else "list",
        notes=[f"Resolved {person_name} to a user ID and looked up linked {collection_name} records."],
        timings={"db": round((time.perf_counter() - started) * 1000)}, direct_answer=direct_answer,
    )
    result.context = _context(result)
    return result


_LOGGED_TIME_PATTERN = re.compile(
    r"\b(?:logged[\s_-]*time|time[\s_-]*logged|timesheets?|time[\s_-]*(?:logs?|entries|tracking|tracked|spent)|"
    r"work[\s_-]*logs?|hours?[\s_-]*(?:logged|worked|tracked|spent|billable)|"
    r"(?:logged|worked|tracked|spent)[\s_-]*hours?|billed[\s_-]*time|billable[\s_-]*hours|"
    r"(?:time[\s_-]*log|time[\s_-]*logs|time[\s_-]*entry|time[\s_-]*entries)|"
    r"(?:ghante|ghanta)[\s_-]*(?:logged|worked|spent)?)\b",
    re.IGNORECASE,
)
_LEAVE_TIME_PATTERN = re.compile(
    r"\b(?:leave|leaves|vacation|time[\s_-]*off|holiday|holidays|accrual|accrued|credit|credited|allowance|policy)\b",
    re.IGNORECASE,
)
_LOGGED_TIME_GENERIC = {
    "all", "all boards", "all logged time", "all projects", "all time", "all time entries",
    "all work logs", "board", "boards", "each", "each board", "each project", "every",
    "every board", "every project", "hour", "hours", "logged time", "loggedtime", "project",
    "projects", "time entries", "time entry", "time logged", "timesheet", "timesheets", "total",
    "total hours", "total logged time", "work log", "work logs", "task", "tasks", "card", "cards",
}
_LOGGED_TIME_STOPS = _NAME_STOPS | {
    "all", "calculate", "compute", "each", "entries", "entry", "every", "full", "give", "hour", "hours",
    "list", "log", "logged", "logs", "show", "sum", "time", "timesheet", "timesheets", "total", "work",
    "worked", "working", "least", "most", "highest", "lowest", "minimum", "maximum", "min", "max",
    "less", "more", "fewer", "fewest", "greater", "higher", "lower",
    "smallest", "largest", "shortest", "longest", "latest", "earliest", "first", "last",
    "my", "mine", "myself", "me", "i", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them", "their", "theirs", "themselves",
    "has", "have", "had", "does", "do", "did", "is", "are", "was", "were", "gets", "got", "takes", "took",
    "sab", "sabse", "sase", "thoda", "thodi", "thode", "kam", "jyada", "jayda", "zyada", "jada", "bada", "badi", "bade",
    "chhota", "chota", "chhoti", "choti", "chote", "chhote", "adhiktam", "nyuntam",
    "ka", "ki", "ke", "ko", "se", "ne", "me", "mein", "par", "pe",
    "user", "member", "employee", "users", "members", "employees",
}

_LEAVE_TYPE_PATTERN = re.compile(
    r"\b(?:full[\s_-]*day\s+leave|full[\s_-]*leave|short[\s_-]*day\s+leave|short[\s_-]*leave|"
    r"halfday\s+leave|half[- ]day\s+leave|fullleave|shortleave|halfdayleave)\b",
    re.IGNORECASE,
)
_LEAVE_TYPES_PATTERN = re.compile(
    r"\b(?:types?\s+of\s+leaves?|leaves?\s+types?|leave\s+types?|leaves?\s+categories?|leave\s+categories?|"
    r"what\s+(?:leave|leaves)(?!\s+did)|which\s+(?:leave|leaves)(?!\s+did)|how\s+many\s+types?\s+of\s+leaves?)\b",
    re.IGNORECASE,
)
_LEAVE_TIME_QUESTION_PATTERN = re.compile(
    r"\b(?:logged[\s_-]*time|time[\s_-]*logged|timesheets?|hours?[\s_-]*(?:logged|worked|tracked|spent)|"
    r"(?:logged|worked|tracked|spent)[\s_-]*hours?)\b",
    re.IGNORECASE,
)
# Superlative leave-count queries: "who takes most leaves", "who takes least full-day leaves",
# "which employee takes leave most frequently", "highest number of full-day leaves", etc.
# Strategy: require a superlative keyword AND a leave keyword in the same query.
_LEAVE_COUNT_SUPERLATIVE_PATTERN = re.compile(
    r"(?:"
    # Superlative keyword
    r"\b(?:most|least|highest|lowest|maximum|minimum|max|min|fewest|fewer|more|less|very\s+most|very\s+least|most\s+frequently|least\s+frequently|more\s+frequently)\b"
    r".*?"
    # Leave keyword
    r"\b(?:leaves?|leave\s+days?|(?:full|short|half)[- ]?(?:day\s+)?leaves?)\b"
    r"|"
    # Leave keyword followed by superlative
    r"\b(?:leaves?|leave\s+days?|(?:full|short|half)[- ]?leave)\b"
    r".*?"
    r"\b(?:most\s+frequently|least\s+frequently|more\s+frequently|most|least|highest|lowest|maximum|minimum|fewest|fewer|less|more)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)


_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# Bare leave-history questions that never mention hours/timesheet wording:
# "leaves of mohit in sep", "mohit's leaves in september", "how many leaves did
# mohit take in august", "leaves taken by rohit". Without this they fall through
# to similarity search and the LLM, which answers them unreliably.
_LEAVE_HISTORY_PATTERN = re.compile(
    r"(?:"
    r"\bleaves?\s+(?:of|for|by)\b"
    r"|\bleaves?\s+(?:taken|take)\s+by\b"
    r"|\b(?:how\s+many|number\s+of)\s+leaves?\b"
    r"|\bdid\s+[A-Za-z][A-Za-z'-]*\s+take\s+(?:any\s+|a\s+)?leaves?\b"
    r"|\b[A-Za-z][A-Za-z'-]*['’]s\s+leaves?\b"
    r"|\b[A-Za-z][A-Za-z'-]+\s+leaves?\s+(?:in|during|for)\b"
    r")",
    re.IGNORECASE,
)

_LEAVE_REQUEST_INTENT = re.compile(r"\b(?:leave|leaves|vacation|time\s+off)\b", re.IGNORECASE)
_LEAVE_REQUEST_EXCLUSIONS = re.compile(
    r"\b(?:logged\s*time|time\s*logged|timesheets?|hours?\s+(?:logged|worked|tracked|spent)|"
    r"leave\s+types?|types?\s+of\s+leaves?|policy|policies|accrual|accrued|credit|credits?|"
    r"allowance|balance|entitlement|project|board|card|task)\b", re.IGNORECASE)
_LEAVE_REQUEST_STATUS_FIELDS = ("status", "approvalStatus", "requestStatus", "state")
_LEAVE_REQUEST_OWNER_FIELDS = (
    "user", "userId", "user_id", "employee", "employeeId", "employee_id", "member", "memberId",
    "requester", "requesterId", "requestedBy", "requestedById", "applicant", "applicantId",
    "createdBy", "createdById", "submittedBy", "submittedById", "owner", "ownerId",
)

# Words that never belong to a person's name when extracting one from a
# leave-history question ("leaves of mohit in sep" -> "mohit").
_LEAVE_NAME_STOPS = _NAME_STOPS | {
    "leave", "leaves", "in", "on", "during", "this", "last", "the", "a", "an", "and",
    "did", "do", "does", "take", "took", "has", "have", "had", "get", "got",
    "month", "year", "week", "today", "yesterday", "request", "requests", "application", "applications",
    "status", "history", "records", "record", "list", "show", "all", "every",
} | set(_MONTH_MAP)

_THIS_PERIOD_PATTERN = re.compile(r"\bthis\s+(week|month|year|quarter|semester|half[\s-]?year|halfyear)\b", re.IGNORECASE)
_BARE_YEAR_PATTERN = re.compile(r"\b(\d{4})\b", re.IGNORECASE)
_BARE_MONTH_PATTERN = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE)


# Concrete date references (absolute dates) that can anchor relative-time questions.
_EXTRACT_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}|\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?)\b"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)


def _extract_date_from_question(text: str, anchor=None):
    """Extract (year, month, day) from question text if present (supports relative dates)."""
    if not text:
        return None
    from datetime import datetime, timedelta
    anchor = anchor or datetime.now()

    # Relative date expressions
    if re.search(r"\b(?:day\s+before\s+yesterday|parso)\b", text, re.IGNORECASE):
        d = anchor - timedelta(days=2)
        return d.year, d.month, d.day
    if re.search(r"\b(?:yesterday|kal\s+(?:ka|ki|ke|ko))\b", text, re.IGNORECASE) or (
        re.search(r"\bkal\b", text, re.IGNORECASE) and re.search(r"\b(?:worked|did|tha|the|thi)\b", text, re.IGNORECASE)
    ):
        d = anchor - timedelta(days=1)
        return d.year, d.month, d.day
    if re.search(r"\b(?:today|aaj)\b", text, re.IGNORECASE):
        return anchor.year, anchor.month, anchor.day

    # 1. YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # 2. Month word + day (e.g. August 19, Aug 19, Aug 19th, August 19 2026)
    m = re.search(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[.,\s]+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\b", text, re.IGNORECASE)
    if m:
        month = _MONTH_MAP[m.group(1).lower()[:3]]
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None
        return year, month, day
    # 3. Day + month word (e.g. 19 August, 19th Aug 2026)
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s*,?\s*(\d{4}))?\b", text, re.IGNORECASE)
    if m:
        day = int(m.group(1))
        month = _MONTH_MAP[m.group(2).lower()[:3]]
        year = int(m.group(3)) if m.group(3) else None
        return year, month, day
    # 4. MM-DD or M-D (e.g. 8-19, 08-19, 8/19)
    m = re.search(r"\b(?:on\s+|date\s+)?(\d{1,2})[-/](\d{1,2})(?:[-/](\d{2,4}))?\b", text, re.IGNORECASE)
    if m:
        num1, num2 = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None
        if year and year < 100:
            year += 2000
        if 1 <= num1 <= 12 and 1 <= num2 <= 31:
            return year, num1, num2
        if 1 <= num2 <= 12 and 1 <= num1 <= 31:
            return year, num2, num1
    # 5. Bare month name (e.g. "September", "Aug", "December 2026") -> month scope
    m_month = _BARE_MONTH_PATTERN.search(text)
    if m_month:
        month = _MONTH_MAP[m_month.group(1).lower()[:3]]
        year = None
        yr_m = _BARE_YEAR_PATTERN.search(text)
        if yr_m:
            year = int(yr_m.group(1))
        if year is None:
            year = (anchor or datetime.now()).year
        return year, month, None
    # 6. Bare year (e.g. "2026") -> year scope
    m_year = _BARE_YEAR_PATTERN.search(text)
    if m_year and not re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        return int(m_year.group(1)), None, None
    return None


def _coerce_entry_dt(d_val):
    """-> datetime | None: entries may store dates as datetimes or as strings like "2026-01-02" / "2026-08-19T12:00:00"."""
    if d_val is None:
        return None
    if hasattr(d_val, "year") and hasattr(d_val, "month") and hasattr(d_val, "day"):
        if isinstance(d_val, datetime):
            return d_val
        return datetime(d_val.year, d_val.month, d_val.day)
    if isinstance(d_val, str):
        m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", d_val)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.search(r"(\d{1,2})[-/](\d{1,2})", d_val)
        if m:
            return datetime(datetime.now().year, int(m.group(1)), int(m.group(2)))
    return None


def _extract_period(text: str, anchor=None) -> dict | None:
    """Resolve the time scope of a question -> period descriptor, or None when unscoped.

    Returned shape: {"kind": "day" | "month" | "year" | "week", "year", "month", "day", "start", "end"}
      day   -> start == end == the day
      month -> start .. end of that month
      year  -> start .. end of that year
      week  -> start .. end of the week containing the anchor
    """
    text = " ".join((text or "").split())
    if not text:
        return None
    anchor = anchor or datetime.now()

    exact = _extract_date_from_question(text, anchor=anchor)
    if exact and exact[2] is not None:
        year = exact[0] or anchor.year
        month = exact[1] or 1
        day = exact[2] or 1
        start = datetime(year, month, day)
        return {"kind": "day", "year": year, "month": exact[1], "day": exact[2], "start": start, "end": start}

    m_this = _THIS_PERIOD_PATTERN.search(text)
    if m_this:
        what = m_this.group(1).lower()
        if what == "week":
            start = anchor - timedelta(days=anchor.weekday())
            return {"kind": "week", "year": start.year, "month": start.month, "day": start.day,
                    "start": start, "end": start + timedelta(days=6)}
        if what == "month":
            start = anchor.replace(day=1)
            end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return {"kind": "month", "year": start.year, "month": start.month, "day": None,
                    "start": start, "end": end}
        if what == "year":
            start = anchor.replace(month=1, day=1)
            end = start.replace(year=anchor.year + 1) - timedelta(days=1)
            return {"kind": "year", "year": start.year, "month": None, "day": None,
                    "start": start, "end": end}

    m_month = _BARE_MONTH_PATTERN.search(text)
    if m_month:
        month = _MONTH_MAP[m_month.group(1).lower()[:3]]
        year = None
        yr_m = _BARE_YEAR_PATTERN.search(text)
        if yr_m:
            year = int(yr_m.group(1))
        if year is None:
            year = anchor.year
        start = datetime(year, month, 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        return {"kind": "month", "year": year, "month": month, "day": None, "start": start, "end": end}

    m_year = _BARE_YEAR_PATTERN.search(text)
    if m_year and not re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        year = int(m_year.group(1))
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        return {"kind": "year", "year": year, "month": None, "day": None, "start": start, "end": end}

    return None


def _entry_in_period(d_val, period) -> bool:
    """True when the entry date falls inside the resolved period."""
    if not period:
        return True
    kind = period.get("kind")
    if kind == "day":
        e = _coerce_entry_dt(d_val)
        if not e:
            return False
        return (e.year, e.month, e.day) == (period.get("year"), period.get("month"), period.get("day"))
    if kind == "month":
        e = _coerce_entry_dt(d_val)
        if not e:
            return False
        month = period.get("month")
        return bool(month) and e.month == month and (period.get("year") is None or e.year == period.get("year"))
    if kind == "year":
        e = _coerce_entry_dt(d_val)
        if not e:
            return False
        year = period.get("year")
        return year is None or e.year == year
    if kind == "week":
        e = _coerce_entry_dt(d_val)
        if not e:
            return False
        week_start = datetime(e.year, e.month, e.day) - timedelta(days=e.weekday())
        return week_start == period.get("start")
    return True


def _period_label(period) -> str | None:
    """Human label for the resolved period, e.g. "2026-09-11" (day), "2026-09" (month), "2026" (year), "2026-11-09" (week)."""
    if not period:
        return None
    kind = period.get("kind")
    if kind == "week":
        start = period.get("start")
        return start.strftime("%Y-%m-%d") if start else None
    if kind == "month":
        year, month = period.get("year"), period.get("month")
        if not month:
            return None
        return f"{year}-{month:02d}" if year else f"{month:02d}"
    if kind == "year":
        return str(period.get("year"))
    if kind == "day":
        year, month, day = period.get("year"), period.get("month"), period.get("day")
        if not month or not day:
            return None
        return f"{year}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"
    return None


def is_logged_time_question(question: str) -> bool:
    """True for logged-time / timesheet / hours-worked questions, and superlative leave-count queries."""
    text = " ".join((question or "").split())
    if not text:
        return False
    # Superlative leave-count queries ("who takes most leaves") are routed via
    # _leave_time_answer (not _logged_time_answer) but we still need this to
    # return True so the caller in rag_service routes them to try_structured.
    if _LEAVE_COUNT_SUPERLATIVE_PATTERN.search(text):
        return True
    if _LEAVE_TIME_PATTERN.search(text):
        return False
    if _LOGGED_TIME_PATTERN.search(text):
        return True
    if _extract_date_from_question(text) and re.search(r"\b(?:work|worked|working|project|projects|task|tasks|logged|hours?)\b", text, re.IGNORECASE):
        return True
    if not re.search(r"\bhours?\b", text, re.IGNORECASE):
        return False
    return bool(re.search(
        r"\b(?:my|mine|worked|work|working|task|tasks|project|projects|logged|tracked|spent|total)\b",
        text, re.IGNORECASE,
    ))


def wants_logged_time(question: str) -> bool:
    """Whether the user explicitly asked for logged time / hours in this turn."""
    text = " ".join((question or "").split())
    if not text:
        return False
    if _LOGGED_TIME_PATTERN.search(text) or _LEAVE_TIME_QUESTION_PATTERN.search(text):
        return True
    return bool(re.search(r"\b(?:hours?|minutes?)\b", text, re.IGNORECASE))


_SUPERLATIVE_REGEX = (
    r"(?:least|lowest|minimum|min|smallest|shortest|less|fewer|fewest|"
    r"most|highest|maximum|max|largest|biggest|longest|more|"
    r"sabse\s*(?:thoda|kam|chhota|chota|nyuntam|jayda|jyada|zyada|jada|bada|adhiktam)|"
    r"sase\s*(?:thoda|kam|chhota|chota|nyuntam|jayda|jyada|zyada|jada|bada|adhiktam)|"
    r"kam\s*se\s*kam|jyada\s*se\s*jyada|jayda\s*se\s*jayda|thoda|kam|jyada|jayda|zyada)"
)

_LOGGED_TIME_REGEX = (
    r"(?:logged[\s_-]*time|loggedtime|time[\s_-]*logged|hours|ghante|ghanta|"
    r"timesheets?|work[\s_-]*logs?|time[\s_-]*(?:logs?|entries|entry))"
)


def _clean_person(raw: str) -> str | None:
    if not raw:
        return None
    words = raw.split()
    # Strip stop words from both ends
    while words and words[0].lower() in _LOGGED_TIME_STOPS:
        words.pop(0)
    while words and words[-1].lower() in _LOGGED_TIME_STOPS:
        words.pop()
    # Stop at the first preposition that leaked into the name span.
    # e.g. "vikas in chatbot" -> stop at "in", keep only "vikas".
    _PERSON_INNER_STOPS = {"in", "on", "at", "for", "from", "by", "of", "with", "to", "into", "about"}
    cleaned = []
    for w in words:
        if w.lower() in _PERSON_INNER_STOPS:
            break
        cleaned.append(w)
    words = cleaned if cleaned else words
    # Strip possessive apostrophe-s but NOT a bare trailing "s" from single-word
    # names like "vikas" (which must not become "vika").
    if words and re.search(r"['‘’`]s$", words[-1]):
        words[-1] = re.sub(r"['‘’`]s$", "", words[-1])
    elif words and len(words) > 1 and words[-1].lower().endswith('s') and not words[-1].lower().endswith('ss') and len(words[-1]) > 4:
        words[-1] = re.sub(r"s$", "", words[-1], flags=re.IGNORECASE)
    candidate = " ".join(words).strip(" .'‘’-")
    if candidate and _norm(candidate) not in {_norm(x) for x in _LOGGED_TIME_GENERIC} and candidate.lower() not in _LOGGED_TIME_STOPS:
        return candidate
    return None


def _logged_time_person_candidate(question: str):
    patterns = (
        r"(?:^|\b(?:in|on|across|me|mein)\s+.*?|\b(?:user|member|employee)\s+)\b(?!sabse\b|sase\b)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})(?:['’]s\s+ka|['’]s|ka|ki|ke|ko|se)?\s+(?:all\s+|total\s+)?" + f"(?:{_SUPERLATIVE_REGEX}\\s+)?" + _LOGGED_TIME_REGEX + r"\b",
        r"\b(?!sabse\b|sase\b)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})(?:['’]s\s+ka|['’]s|ka|ki|ke|ko|se)\s+(?:all\s+|total\s+)?" + f"(?:{_SUPERLATIVE_REGEX}\\s+)?" + _LOGGED_TIME_REGEX + r"\b",
        r"\b(?!sabse\b|sase\b)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})\s+(?:all\s+|total\s+)?" + f"(?:{_SUPERLATIVE_REGEX}\\s+)" + _LOGGED_TIME_REGEX + r"\b",
        f"(?:{_SUPERLATIVE_REGEX}\\s+)?" + _LOGGED_TIME_REGEX + r"\s+(?:of|for|by|from|kiska|kiski|kiske)\s+(?:user|member|employee)?\s*(?!sabse\b|sase\b)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})(?=\s+(?:in|across|on|from|between|me|mein|ka|ki|ke)\b|[?.!,]|$)",
        f"(?:{_SUPERLATIVE_REGEX}\\s+)?" + _LOGGED_TIME_REGEX + r"\s+(?:user|member|employee)?\s*(?!sabse\b|sase\b)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})\s*(?:ka|ki|ke|ko|se)\b",
        r"\b(?:of|for|by)\s+(?:user|member|employee)?\s*(?!sabse\b|sase\b)([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})(?=\s+(?:in|across|on|from|between|me|mein)\b|[?.!,]|$)",
        r"\b([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2})\s+in\s+\S+\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            candidate = _clean_person(match.group(1))
            if not candidate:
                continue
            # Security: reject names that are too short (< 4 chars) or are generic pronouns/stop words
            if len(candidate) < 4:
                continue
            if candidate.lower() in _LOGGED_TIME_STOPS:
                continue
            if candidate.lower() in {"my", "mine", "myself", "me"}:
                return None
            return candidate
    return None


def _logged_time_project_candidate(question: str, person_name: str = None):
    patterns = (
        r"\b(?:in|on|across)\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?(.+?)(?=\s+(?:from|between|by|of|for|with|during|me|mein)\b|[?.!,]|$)",
        r"\bfor\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?(.+?)(?=\s+(?:from|between|by|of|with|during|me|mein)\b|[?.!,]|$)",
        r"\b(?:logged[\s_-]*time|time[\s_-]*logged|time|hours|ghante|logs?|entries)\s+(?:in\s+|pe\s+|par\s+)?(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?(.+?)\s*(?:project|board|task|card)?\s+(?:me|mein)\b",
        r"^\s*(?:in\s+)?(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?(.+?)\s*(?:project|board|task|card)?\s+(?:me|mein)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).strip(" .'-")
        candidate = re.sub(r"^(?:the\s+)?(?:project|board|task|card)\s+", "", raw, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+(?:project|board|task|card)$", "", candidate, flags=re.IGNORECASE).strip(" .'-")
        candidate = re.sub(r"\s+(?:me|mein|par|pe|ka|ki|ke)$", "", candidate, flags=re.IGNORECASE).strip(" .'-")
        if person_name:
            candidate = re.sub(rf"^{re.escape(person_name)}\s*(?:ka|ki|ke|ko|se)?\s*", "", candidate, flags=re.IGNORECASE).strip(" .'-")
        words = candidate.split()
        while words and words[0].lower() in _LOGGED_TIME_STOPS:
            words.pop(0)
        while words and words[-1].lower() in _LOGGED_TIME_STOPS:
            words.pop()
        candidate = " ".join(words).strip(" .'-")
        if not candidate or _norm(candidate) in {_norm(x) for x in _LOGGED_TIME_GENERIC}:
            continue
        if re.search(r"\b(?:which|what|who|whom|whose|where|konsa|konse|kisme)\b", candidate, re.IGNORECASE):
            continue
        if _extract_date_from_question(candidate):
            continue
        if person_name and _norm(candidate) == _norm(person_name):
            continue
        return candidate
    return None


def _logged_time_entry_user(entry: dict):
    value = entry.get("user", entry.get("userId", entry.get("user_id")))
    if isinstance(value, dict):
        value = value.get("_id") or value.get("id")
    return str(value) if value is not None else None


def _format_duration(total_minutes: int) -> str:
    return f"{total_minutes // 60}h {total_minutes % 60}m"


_HEX24_STANDALONE = re.compile(r"(?<![\w-])[0-9a-fA-F]{24}(?![\w-])")


def _sanitize_ids(text: str) -> str:
    """Remove standalone 24-char hex ObjectId strings and resolve user IDs to names in user-facing text."""
    if not isinstance(text, str) or not text:
        return text

    # Try resolving user IDs if MongoDB is accessible
    found_ids = list(set(_HEX24_STANDALONE.findall(text)))
    if found_ids:
        try:
            db = mongo_service.get_db()
            candidates = []
            for sid in found_ids:
                candidates.append(sid)
                if len(sid) == 24:
                    try:
                        candidates.append(ObjectId(sid))
                    except Exception:
                        pass
            users_map = {}
            for u in db["users"].find({"_id": {"$in": candidates}}, {"name": 1, "fullName": 1, "firstName": 1, "lastName": 1, "username": 1}):
                uid = str(u["_id"])
                uname = u.get("name") or u.get("fullName") or " ".join(filter(None, [u.get("firstName"), u.get("lastName")])) or u.get("username")
                if uname:
                    users_map[uid] = uname

            for uid, uname in users_map.items():
                pattern = re.compile(rf'(?:(?:the\s+)?(?:user\s+(?:with\s+)?(?:ID|id|Id):?\s*)|(?:(?:ID|id|Id):?\s*))?{uid}\b', re.IGNORECASE)
                text = pattern.sub(uname, text)
        except Exception:
            pass

    # Clean up phrases like "user with ID <hex>", "user ID <hex>"
    text = re.compile(r'(?:the\s+)?user\s+(?:with\s+)?(?:ID|id|Id):?\s*[0-9a-fA-F]{24}\b', re.IGNORECASE).sub('a team member', text)
    text = re.compile(r'(?:the\s+)?user\s+ID:?\s*[0-9a-fA-F]{24}\b', re.IGNORECASE).sub('a team member', text)
    text = re.compile(r'(?:(?:ID|id|Id):?\s*)?[0-9a-fA-F]{24}\b').sub('', text)
    # Clean up formatting artifacts like (, , ...) or dangling punctuation
    text = re.sub(r'\(\s*,\s*', '(', text)
    text = re.sub(r',\s*,+', ', ', text)
    text = re.sub(r'\(\s*,+\s*', '(', text)
    text = re.sub(r',\s*\)', ')', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


def _clean_label(value, fallback: str) -> str:
    text = " ".join(re.sub(r"[\u200b\u200c\u200d\ufeff]", "", str(value or "")).split())
    return text or fallback


def _leave_type_from_question(question: str) -> str | None:
    """Extract the leave type (Full Leave / Short Leave / Halfday Leave) from a question."""
    text = " ".join((question or "").split())
    m = _LEAVE_TYPE_PATTERN.search(text)
    if not m:
        return None
    label = m.group(0).strip()
    norm = label.casefold().replace("-", " ").replace("_", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm.startswith("full"):
        return "Full Leave"
    if norm.startswith("short"):
        return "Short Leave"
    if norm.startswith("halfday") or norm.startswith("half day"):
        return "Halfday Leave"
    return label


def _leave_time_empty(person_name: str | None, leave_type: str | None, note: str):
    scope = []
    if person_name:
        scope.append(person_name)
    if leave_type:
        scope.append(f"'{leave_type}' tasks")
    conditions = " and ".join(scope) if scope else "leave tasks"
    message = f"No logged-time entries were found for {conditions}."
    return ExactResult(
        collection="cards", total=0, records=[], conditions=conditions, complete=True,
        source="planner", operation="list", notes=[note], direct_answer=message,
    )


def _leave_board_ids(db) -> set:
    """Return ids for the dedicated Leave project(s), not every card with 'leave' in a title."""
    if "boards" not in set(db.list_collection_names()):
        return set()
    ids = set()
    for board in db["boards"].find({}, {"name": 1, "project": 1}):
        label = _norm(board.get("name") or board.get("project") or "")
        if (label in {"leave", "leaves"}
                or label.startswith("leave ")
                or label.endswith(" leave")
                or label.endswith(" leaves")):
            ids.add(str(board["_id"]))
    return ids


def _leave_person_candidate(text: str) -> str | None:
    """Person for leave questions ("leaves of mohit in sep", "mohit's leaves", "Anil Bansal leave requests")."""
    patterns = (
        r"\bleaves?\s+(?:of|for|by|taken\s+by|take\s+by)\s+((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)",
        r"\b((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)['’]s\s+leaves?\b",
        r"\bleaves?\s+did\s+((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)\s+take\b",
        r"\bdid\s+((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)\s+take\s+(?:any\s+|a\s+)?leaves?\b",
        r"\b((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)\s+leaves?\s+(?:in|during|for)\b",
        r"^((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)\s+(?:leave\s+(?:requests?|status|history|applications?)|leaves?)\b",
        r"\bleave\s+(?:requests?|status|history|applications?)\s+(?:of|for|by)\s+((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)",
        r"\b((?:[A-Za-z][A-Za-z'-]+\s+){0,2}[A-Za-z][A-Za-z'-]+)\s+(?:has|had|take|took|requested|applied\s+for)\s+(?:any\s+|a\s+)?leaves?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        words = [word for word in re.findall(r"[A-Za-z][A-Za-z'-]*", match.group(1))
                 if word.lower() not in _LEAVE_NAME_STOPS]
        while words:
            candidate = " ".join(words)
            if _ids_by_name("users", candidate):
                return candidate
            words.pop()      # drop the trailing word: "mohit in sep" -> "mohit"
    return None


def _leave_requests_answer(question: str, current_user_id: str = None, history: list = None,
                           user_name: str = None, workspace_id=None, is_admin: bool = False, **kwargs):
    """Query employee leave requests from the 'leaverequests' collection.

    Resolves requesters via the 'users' collection and leave types via the 'leavetypes' collection.
    Never reads from or confuses with the administrative 'Leave' board or task cards.
    """
    if isinstance(current_user_id, list):
        history, current_user_id = current_user_id, kwargs.get("current_user_id")
    if not current_user_id and "user_id" in kwargs:
        current_user_id = kwargs["user_id"]
    if isinstance(history, str) and not current_user_id:
        current_user_id, history = history, None
    if not isinstance(history, list):
        history = None

    text = " ".join((question or "").split())
    has_leave_kw = bool(_LEAVE_TIME_PATTERN.search(text) or _LEAVE_HISTORY_PATTERN.search(text)
                        or _LEAVE_REQUEST_INTENT.search(text) or re.search(r"\bleaves?\b", text, re.IGNORECASE))
    if not has_leave_kw:
        return None
    # Exclude questions explicitly about the Leave board / project cards
    if re.search(r"\b(?:leave\s+(?:board|project)|project\s+called\s+leave)\b", text, re.IGNORECASE):
        return None

    db = mongo_service.get_db()
    existing = set(db.list_collection_names())
    if "leaverequests" not in existing:
        return None

    started = time.perf_counter()
    is_personal = bool(re.search(r"\b(?:my|mine|myself)\b", text, re.IGNORECASE))

    # 1. Person resolution
    target_ids = None
    target_person_name = None
    if is_personal:
        if not current_user_id:
            return ExactResult(
                collection="leaverequests", total=0, records=[], conditions="unverified account",
                complete=True, source="auth", operation="list",
                direct_answer="Please sign in to view your leave requests.", timings={"db": 0}
            )
        target_ids = _id_variants([current_user_id])
        target_person_name = user_name or "Your"
    else:
        person_candidate = _leave_person_candidate(text)
        if person_candidate and person_candidate.lower() in _LEAVE_NAME_STOPS:
            person_candidate = None
        if not person_candidate and history:
            for turn in reversed(history):
                if isinstance(turn, dict):
                    prior_q = turn.get("question", "")
                    if cand := _leave_person_candidate(prior_q):
                        if cand.lower() not in _LEAVE_NAME_STOPS:
                            person_candidate = cand
                            break
        if person_candidate:
            user_ids = _ids_by_name("users", person_candidate)
            if user_ids:
                target_ids = _id_variants(user_ids)
                u_doc = db["users"].find_one({"_id": {"$in": target_ids}}, {"name": 1, "fullName": 1})
                target_person_name = (u_doc or {}).get("name") or (u_doc or {}).get("fullName") or person_candidate

        # Privacy check: non-admins cannot query other users' leaves
        if target_ids and not is_admin and current_user_id:
            curr_variants = set(str(v) for v in _id_variants([current_user_id]))
            if not any(str(tid) in curr_variants for tid in target_ids):
                return ExactResult(
                    collection="leaverequests", total=0, records=[], conditions="unauthorized",
                    complete=True, source="auth", operation="list",
                    direct_answer="You are not authorized to view leave records of other team members.",
                    timings={"db": 0}
                )

    is_who_on_leave = bool(re.search(r"\bwho\s+(?:is|was|are|were)\s+on\s+leave\b|\bis\s+.*?\s+on\s+leave\b|\bon\s+leave\b", text, re.IGNORECASE))
    is_superlative = bool(_LEAVE_COUNT_SUPERLATIVE_PATTERN.search(text))

    # If non-admin and no target specified, default to own records (unless asking org-wide who is on leave or superlative)
    if not is_admin and not target_ids and current_user_id and not is_who_on_leave and not is_superlative:
        target_ids = _id_variants([current_user_id])
        target_person_name = user_name or "Your"

    # 2. Build MongoDB query for leaverequests
    clauses = []
    if target_ids:
        clauses.append({"requester": {"$in": target_ids}})

    # Status filter
    status_label = None
    if is_who_on_leave:
        clauses.append({"status": re.compile(r"^approved$", re.IGNORECASE)})
        status_label = "approved"
    elif re.search(r"\b(?:pending|waiting|awaiting|in review)\b", text, re.IGNORECASE):
        clauses.append({"status": re.compile(r"^pending$", re.IGNORECASE)})
        status_label = "pending"
    elif re.search(r"\b(?:approved|accepted)\b", text, re.IGNORECASE):
        clauses.append({"status": re.compile(r"^approved$", re.IGNORECASE)})
        status_label = "approved"
    elif re.search(r"\b(?:rejected|declined|denied)\b", text, re.IGNORECASE):
        clauses.append({"status": re.compile(r"^rejected$", re.IGNORECASE)})
        status_label = "rejected"
    elif re.search(r"\b(?:cancelled|canceled)\b", text, re.IGNORECASE):
        clauses.append({"status": re.compile(r"^(cancelled|canceled)", re.IGNORECASE)})
        status_label = "cancelled"

    # Specific leave type filter
    lt_label = _leave_type_from_question(text)
    if lt_label and "leavetypes" in existing:
        lt_matches = list(db["leavetypes"].find({"name": re.compile(re.escape(lt_label), re.IGNORECASE)}, {"_id": 1}))
        if lt_matches:
            clauses.append({"leaveType": {"$in": [m["_id"] for m in lt_matches]}})

    # Month / period / date filter
    period = _extract_period(text)
    period_label = ""
    if period and "start" in period and "end" in period:
        from datetime import datetime, time as dtime
        p_start = datetime.combine(period["start"].date(), dtime.min)
        p_end = datetime.combine(period["end"].date(), dtime.max)
        clauses.append({
            "$or": [
                {"startDate": {"$gte": p_start, "$lte": p_end}},
                {"endDate": {"$gte": p_start, "$lte": p_end}},
                {"$and": [{"startDate": {"$lte": p_start}}, {"endDate": {"$gte": p_end}}]}
            ]
        })
        if period.get("kind") == "day" and period.get("day"):
            period_label = f" on {period.get('day')} {period['start'].strftime('%b %Y')}"
        elif period.get("month"):
            period_label = f" in {period['start'].strftime('%B %Y')}"

    ws_flt = _workspace_filter("leaverequests", workspace_id, db)
    final_query = _and_merge(ws_flt, *clauses)

    collection = db["leaverequests"]
    try:
        total = _count(collection, final_query)
        docs = list(collection.find(final_query).sort("startDate", -1).limit(settings.STRUCT_MAX_RECORDS))
    except Exception:
        logger.exception("Could not query leaverequests collection")
        return None

    # Resolve requesters and leave types for display
    u_ids = [d.get("requester") for d in docs if d.get("requester")]
    lt_ids = [d.get("leaveType") for d in docs if d.get("leaveType")]

    user_map = {}
    if u_ids and "users" in existing:
        for u in db["users"].find({"_id": {"$in": _id_variants(u_ids)}}, {"name": 1, "fullName": 1, "email": 1}):
            user_map[str(u["_id"])] = u.get("name") or u.get("fullName") or u.get("email")

    ltype_map = {}
    if lt_ids and "leavetypes" in existing:
        for lt in db["leavetypes"].find({"_id": {"$in": _id_variants(lt_ids)}}, {"name": 1, "key": 1}):
            ltype_map[str(lt["_id"])] = lt.get("name") or lt.get("key")

    count_only = bool(re.search(r"\bhow\s+many\b|\bcount\b|\btotal\b", text, re.IGNORECASE))
    is_superlative = bool(_LEAVE_COUNT_SUPERLATIVE_PATTERN.search(text))

    # Superlative handling ("who has least leaves in oct", "who takes most leaves")
    if is_superlative:
        approved_docs = [d for d in docs if str(d.get("status", "")).upper() == "APPROVED"]
        if not approved_docs:
            cancelled_docs = [d for d in docs if "CANCEL" in str(d.get("status", "")).upper()]
            cancelled_note = f" ({len(cancelled_docs)} leave request(s) were cancelled)" if cancelled_docs else ""
            msg = f"No team members have approved leaves recorded{period_label}{cancelled_note}. All team members currently have 0 leaves for this period."
            return ExactResult(
                collection="leaverequests", total=0, records=[],
                conditions=f"approved leaves{period_label}", complete=True,
                source="planner", operation="list",
                direct_answer=msg, timings={"db": round((time.perf_counter() - started) * 1000)}
            )
        # Group approved leaves by user
        user_totals = {}
        for d in approved_docs:
            uid = str(d.get("requester"))
            user_totals[uid] = user_totals.get(uid, 0) + d.get("totalRequestedDayUnits", 1)
        wants_least = bool(re.search(r"\b(?:least|lowest|minimum|min|fewest|fewer)\b", text, re.IGNORECASE))
        sorted_users = sorted(user_totals.items(), key=lambda x: x[1], reverse=not wants_least)
        rank_lines = [f"Leave counts{period_label} ({'least' if wants_least else 'most'} first):", ""]
        for uid, days in sorted_users:
            uname = user_map.get(uid, "Employee")
            rank_lines.append(f"- **{uname}**: {days} day(s)")
        return ExactResult(
            collection="leaverequests", total=len(sorted_users), records=[],
            conditions=f"{'least' if wants_least else 'most'} leaves{period_label}", complete=True,
            source="planner", operation="list",
            direct_answer="\n".join(rank_lines), timings={"db": round((time.perf_counter() - started) * 1000)}
        )

    # Format user-facing answer
    owner_str = f"{target_person_name}'s" if (target_person_name and target_person_name != "Your") else ("Your" if is_personal else "Team")
    filter_str = f" ({status_label})" if status_label else ""
    header = f"{owner_str} leave requests{filter_str}{period_label} ({total} total):"

    lines = [header, ""]
    records = []
    total_days = 0
    for d in docs:
        uname = user_map.get(str(d.get("requester")), target_person_name or "Employee")
        ltname = ltype_map.get(str(d.get("leaveType")), "Leave")
        sdate = d.get("startDate")
        edate = d.get("endDate")
        date_str = ""
        if sdate and edate:
            sd = sdate.strftime("%d %b %Y")
            ed = edate.strftime("%d %b %Y")
            date_str = sd if sd == ed else f"{sd} to {ed}"
        elif sdate:
            date_str = sdate.strftime("%d %b %Y")
        units = d.get("totalRequestedDayUnits", 1)
        total_days += units
        st = str(d.get("status", "PENDING")).replace("_", " ").title()
        reason = str(d.get("reason") or "").strip()
        reason_str = f" — Reason: {reason}" if reason else ""

        detail = f"{uname} | {ltname} ({units} day(s), {date_str}) — Status: `{st}`{reason_str}"
        lines.append(f"- **{uname}** — {ltname} ({units} day(s), {date_str}) — `{st}`{reason_str}")
        records.append({
            "document_id": str(d.get("_id", "")),
            "name": f"{uname} - {ltname}",
            "text": detail,
        })

    if not docs:
        if is_who_on_leave:
            direct_ans = f"No team members are recorded on approved leave{period_label}."
        elif status_label:
            direct_ans = f"There are 0 {status_label} leave requests found{period_label}."
        elif target_person_name:
            direct_ans = f"No leave requests were found for {target_person_name}{period_label}."
        else:
            direct_ans = f"No leave requests were found{period_label}."
    elif is_who_on_leave:
        direct_ans = f"Team members on approved leave{period_label}:\n\n" + "\n".join(
            f"- **{r['name'].split(' - ')[0]}** ({r['text']})" for r in records
        )
    elif count_only:
        status_desc = f" with status '{status_label}'" if status_label else ""
        person_desc = f" for {target_person_name}" if target_person_name else ""
        direct_ans = f"There are {total} leave request(s){status_desc}{person_desc}{period_label} (totaling {total_days} day(s))."
    else:
        direct_ans = "\n".join(lines).strip()

    conds = []
    if target_person_name:
        conds.append(f"employee = {target_person_name}")
    if status_label:
        conds.append(f"status = {status_label}")
    if lt_label:
        conds.append(f"type = {lt_label}")
    if period_label:
        conds.append(period_label.strip())

    return ExactResult(
        collection="leaverequests",
        total=total,
        records=records,
        conditions="; ".join(conds) or "all leave requests",
        complete=total <= len(records),
        source="planner",
        operation="count" if count_only else "list",
        notes=[f"Read {total} leave request(s) directly from 'leaverequests' collection."],
        timings={"db": round((time.perf_counter() - started) * 1000)},
        direct_answer=direct_ans,
    )


# Backward compatibility alias
_user_leave_requests_answer = _leave_requests_answer


def _leave_time_answer(question: str, history: list = None, current_user_id: str = None,
                       is_admin: bool = False, workspace_id: str = None):
    """Route leave queries to the canonical 'leaverequests' collection."""
    text = " ".join((question or "").split())
    # Only route to task loggedTime if the query specifically asks for hours/logged time on leave cards
    if not _LEAVE_TIME_QUESTION_PATTERN.search(text):
        return _leave_requests_answer(question, current_user_id=current_user_id, history=history,
                                     is_admin=is_admin, workspace_id=workspace_id)
    if not (_LEAVE_TYPE_PATTERN.search(text) or re.search(r"\bleaves?\b", text, re.IGNORECASE)):
        return None

    leave_type = _leave_type_from_question(text)
    # When no specific leave type is named, match any card with "leave" in the title.
    title_pattern = re.compile(re.escape(leave_type), re.IGNORECASE) if leave_type else re.compile(r"leave", re.IGNORECASE)
    personal = bool(current_user_id and re.search(r"\b(?:my|mine|myself)\b", text, re.IGNORECASE))
    requested_person = None if personal else (_leave_person_candidate(text)
                                              or _logged_time_person_candidate(text))
    target_date = _extract_date_from_question(text)
    # Month/year/week scoping ("who has highest leaves in august month"): like
    # _logged_time_answer, use the period range so month-only questions do not
    # require an exact day.
    period = _extract_period(text)
    target_date = None if period is not None else target_date

    db = mongo_service.get_db()
    person_name, person_ids = None, []
    if personal:
        person_ids = _id_variants([current_user_id])
        person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")
    elif requested_person:
        person_ids = _id_variants(_ids_by_name("users", requested_person))
        if person_ids:
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), requested_person)

    # Privacy: non-admin accounts may only read their own logged time. A caller
    # without a verified identity (anonymous/guest) is denied outright.
    if not is_admin:
        if not current_user_id:
            return ExactResult(
                collection="cards", total=0, records=[],
                conditions="not authorized: unauthenticated", complete=True,
                source="planner", operation="list",
                notes=["Not authorized: sign in to view leave logged time."],
                direct_answer="Please sign in to view leave logged time.",
            )
        me = str(current_user_id)
        if personal:
            person_ids = _id_variants([current_user_id])
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")
        elif requested_person:
            resolved = {str(value) for value in person_ids}
            if me not in resolved:
                return _leave_time_empty(requested_person, leave_type,
                                         f"Not authorized to view leave logged time for '{requested_person}'.")
            person_ids = _id_variants([current_user_id])
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")
        else:
            person_ids = _id_variants([current_user_id])
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")

    if not person_ids and not is_admin:
        return _leave_time_empty(person_name, leave_type, "No user resolved for the leave logged-time query.")

    existing = set(db.list_collection_names())
    if "cards" not in existing:
        return _leave_time_empty(person_name, leave_type, "The 'cards' collection does not exist.")

    # Find leave task cards: title contains the leave type (e.g. "Full Leave") or any "leave" task.
    card_flt = {"title": title_pattern, "loggedTime.0": {"$exists": True}}
    leave_board_ids = _leave_board_ids(db)
    if leave_board_ids:
        card_flt["board"] = {"$in": _id_variants(list(leave_board_ids))}
    if person_ids:
        card_flt["loggedTime.user"] = {"$in": person_ids}

    projection = {"title": 1, "board": 1, "status": 1, "loggedTime": 1}
    cards = list(db["cards"].find(card_flt, projection).sort("_id", 1))

    boards = {}
    if "boards" in existing:
        for board in db["boards"].find({}, {"name": 1, "project": 1}):
            boards[str(board["_id"])] = _clean_label(board.get("name") or board.get("project"), "Unknown project")

    matched = []
    cards_used = 0
    for card in cards:
        title = _clean_label(strip_html(card.get("title")), "Untitled task")
        board_id = card.get("board")
        project = boards.get(str(board_id), "Unknown project") if board_id is not None else "Unknown project"
        entries = card.get("loggedTime") or []
        if isinstance(entries, dict):
            entries = [entries]
        card_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            user_id = _logged_time_entry_user(entry)
            if person_ids and user_id not in {str(value) for value in person_ids}:
                continue
            d_val = entry.get("date") or entry.get("createdAt")
            if period is not None:
                if not _entry_in_period(d_val, period):
                    continue
            elif target_date:
                t_year, t_month, t_day = target_date
                if not d_val:
                    continue
                e = _coerce_entry_dt(d_val)
                e_year, e_month, e_day = (e.year, e.month, e.day) if e else (None, None, None)
                # A month-only target (day=None) matches every day in that month.
                if e_month != t_month or (t_day is not None and e_day != t_day):
                    continue
                if t_year is not None and e_year is not None and e_year != t_year:
                    continue
            card_entries.append((entry, user_id))
        if not card_entries:
            continue
        cards_used += 1
        for entry, user_id in card_entries:
            hours = _to_number(entry.get("hours")) or 0
            minutes = _to_number(entry.get("minutes")) or 0
            date_value = entry.get("date") or entry.get("createdAt")
            if hasattr(date_value, "isoformat"):
                date_value = date_value.isoformat()
            extras = []
            for key, value in entry.items():
                if key in {"_id", "user", "userId", "user_id", "hours", "minutes", "date",
                           "createdAt", "description"} or is_sensitive(str(key)):
                    continue
                if isinstance(value, bool):
                    extras.append(f"{humanize_key(key)}: {'Yes' if value else 'No'}")
                elif isinstance(value, (str, int, float)) and str(value).strip():
                    extras.append(f"{humanize_key(key)}: {' '.join(str(value).split())}")
            matched.append({
                "document_id": str(card.get("_id", "")),
                "title": title,
                "project": project,
                "status": str(card.get("status") or "").strip(),
                "person_id": user_id,
                "person": None,
                "minutes": round(hours * 60 + minutes),
                "date": date_value,
                "description": strip_html(entry.get("description")) if entry.get("description") else None,
                "extras": extras,
            })

    if not matched:
        date_suffix = ""
        if period is not None:
            date_suffix = f" in {_period_label(period)}" if _period_label(period) else ""
        elif target_date:
            date_suffix = f" on {target_date[0]}-{target_date[1]:02d}" + (f"-{target_date[2]:02d}" if target_date[2] else "")
        scope = (f"'{person_name}'" if person_name else "")
        if leave_type:
            scope = f"{scope} in '{leave_type}' tasks".strip()
        conditions = f"loggedTime.user = {person_name}" if person_name else f"leave type = {leave_type or 'leave'}"
        return _leave_time_empty(person_name, leave_type,
                                 f"No nested loggedTime entries matched in 'cards' for {scope}{date_suffix}.")

    user_names = _logged_time_user_names(db, {item["person_id"] for item in matched if item["person_id"]})
    for item in matched:
        item["person"] = user_names.get(item["person_id"], person_name or "Unidentified contributor")

    matched.sort(key=lambda item: (str(item["person"]).casefold(), str(item["project"]).casefold(),
                                   str(item["title"]).casefold(), str(item["date"] or ""), item["document_id"]))
    total_minutes = sum(item["minutes"] for item in matched)

    # --- Superlative handling: "who takes most/least leaves" / "who has most/least logged time in leave" ---
    _sup_min_pat = re.compile(
        r"\b(?:least|lowest|minimum|min|less|fewer|fewest|smallest|least\s+frequently|"
        r"sabse\s*(?:thoda|kam|chhota|chota|nyuntam)|"
        r"sase\s*(?:thoda|kam|chhota|chota|nyuntam)|"
        r"kam\s*se\s*kam|thoda|kam)\b",
        re.IGNORECASE,
    )
    _sup_max_pat = re.compile(
        r"\b(?:most|highest|maximum|max|more|largest|biggest|longest|most\s+frequently|"
        r"sabse\s*(?:jayda|jyada|zyada|jada|bada|adhiktam)|"
        r"sase\s*(?:jayda|jyada|zyada|jada|bada|adhiktam)|"
        r"jyada\s*se\s*jyada|jayda\s*se\s*jayda|jayda|jyada|zyada)\b",
        re.IGNORECASE,
    )
    is_sup = is_superlative or bool(_sup_min_pat.search(text) or _sup_max_pat.search(text))
    if is_sup and not person_name and not personal:
        extremum = "least" if _sup_min_pat.search(text) else "most" if _sup_max_pat.search(text) else "most"
        wants_time = bool(re.search(r"\b(?:logged[\s_-]*time|time|hours?|ghante|ghanta|duration)\b", text, re.IGNORECASE))
        per_person_entries: dict = {}
        for item in matched:
            per_person_entries.setdefault(item["person"], []).append(item)

        candidate_pool = {p: entries for p, entries in per_person_entries.items() if p != "Unidentified contributor"}
        if not candidate_pool:
            candidate_pool = per_person_entries

        if not candidate_pool:
            return _leave_time_empty(None, leave_type, "No leave entries found.")

        leave_label = leave_type or "leave"
        if wants_time:
            metrics = {p: sum(e["minutes"] for e in entries) for p, entries in candidate_pool.items()}
            target_val = min(metrics.values()) if extremum == "least" else max(metrics.values())
            winners = [p for p, m in metrics.items() if m == target_val]
            qualifier = "least" if extremum == "least" else "most"
            if len(winners) == 1:
                entry_count = len(candidate_pool[winners[0]])
                entry_word = "entry" if entry_count == 1 else "entries"
                answer = (
                    f"{winners[0]} has logged the {qualifier} time in {leave_label} with "
                    f"{_format_duration(target_val)} across {entry_count} {entry_word}."
                )
            else:
                names_str = ", ".join(winners[:-1]) + f" and {winners[-1]}"
                answer = (
                    f"{names_str} are tied with the {qualifier} logged time in {leave_label}: "
                    f"{_format_duration(target_val)} each."
                )
        else:
            metrics = {p: len(entries) for p, entries in candidate_pool.items()}
            target_val = min(metrics.values()) if extremum == "least" else max(metrics.values())
            winners = [p for p, c in metrics.items() if c == target_val]
            qualifier = "fewest" if extremum == "least" else "most"
            entry_word = "entry" if target_val == 1 else "entries"
            if len(winners) == 1:
                answer = (
                    f"{winners[0]} takes the {qualifier} {leave_label} with "
                    f"{target_val} {entry_word} recorded."
                )
            else:
                names_str = ", ".join(winners[:-1]) + f" and {winners[-1]}"
                answer = (
                    f"{names_str} are tied with the {qualifier} {leave_label}: "
                    f"{target_val} {entry_word} each."
                )

        lines = [answer, ""]
        for p in winners:
            person_minutes = sum(e["minutes"] for e in candidate_pool[p])
            count = len(candidate_pool[p])
            e_noun = "entry" if count == 1 else "entries"
            lines.append(f"{p}: {count} {e_noun} ({_format_duration(person_minutes)})")
        records = [{"document_id": "", "name": leave_label, "text": answer}]
        conditions = f"superlative={extremum}; leave type={leave_label}"
        return ExactResult(
            collection="cards", total=len(matched), records=records, conditions=conditions,
            complete=True, source="planner", operation="list",
            notes=[f"Leave-count superlative ({extremum}): {target_val}; {len(winners)} winner(s)."],
            direct_answer="\n".join(lines).strip(),
            timings={"db": 0},
        )
    # --- End superlative handling ---

    groups = {}
    for item in matched:
        person_group = groups.setdefault(item["person"], {})
        project_group = person_group.setdefault(item["project"], {})
        project_group.setdefault((item["title"], item["status"], item["document_id"]), []).append(item)

    heading = f"All {leave_type or 'leave'} logged-time entries"
    if person_name:
        heading += f" for {person_name}"
    heading += ":"
    lines = [heading,
             f"Total: {_format_duration(total_minutes)} across {len(matched)} entries on {cards_used} tasks.",
             ""]
    for person, person_projects in groups.items():
        person_minutes = sum(item["minutes"] for project in person_projects.values()
                             for entries in project.values() for item in entries)
        if len(groups) > 1:
            lines.append(f"{person} — {_format_duration(person_minutes)}")
        for project, project_cards in person_projects.items():
            lines.append(f"Project: {project}")
            for (title, status, _doc_id), entries in project_cards.items():
                suffix = f" [{status}]" if status else ""
                lines.append(f"- {title}{suffix}")
                for item in entries:
                    detail = f"  - {item['date'] + ': ' if item['date'] else ''}{_format_duration(item['minutes'])}"
                    if item["description"]:
                        detail += f" — {item['description']}"
                    if item["extras"]:
                        detail += "; " + "; ".join(item["extras"])
                    lines.append(detail)
            lines.append("")
        if len(groups) > 1:
            lines.append("")

    records = []
    record_text = {}
    for item in matched:
        key = (item["document_id"], item["title"], item["project"], item["status"])
        detail = f"{item['date'] + ': ' if item['date'] else ''}{_format_duration(item['minutes'])}"
        if item["description"]:
            detail += f" — {item['description']}"
        if item["extras"]:
            detail += "; " + "; ".join(item["extras"])
        record_text.setdefault(key, []).append(detail)
    for (document_id, title, project, status), details in record_text.items():
        prefix = f"Project: {project}; Task: {title}" + (f"; Status: {status}" if status else "")
        records.append({"document_id": document_id, "name": title,
                        "text": prefix + "; Entries: " + "; ".join(details)})

    conditions = f"leave type = {leave_type or 'leave'}" + (f"; user = {person_name}" if person_name else "")
    return ExactResult(
        collection="cards", total=len(matched), records=records, conditions=conditions,
        complete=True, source="planner", operation="list",
        notes=[f"Read {len(matched)} {leave_type} logged-time entries from {cards_used} cards; the list is complete."],
        direct_answer="\n".join(lines).strip(),
        timings={"db": 0},
    )

def _logged_time_user_names(db, ids) -> dict:
    names = {}
    if not ids:
        return names
    try:
        users = db["users"].find({"_id": {"$in": _id_variants(ids)}},
                                 {"name": 1, "firstName": 1, "lastName": 1})
    except Exception:
        return names
    for user in users:
        name = user.get("name") or " ".join(filter(None, (user.get("firstName"), user.get("lastName"))))
        if name:
            names[str(user["_id"])] = name
    return names


def _logged_time_empty(conditions: str, note: str):
    message = f"No logged-time entries were found for {conditions}."
    return ExactResult(
        collection="cards", total=0, records=[], conditions=conditions, complete=True,
        source="planner", operation="list", notes=[note], direct_answer=message,
    )


_ASSIGNEE_TERM = re.compile(
    r"\b(?:assignees?|employee[s]?|person[s]?|peoples?|assigned\s+(?:to|people|persons|users|employees|members|team))\b",
    re.IGNORECASE)

_ASSIGNEE_NOUN_MAP = {
    "assignee": "assignee", "assignees": "assignee",
    "employee": "employee", "employees": "employee",
    "person": "person", "persons": "person",
    "people": "people", "peoples": "people",
}
_ASSIGNEE_COUNT_REQUEST = re.compile(
    r"\b(?:how\s+many|count|number\s+of|total)\b", re.IGNORECASE)


def is_assignee_count_question(question: str) -> bool:
    text = question or ""
    return bool(_ASSIGNEE_TERM.search(text) and _ASSIGNEE_COUNT_REQUEST.search(text))


def _assignee_target(question: str):
    """Extract the task/project name and its type from an assignee-count question."""
    text = " ".join((question or "").split()).strip(" ?.!")
    type_hint = ("project" if re.search(r"\b(?:projects?|boards?)\b", text, re.IGNORECASE)
                 else "task" if re.search(r"\b(?:tasks?|cards?|tickets?|work\s+items?)\b", text, re.IGNORECASE)
                 else None)
    patterns = (
        (r"\b(?:in|on|for|of)\s+(?:the\s+)?(?:project|board)\s+(.+)$", "project"),
        (r"\b(?:in|on|for|of)\s+(?:the\s+)?(?:task|card|ticket|work\s+item)\s+(.+)$", "task"),
        (r"\b(?:project|board)\s+(.+?)\s+(?:has|have|contains?|with)\b", "project"),
        (r"\b(?:task|card|ticket|work\s+item)\s+(.+?)\s+(?:has|have|contains?|with)\b", "task"),
        (r"\b(?:does|do|did)\s+(.+?)\s+(?:have|contain)\b", None),
        (r"\bassigned\s+to\s+(.+?)(?=\s+(?:task|card|ticket|work\s+item|project|board)\b|[?.!,]|$)", None),
        (r"\b(?:in|on|for|of)\s+['\"](.+?)['\"]", None),
        (r"\b(?:in|on|for|of)\s+(.+)$", None),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip()
        name = re.sub(r"^(?:the\s+)?(?:project|board|task|card|ticket|work\s+item)\s+", "",
                      name, flags=re.IGNORECASE)
        name = re.sub(r"\s+(?:project|board|task|card|ticket|work\s+item)$", "",
                      name, flags=re.IGNORECASE)
        name = name.strip(" '\"?.!,")
        if name and _norm(name) not in {
                "all", "assignee", "assignees", "people", "peoples", "persons", "users", "employees",
                "members", "team", "task", "tasks", "project", "projects"}:
            return name, kind or type_hint
    return None, type_hint


def _assignee_identity(value):
    if isinstance(value, dict):
        for key in ("_id", "id", "user", "userId", "user_id", "email",
                    "name", "displayName", "fullName"):
            item = value.get(key)
            if isinstance(item, dict):
                item = item.get("_id") or item.get("id") or item.get("userId") or item.get("user")
            if item is not None and not isinstance(item, (dict, list, tuple, set)):
                return str(item)
        return None
    if value is not None and not isinstance(value, (list, tuple, set)):
        text = str(value).strip()
        return text or None
    return None


def _assignee_ids(doc: dict) -> set:
    raw = doc.get("assignees")
    if isinstance(raw, dict):
        raw = [raw]
    elif not isinstance(raw, (list, tuple, set)):
        raw = [raw] if raw else []
    return {ref for ref in (_assignee_identity(item) for item in raw) if ref}


def _assignee_summary(db, docs: list) -> tuple[int, list]:
    """Count unique assignees and resolve their display names from 'users'."""
    people = {}
    for doc in docs:
        raw = doc.get("assignees")
        if isinstance(raw, dict):
            raw = [raw]
        elif not isinstance(raw, (list, tuple, set)):
            raw = [raw] if raw else []
        for item in raw:
            ref = _assignee_identity(item)
            if ref:
                display = None
                if isinstance(item, dict):
                    for key in ("name", "displayName", "fullName", "email"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            display = value.strip()
                            break
                people.setdefault(ref, display)
        for entry in (doc.get("loggedTime") or []):
            if isinstance(entry, dict):
                uid = _logged_time_entry_user(entry)
                if uid:
                    people.setdefault(str(uid), None)
    resolved = _logged_time_user_names(db, people.keys())
    names = []
    for ref, display in people.items():
        name = display or resolved.get(ref)
        if not name and not HEX24.match(str(ref)):
            name = str(ref)
        if name:
            names.append(name)
    return len(people), sorted(set(names), key=str.casefold)


def _assignee_card_match(db, name: str):
    wanted = _norm(name)
    wanted_tokens = wanted.split()
    exact, partial = [], []
    for card in db["cards"].find({"title": {"$exists": True}}, {"title": 1, "assignees": 1, "board": 1, "loggedTime": 1}):
        title = _clean_label(strip_html(card.get("title")), "")
        title_norm = _norm(title)
        if not title_norm:
            continue
        if title_norm == wanted:
            exact.append(card)
        elif wanted_tokens and all(token in title_norm.split() for token in wanted_tokens):
            partial.append(card)
    if exact:
        return exact[0], [exact[0]]
    partial.sort(key=lambda card: (len(str(card.get("title") or "")), str(card.get("_id") or "")))
    return (partial[0], partial) if partial else (None, [])


def _assignee_board_match(db, name: str):
    wanted = _norm(name)
    wanted_tokens = wanted.split()
    exact, partial = [], []
    for board in db["boards"].find({}, {"name": 1, "project": 1, "title": 1, "department": 1, "owner": 1, "members": 1}):
        label = _clean_label(board.get("name") or board.get("project") or board.get("title"), "")
        label_norm = _norm(label)
        if not label_norm:
            continue
        if label_norm == wanted:
            exact.append(board)
        elif wanted_tokens and all(token in label_norm.split() for token in wanted_tokens):
            partial.append(board)
    if exact:
        return exact[0]
    partial.sort(key=lambda board: (len(str(board.get("name") or board.get("project") or "")),
                                    str(board.get("_id") or "")))
    return partial[0] if partial else None


def _extract_team_target(question: str) -> str | None:
    """Extract project or task target from team and assignment queries."""
    clean_q = " ".join((question or "").split()).strip()
    _TEAM_TARGET_PATTERNS = [
        re.compile(r"\bhow\s+large\s+is\s+(?:the\s+)?(?:project\s+|board\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project\s+team|project|board|team))?\??$", re.IGNORECASE),
        re.compile(r"\b(?:working\s+versus\s+assigned|working\s+vs\s+assigned|active\s+vs\s+assigned)\s+(?:in|for|on)\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"\b(?:who\s+(?:is|are)\s+(?:currently\s+)?actively\s+contributing\s+to)\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"\b(?:working\s+on|assigned\s+to|involved\s+in|responsible\s+for|allocated\s+to|handling|part\s+of|active\s+in|contributing\s+to|participating\s+in|engaged\s+in)\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"\b(?:team\s+size\s+(?:of|for)|team\s+members\s+(?:of|in)|assignees\s+(?:of|in)|members\s+(?:of|in|assigned\s+to)|contributors\s+(?:of|in)|participants\s+(?:of|in))\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"\b(?:how\s+many|count|number\s+of|total\s+number\s+of|show\s+(?:the\s+)?total\s+(?:number\s+of)?|tell\s+me\s+(?:the\s+)?(?:current\s+|assigned\s+)?team\s+size\s+(?:of|for))\s*(?:people|team\s+members|employees|users|members|resources|assignees)?\s*(?:are\s+)?(?:currently\s+)?(?:working\s+on|assigned\s+to|involved\s+in|allocated\s+to|participating\s+in|linked\s+to|contributing\s+to|engaged\s+in|in\s+the|in|for)?\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"\b(?:list\s+all|list\s+the|list|show\s+(?:the\s+)?names\s+of|show\s+all|show\s+everyone|show\s+assigned|show\s+members|give\s+me\s+(?:the\s+)?list\s+of|which\s+employees|which\s+users|which\s+members|which\s+assigned\s+members)\s+(?:team\s+members|members|employees|people|users|participants|contributors|assignees)?\s*(?:working\s+on|assigned\s+to|involved\s+in|allocated\s+to|contributing\s+to|of|in)\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"\b(?:anyone|anybody|someone)\s+working\s+on\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?(?=[?.!,]|$)", re.IGNORECASE),
        re.compile(r"^(?:list\s+|show\s+)?['\"]?([^'\"?.,!]+?)['\"]?\s+(?:contributors|participants|team\s+size|team\s+members|project\s+team)\s*\??$", re.IGNORECASE),
        re.compile(r"\b(?:in|for|on)\s+(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)?['\"]?([^'\"?.,!]+?)['\"]?(?:\s+(?:project|board|task|card))?\b(?=[?.!,]|$)", re.IGNORECASE),
    ]
    _GENERIC_TARGETS = {
        "all", "every", "project", "board", "task", "card", "this", "that", "this project",
        "that project", "this card", "this task", "the project", "the task", "assigned tasks",
        "assigned task", "these tasks", "those tasks", "work"
    }
    for pat in _TEAM_TARGET_PATTERNS:
        m = pat.search(clean_q)
        if m:
            target = m.group(1).strip(" '\"?.!,")
            target = re.sub(r"^(?:the\s+)?(?:project\s+|board\s+|task\s+|card\s+)", "", target, flags=re.IGNORECASE)
            target = re.sub(r"\s+(?:project|board|task|card)$", "", target, flags=re.IGNORECASE)
            target = re.sub(r"^(?:the|this|that)\s+", "", target, flags=re.IGNORECASE).strip(" '\"?.!,")
            if target and _norm(target) not in _GENERIC_TARGETS:
                return target
    return None


def _resolve_team_target_from_history(history, db):
    """If the question refers to 'this project' or 'this card', find the referenced item from recent turns."""
    if not history:
        return None, None
    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        for text in [turn.get("question", ""), turn.get("answer", "")]:
            if not text:
                continue
            b = _assignee_board_match(db, text)
            if b:
                return b, "project"
            c, _ = _assignee_card_match(db, text)
            if c:
                return c, "task"
    return None, None


def _project_or_task_team_answer(question: str, history: list = None, workspace_id=None):
    """Answer team size, assignee, active vs assigned, and member queries for projects and tasks.

    For projects (boards), it searches: board (project) -> department (id) -> user (id)
    along with board owner, direct board members, and card assignees/logged-time contributors.
    """
    _TEAM_INTENT_PATTERN = re.compile(
        r"\b(?:"
        r"how\s+many\s+(?:people|team\s+members|employees|users|members|resources|assignees)|"
        r"who\s+(?:is|are)\s+(?:currently\s+)?(?:working|assigned|involved|responsible|allocated|handling|part\s+of|active|contributing|(?:the\s+)?(?:team\s+members|members)\s+of)|"
        r"who\s+has\s+been\s+allocated|who\s+is\s+part\s+of|who\s+is\s+handling|who\s+are\s+(?:the\s+)?team\s+members|"
        r"team\s+size|project\s+team|how\s+large\s+is|"
        r"list\s+(?:all\s+)?(?:the\s+)?(?:members|assignees|employees|people|participants|contributors)|"
        r"show\s+(?:all\s+|the\s+)?(?:names\s+of\s+)?(?:members|employees|people|users|assignees|contributors|project\s+participants)|"
        r"which\s+(?:employees|users|members|assignees)|"
        r"give\s+me\s+(?:the\s+)?list\s+of\s+(?:people|members|employees|users)|"
        r"count\s+(?:the\s+)?(?:team\s+members|project\s+members|users|assignees|members)|"
        r"actively\s+(?:working|contributing)|active\s+(?:in|members|contributors)|"
        r"haven['’]t\s+started|havent\s+started|not\s+started|not\s+logged\s+any\s+time|no\s+logged\s+time|without\s+logged\s+time|pending\s+tasks|"
        r"completed\s+(?:their\s+)?tasks|"
        r"working\s+versus\s+assigned|working\s+vs\s+assigned|active\s+vs\s+assigned|"
        r"anyone\s+working\s+on|total\s+assignees|"
        r"people\s+working\s+on|members\s+assigned\s+to|people\s+assigned\s+to|"
        r"people\s+involved\s+in|how\s+many\s+resources\s+are\s+allocated|"
        r"number\s+of\s+project\s+assignees|show\s+everyone\s+contributing|"
        r"show\s+project\s+participants|how\s+many\s+users\s+are\s+involved|"
        r"contributors\b|participants\b|project\s+assignments\b|"
        r"team\s+members\s+of\b"
        r")\b",
        re.IGNORECASE,
    )
    text = question or ""
    if not _TEAM_INTENT_PATTERN.search(text) and not is_assignee_count_question(text):
        return None

    db = mongo_service.get_db()
    existing = set(db.list_collection_names())
    if "boards" not in existing and "cards" not in existing:
        return None

    target_str = _extract_team_target(text)
    if not target_str:
        target_str, _ = _assignee_target(text)

    card, board = None, None
    kind = "project" if re.search(r"\b(?:projects?|boards?)\b", text, re.IGNORECASE) else (
           "task" if re.search(r"\b(?:tasks?|cards?|tickets?)\b", text, re.IGNORECASE) else None)

    if target_str:
        if kind == "task":
            card, matches = _assignee_card_match(db, target_str)
        elif kind == "project":
            board = _assignee_board_match(db, target_str)
        else:
            board = _assignee_board_match(db, target_str)
            if not board:
                card, matches = _assignee_card_match(db, target_str)
    else:
        obj, h_kind = _resolve_team_target_from_history(history, db)
        if h_kind == "project":
            board = obj
        elif h_kind == "task":
            card = obj

    if not board and not card:
        return None

    started = time.perf_counter()

    # -------------------------------------------------------------
    # CASE A: TARGET IS A CARD (TASK)
    # -------------------------------------------------------------
    if card is not None:
        card = db["cards"].find_one({"_id": card["_id"]}) or card
        c_title = _clean_label(strip_html(card.get("title")), "Task")
        b_doc = db["boards"].find_one({"_id": card.get("board")}) if "boards" in existing and card.get("board") else None
        b_name = _clean_label(b_doc.get("name") if b_doc else "Unknown project", "Project")
        status = str(card.get("status") or "open").strip()
        assignee_uids = [str(a) for a in (card.get("assignees") or []) if a]
        logged_uids = {}
        for entry in (card.get("loggedTime") or []):
            if isinstance(entry, dict) and entry.get("user"):
                u = str(entry.get("user"))
                m = round((_to_number(entry.get("hours")) or 0) * 60 + (_to_number(entry.get("minutes")) or 0))
                logged_uids[u] = logged_uids.get(u, 0) + m
        all_u = set(assignee_uids) | set(logged_uids.keys())
        u_names = _logged_time_user_names(db, all_u)
        assignee_names = [u_names[u] for u in assignee_uids if u in u_names]
        count = len(assignee_names)
        noun = "person" if count == 1 else "people"

        lines = [f"Task '{c_title}' (Project: {b_name}, Status: `{status}`):"]
        if count == 0:
            lines.append("No team members are currently assigned to this task.")
        else:
            lines.append(f"{count} {noun} assigned: " + ", ".join(assignee_names))
        if logged_uids:
            lines.append("")
            lines.append("Logged time on this task:")
            for u in sorted(logged_uids.keys(), key=lambda x: logged_uids[x], reverse=True):
                uname = u_names.get(u, "Team member")
                lines.append(f"- **{uname}**: {_format_duration(logged_uids[u])}")

        direct_answer = "\n".join(lines).strip()
        records = [{
            "document_id": str(card.get("_id", "")),
            "name": c_title,
            "text": f"Task: {c_title}; Project: {b_name}; Status: {status}; Assignees: {', '.join(assignee_names) or count}"
        }]
        return ExactResult(
            collection="cards", total=1, records=records,
            conditions=f"task = {c_title}", complete=True, source="planner",
            operation="lookup" if re.search(r"\bwho\b", text, re.IGNORECASE) else "count",
            direct_answer=direct_answer, timings={"db": round((time.perf_counter() - started) * 1000)}
        )

    # -------------------------------------------------------------
    # CASE B: TARGET IS A BOARD (PROJECT)
    # Flow: board (project) -> department (id) -> user (id)
    # Also includes board owner, board members, card assignees, and active contributors.
    # -------------------------------------------------------------
    board = db["boards"].find_one({"_id": board["_id"]}) or board
    b_id = board["_id"]
    b_name = _clean_label(board.get("name") or board.get("project"), "Project")

    # 1. Department linkage: board.department -> department -> users
    dept_id = board.get("department")
    dept_name = None
    dept_uids = set()
    if dept_id and "departments" in existing:
        dept = db["departments"].find_one({"_id": dept_id})
        if not dept and ObjectId.is_valid(str(dept_id)):
            try:
                dept = db["departments"].find_one({"_id": ObjectId(str(dept_id))})
            except Exception:
                pass
        if dept:
            dept_name = dept.get("name")
            for m in (dept.get("members") or []):
                if m: dept_uids.add(str(m))
            for m in (dept.get("managers") or []):
                if m: dept_uids.add(str(m))
        d_oid = ObjectId(str(dept_id)) if ObjectId.is_valid(str(dept_id)) else dept_id
        if "users" in existing:
            for u in db["users"].find({"$or": [{"department": d_oid}, {"department": str(dept_id)}, {"department": dept_id}]}, {"_id": 1}):
                dept_uids.add(str(u["_id"]))

    owner_id = str(board["owner"]) if board.get("owner") else None
    board_member_ids = {str(m) for m in (board.get("members") or []) if m}

    # 2. Cards under board
    b_vars = _id_variants([b_id])
    cards = list(db["cards"].find({"board": {"$in": b_vars}}, {"title": 1, "status": 1, "assignees": 1, "loggedTime": 1, "isArchived": 1})) if "cards" in existing else []

    card_assignee_ids = set()
    user_logged_mins = {}
    user_assigned_cards = {}

    for c in cards:
        c_title = _clean_label(strip_html(c.get("title")), "Untitled task")
        c_status = str(c.get("status") or "to-do").strip().lower()
        c_assignees = [str(a) for a in (c.get("assignees") or []) if a]
        card_assignee_ids.update(c_assignees)
        for a in c_assignees:
            user_assigned_cards.setdefault(a, []).append((c_title, c_status))
        for entry in (c.get("loggedTime") or []):
            if isinstance(entry, dict) and entry.get("user"):
                uid = str(entry.get("user"))
                m = round((_to_number(entry.get("hours")) or 0) * 60 + (_to_number(entry.get("minutes")) or 0))
                user_logged_mins[uid] = user_logged_mins.get(uid, 0) + m

    all_uids = dept_uids | card_assignee_ids | board_member_ids
    if owner_id:
        all_uids.add(owner_id)
    u_names = _logged_time_user_names(db, all_uids)

    active_uids = {uid for uid in all_uids if user_logged_mins.get(uid, 0) > 0}
    for uid, c_list in user_assigned_cards.items():
        if any(st in {"in-progress", "inprogress", "doing", "active"} for _, st in c_list):
            active_uids.add(uid)

    inactive_assigned_uids = card_assignee_ids - active_uids
    completed_uids = {uid for uid, c_list in user_assigned_cards.items() if c_list and all(st in {"done", "completed", "closed"} for _, st in c_list)}
    pending_uids = {uid for uid, c_list in user_assigned_cards.items() if any(st not in {"done", "completed", "closed"} for _, st in c_list)}

    q_lower = text.lower()

    # Sub-intent: Haven't started / No logged time
    if re.search(r"\b(?:haven['’]t\s+started|havent\s+started|not\s+started|not\s+logged\s+any\s+time|no\s+logged\s+time|without\s+logged\s+time)\b", q_lower):
        names_inactive = [u_names[u] for u in inactive_assigned_uids if u in u_names]
        if not names_inactive:
            direct_answer = f"All assigned members in project '{b_name}' have started working or logged time."
        else:
            lines = [f"Assigned members in project '{b_name}' who have not logged any time ({len(names_inactive)}):"]
            for u in sorted(names_inactive, key=str.casefold):
                lines.append(f"- **{u}**")
            direct_answer = "\n".join(lines)

    # Sub-intent: Completed their tasks
    elif re.search(r"\bcompleted\s+(?:their\s+)?tasks?\b", q_lower):
        names_comp = [u_names[u] for u in completed_uids if u in u_names]
        if not names_comp:
            direct_answer = f"No assigned members have completed all their tasks in project '{b_name}' yet ({len(pending_uids)} have pending tasks)."
        else:
            lines = [f"Assigned members who have completed all their tasks in project '{b_name}' ({len(names_comp)}):"]
            for u in sorted(names_comp, key=str.casefold):
                lines.append(f"- **{u}**")
            direct_answer = "\n".join(lines)

    # Sub-intent: Pending tasks
    elif re.search(r"\bpending\s+tasks?\b", q_lower):
        names_pend = [u_names[u] for u in pending_uids if u in u_names]
        lines = [f"Assigned members with pending tasks in project '{b_name}' ({len(names_pend)}):"]
        for u in sorted(names_pend, key=str.casefold):
            tasks_str = ", ".join(t[0] for t in user_assigned_cards.get(u, []) if t[1] not in {"done", "completed"})
            lines.append(f"- **{u}**" + (f" (Tasks: {tasks_str})" if tasks_str else ""))
        direct_answer = "\n".join(lines)

    # Sub-intent: Actively working / Currently active / Contributing
    elif re.search(r"\b(?:actively\s+working|active\s+in|currently\s+active|actively\s+contributing|who\s+is\s+currently\s+active)\b", q_lower):
        names_act = [u_names[u] for u in active_uids if u in u_names]
        if not names_act:
            direct_answer = f"No members are currently actively working or have logged time on project '{b_name}'."
        else:
            lines = [f"Members actively working / contributing to project '{b_name}' ({len(names_act)}):"]
            for u in sorted(active_uids, key=lambda x: user_logged_mins.get(x, 0), reverse=True):
                if u in u_names:
                    uname = u_names[u]
                    mins = user_logged_mins.get(u, 0)
                    time_str = f" — {_format_duration(mins)} logged" if mins > 0 else " — currently in-progress"
                    lines.append(f"- **{uname}**{time_str}")
            direct_answer = "\n".join(lines)

    # Sub-intent: Working vs assigned
    elif re.search(r"\b(?:working\s+versus\s+assigned|working\s+vs\s+assigned|active\s+vs\s+assigned)\b", q_lower):
        names_act = [u_names[u] for u in active_uids if u in u_names]
        names_ass = [u_names[u] for u in card_assignee_ids if u in u_names]
        inactive_names = [u_names[u] for u in inactive_assigned_uids if u in u_names]
        lines = [
            f"Project '{b_name}' — Active vs Assigned Members:",
            f"- **Actively Working ({len(names_act)})**: " + (", ".join(sorted(names_act, key=str.casefold)) or "None"),
            f"- **Total Assigned to Tasks ({len(names_ass)})**: " + (", ".join(sorted(names_ass, key=str.casefold)) or "None"),
            f"- **Assigned but Not Started (0 logged time) ({len(inactive_names)})**: " + (", ".join(sorted(inactive_names, key=str.casefold)) or "None"),
        ]
        direct_answer = "\n".join(lines)

    # General / Count / Team size / Names of assigned people / Short queries
    else:
        owner_name = u_names.get(owner_id)
        assigned_names = sorted({u_names[u] for u in card_assignee_ids if u in u_names}, key=str.casefold)
        dept_member_names = sorted({u_names[u] for u in dept_uids if u in u_names}, key=str.casefold)
        all_member_names = sorted(set(u_names.values()), key=str.casefold)

        dept_str = f" (Department: **{dept_name}**)" if dept_name else ""
        lines = [f"Project **{b_name}**{dept_str} team overview ({len(all_member_names)} total linked):"]
        if owner_name:
            lines.append(f"- **Project Owner**: {owner_name}")
        if assigned_names:
            lines.append(f"- **Task Assignees ({len(assigned_names)})**: " + ", ".join(assigned_names))
        if dept_member_names:
            lines.append(f"- **Department Team ({len(dept_member_names)})**: " + ", ".join(dept_member_names))
        active_names = sorted({u_names[u] for u in active_uids if u in u_names}, key=str.casefold)
        if active_names:
            lines.append(f"- **Active Contributors ({len(active_names)})**: " + ", ".join(active_names))
        lines.append(f"- **Tasks / Cards**: {len(cards)} total.")
        direct_answer = "\n".join(lines)

    records = [{
        "document_id": str(b_id or ""),
        "name": b_name,
        "text": f"Project: {b_name}; Department: {dept_name or 'N/A'}; Linked people: {len(all_uids)}; Cards: {len(cards)}"
    }]
    return ExactResult(
        collection="boards", total=len(all_uids), records=records,
        conditions=f"team members in project '{b_name}'", complete=True, source="planner",
        operation="lookup" if re.search(r"\b(?:who|list|show|give)\b", text, re.IGNORECASE) else "count",
        direct_answer=direct_answer, timings={"db": round((time.perf_counter() - started) * 1000)}
    )


# Backward-compatibility alias
_assignee_count_answer = _project_or_task_team_answer


def _mentions_self(text: str, user_id: str) -> bool:
    """True when a logged-time question refers to the signed-in account itself."""
    if not user_id:
        return False
    db = mongo_service.get_db()
    names = _logged_time_user_names(db, _id_variants([user_id]))
    current_user_name = next(iter(names.values()), None) if names else None
    return bool(user_id and authorization_service.mentions_current_user(text, user_id, current_user_name))


def _personal_logged_time_question(question: str) -> bool:
    return bool(re.search(r"\b(?:my|mine|myself)\b", question or "", re.IGNORECASE))


def is_logged_time_followup(question: str, history: list = None) -> bool:
    """True when a calculation/total follow-up continues a prior logged-time question."""
    if not re.search(r"\b(?:calculate|compute|sum|total)\b", question or "", re.IGNORECASE):
        return False
    return any(is_logged_time_question(turn.get("question", "")) for turn in (history or []))


def _logged_time_answer(question: str, history: list = None, current_user_id: str = None,
                        is_admin: bool = False):
    """Complete logged-time answer read only from nested `loggedTime` entries on 'cards'.

    This path never falls back to similarity search and never truncates entries: every matching
    logged-time entry is returned, regardless of STRUCT_MAX_RECORDS or MAX_RECORD_CHARS.
    """
    text = " ".join((question or "").split())
    followup = is_logged_time_followup(text, history)
    if not is_logged_time_question(text) and not followup:
        return None

    candidate_person = _logged_time_person_candidate(text)
    if not candidate_person and not re.search(r"^(?:who|which\s+(?:employee|user|member|person|contributor|staff))\b", text, re.IGNORECASE):
        candidate_person = _employee_name_from_question(text)
    # The broad employee-name fallback can capture "my" from phrases such as
    # "my work on 12 May". Pronouns are not employee names; let the verified
    # account ID select the caller's own entries below.
    if candidate_person and _norm(candidate_person).split()[0] in {"my", "mine", "myself", "me"}:
        candidate_person = None
    # Demonstratives are references to prior conversational context, not employee names.
    # Treating "this" as a person can otherwise leave the query unscoped and expose every
    # logged-time entry to an admin account.
    if candidate_person and _norm(candidate_person) in {"this", "that", "these", "those"}:
        candidate_person = None
    period = _extract_period(text)
    target_date = _extract_date_from_question(text)
    target_date_str = _period_label(period)
    personal = bool(
        current_user_id and
        not candidate_person and
        (_personal_logged_time_question(text) or _mentions_self(text, current_user_id))
    )
    scope_text = text
    requested_person = None if personal else candidate_person
    if (not personal and not requested_person and not _logged_time_project_candidate(text)
            and re.search(r"\b(?:calculate|compute|sum|total)\b", text, re.IGNORECASE)):
        for turn in reversed(history or []):
            prior = turn.get("question", "")
            if not is_logged_time_question(prior):
                continue
            prior_person = _logged_time_person_candidate(prior)
            prior_project = _logged_time_project_candidate(prior, prior_person)
            if _personal_logged_time_question(prior):
                personal = bool(current_user_id)
                scope_text = prior
                break
            if prior_person or prior_project:
                scope_text, requested_person = prior, prior_person
                break

    db = mongo_service.get_db()
    person_name, person_ids = None, []
    if personal:
        person_ids = _id_variants([current_user_id])
        person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")
    elif requested_person:
        person_ids = _id_variants(_ids_by_name("users", requested_person))
        if person_ids:
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), requested_person)

    # Privacy: a non-admin account may only read its own logged-time data. It may never
    # view another person's logged time, and "all logged time" is restricted to the asker.
    # A caller without a verified identity (anonymous/guest) is denied outright.
    if not is_admin:
        if not current_user_id:
            return ExactResult(
                collection="cards", total=0, records=[], conditions="not authorized: unauthenticated",
                complete=True, source="planner", operation="list",
                notes=["Not authorized: sign in to view logged time."],
                direct_answer="Please sign in to view logged time.",
            )
        me = str(current_user_id)
        if personal:
            person_ids = _id_variants([current_user_id])
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")
        elif requested_person:
            resolved = {str(value) for value in person_ids}
            if me not in resolved:
                return ExactResult(
                    collection="cards", total=0, records=[], conditions=f"not authorized: {requested_person}",
                    complete=True, source="planner", operation="list",
                    notes=[f"Not authorized to view logged time for '{requested_person}'."],
                    direct_answer=f"You are not authorized to view logged time for '{requested_person}'.",
                )
            person_ids = _id_variants([current_user_id])
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")
        else:
            person_ids = _id_variants([current_user_id])
            person_name = next(iter(_logged_time_user_names(db, person_ids).values()), "you")

    requested_project = _logged_time_project_candidate(scope_text, person_name)
    if requested_person and not person_ids:
        if requested_project and _norm(requested_person) == _norm(requested_project):
            requested_person = None
        else:
            return _logged_time_empty(f"'{requested_person}' (logged time)",
                                      f"No user matched the name '{requested_person}'.")

    existing = set(db.list_collection_names())
    boards = {}
    if "boards" in existing:
        for board in db["boards"].find({}, {"name": 1, "project": 1}):
            boards[str(board["_id"])] = _clean_label(
                board.get("name") or board.get("project"), "Unknown project")

    if not requested_project and boards:
        q_tokens = set(re.findall(r"[a-z0-9]+", scope_text.lower()))
        matches = []
        for board_id, board_name in boards.items():
            tokens = [t for t in re.findall(r"[a-z0-9]+", str(board_name).lower())
                      if len(t) > 1 and t not in _LOGGED_TIME_STOPS]
            if tokens and all(token in q_tokens for token in tokens):
                matches.append((len(tokens), board_name, board_id))
        if matches:
            requested_project = max(matches)[1]

    if (re.search(r"\b(?:this|that|these|those)\b", text, re.IGNORECASE)
            and not personal and not requested_person and not requested_project):
        return ExactResult(
            collection="cards", total=0, records=[], conditions="ambiguous logged-time target",
            complete=True, source="planner", operation="clarify",
            direct_answer="Which task or project do you mean? Please share its name, and I can find its logged time.",
            notes=["The question contains a demonstrative without a resolvable task or project in the available context."],
            timings={"db": 0},
        )

    if requested_project and _norm(requested_project) in {"all", "every", "each", "projects"}:
        requested_project = None

    flt = {"loggedTime.0": {"$exists": True}}
    if person_ids:
        flt["loggedTime.user"] = {"$in": person_ids}

    projection = {"title": 1, "board": 1, "status": 1, "loggedTime": 1}
    cards = list(db["cards"].find(flt, projection).sort("_id", 1)) if "cards" in existing else []

    project_tokens = [t for t in re.findall(r"[a-z0-9]+", (requested_project or "").lower())
                      if t not in {"task", "tasks", "card", "cards", "project", "projects", "board", "boards"}]
    matched = []
    cards_used = 0
    for card in cards:
        title = _clean_label(card.get("title"), "Untitled task")
        board_id = card.get("board")
        project = boards.get(str(board_id), "Unknown project") if board_id is not None else "Unknown project"
        if project_tokens:
            haystack = f"{project} {title}".lower()
            if not all(
                token in haystack or (len(token) > 3 and token.endswith("s") and token[:-1] in haystack)
                for token in project_tokens
            ):
                continue
        entries = card.get("loggedTime") or []
        if isinstance(entries, dict):
            entries = [entries]
        card_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            user_id = _logged_time_entry_user(entry)
            if person_ids and user_id not in {str(value) for value in person_ids}:
                continue
            d_val = entry.get("date") or entry.get("createdAt")
            if period is not None:
                if not _entry_in_period(d_val, period):
                    continue
            elif target_date:
                t_year, t_month, t_day = target_date
                if not d_val:
                    continue
                e = _coerce_entry_dt(d_val)
                e_year, e_month, e_day = (e.year, e.month, e.day) if e else (None, None, None)
                if e_month != t_month or e_day != t_day:
                    continue
                if t_year is not None and e_year is not None and e_year != t_year:
                    continue
            card_entries.append((entry, user_id))
        if not card_entries:
            continue
        cards_used += 1
        for entry, user_id in card_entries:
            hours = _to_number(entry.get("hours")) or 0
            minutes = _to_number(entry.get("minutes")) or 0
            date_value = entry.get("date") or entry.get("createdAt")
            if hasattr(date_value, "isoformat"):
                date_value = date_value.isoformat()
            extras = []
            for key, value in entry.items():
                if key in {"_id", "user", "userId", "user_id", "hours", "minutes", "date",
                           "createdAt", "description"} or is_sensitive(str(key)):
                    continue
                if isinstance(value, bool):
                    extras.append(f"{humanize_key(key)}: {'Yes' if value else 'No'}")
                elif isinstance(value, (str, int, float)) and str(value).strip():
                    extras.append(f"{humanize_key(key)}: {' '.join(str(value).split())}")
                elif isinstance(value, (list, tuple)) and all(
                        isinstance(item, (str, int, float, bool)) for item in value):
                    extras.append(f"{humanize_key(key)}: {', '.join(str(item) for item in value)}")
            matched.append({
                "document_id": str(card.get("_id", "")),
                "title": title,
                "project": project,
                "status": str(card.get("status") or "").strip(),
                "person_id": user_id,
                "person": None,
                "minutes": round(hours * 60 + minutes),
                "date": date_value,
                "description": strip_html(entry.get("description")) if entry.get("description") else None,
                "extras": extras,
            })

    if not matched:
        date_suffix = f" on {target_date_str}" if target_date_str else ""
        scope = (f"'{person_name}'" if person_name else f"project '{requested_project}'"
                 if requested_project else "the cards collection")
        conditions = ("loggedTime.user = " + str(person_name) if person_name else
                      f"project = {requested_project}" if requested_project else "all loggedTime entries") + date_suffix
        return _logged_time_empty(scope, f"No nested loggedTime entries matched in 'cards' ({conditions}).")

    user_names = _logged_time_user_names(db, {item["person_id"] for item in matched if item["person_id"]})
    for item in matched:
        item["person"] = user_names.get(item["person_id"], person_name or "Unidentified contributor")

    # Check for superlative / extremum queries: least vs most
    min_pat = re.compile(
        r"\b(?:least|lowest|minimum|min|smallest|shortest|less|fewer|fewest|"
        r"sabse\s*(?:thoda|kam|chhota|chota|nyuntam)|"
        r"sase\s*(?:thoda|kam|chhota|chota|nyuntam)|"
        r"kam\s*se\s*kam|thoda|kam)\b",
        re.IGNORECASE,
    )
    max_pat = re.compile(
        r"\b(?:most|highest|maximum|max|largest|biggest|longest|more|"
        r"sabse\s*(?:jayda|jyada|zyada|jada|bada|adhiktam)|"
        r"sase\s*(?:jayda|jyada|zyada|jada|bada|adhiktam)|"
        r"jyada\s*se\s*jyada|jayda\s*se\s*jayda|jayda|jyada|zyada)\b",
        re.IGNORECASE,
    )
    extremum = "least" if min_pat.search(scope_text) else "most" if max_pat.search(scope_text) else None

    team_ranking = False
    if extremum and not person_name and not personal:
        # Team-level superlative (question like "who has the least logged time in the frontend team?"):
        # rank every contributor in the scoped project(s) by their total logged time and
        # answer about the top performer instead of dumping all entries plus the grand total.
        per_person = {}
        for item in matched:
            person = item["person"] or "Unidentified contributor"
            bucket = per_person.setdefault(person, {"minutes": 0, "entries": 0})
            bucket["minutes"] += item["minutes"]
            bucket["entries"] += 1
        if per_person:
            target_person = (
                min(per_person.items(), key=lambda pair: pair[1]["minutes"])[0]
                if extremum == "least"
                else max(per_person.items(), key=lambda pair: pair[1]["minutes"])[0]
            )
            attributed = [item for item in matched if (item["person"] or "Unidentified contributor") == target_person]
            if attributed:
                person_name = target_person
                matched = attributed
                team_ranking = True

    # Keep the full team ranking so team-level superlative answers can show every
    # contributor's total, not just the winner's entries.
    ranking = []
    if team_ranking:
        ranking = sorted(per_person.items(), key=lambda pair: pair[1]["minutes"],
                         reverse=(extremum == "most"))

    # The per-entry extremum filter must only apply when a person was explicitly named
    # ("Anil's highest logged entry"); after team-level ranking the winner is already
    # chosen by TOTAL minutes, so filtering per entry would discard their total.
    if extremum == "least" and not team_ranking:
        target_val = min(item["minutes"] for item in matched)
        matched = [item for item in matched if item["minutes"] == target_val]
    elif extremum == "most" and not team_ranking:
        target_val = max(item["minutes"] for item in matched)
        matched = [item for item in matched if item["minutes"] == target_val]

    matched.sort(key=lambda item: (str(item["person"]).casefold(), str(item["project"]).casefold(),
                                   str(item["title"]).casefold(), str(item["date"] or ""), item["document_id"]))
    total_minutes = sum(item["minutes"] for item in matched)
    cards_used = len({item["document_id"] for item in matched})

    groups = {}
    for item in matched:
        person_group = groups.setdefault(item["person"], {})
        project_group = person_group.setdefault(item["project"], {})
        project_group.setdefault((item["title"], item["status"], item["document_id"]), []).append(item)

    qualifier = "Least logged-time" if extremum == "least" else "Highest logged-time" if extremum == "most" else "All logged-time"
    entry_noun = "entry" if len(matched) == 1 else "entries"
    matched_as_task = (
        bool(requested_project) and
        any(_norm(requested_project) in _norm(item.get("title", "")) for item in matched)
    )
    scope_kind = "task" if (
        matched_as_task and
        not any(_norm(requested_project) in _norm(item.get("project", "")) for item in matched)
    ) else "project"
    date_suffix = f" on {target_date_str}" if target_date_str else ""
    if person_name and requested_project:
        heading = f"{qualifier} {entry_noun} for {person_name} in {scope_kind} {requested_project}{date_suffix}:"
    elif person_name:
        heading = f"{qualifier} {entry_noun} for {person_name}{date_suffix}:"
    elif requested_project:
        heading = f"{qualifier} {entry_noun} for {scope_kind} {requested_project}{date_suffix}:"
    else:
        heading = f"{qualifier} {entry_noun}{date_suffix}:"

    task_noun = "task" if cards_used == 1 else "tasks"
    if extremum:
        lines = [heading,
                 f"Duration: {_format_duration(total_minutes)} ({_format_duration(matched[0]['minutes'])} each) across {len(matched)} {entry_noun} on {cards_used} {task_noun}." if len(matched) > 1 else
                 f"Duration: {_format_duration(total_minutes)} across 1 entry on {cards_used} {task_noun}.",
                 ""]
    else:
        lines = [heading,
                 f"Total: {_format_duration(total_minutes)} across {len(matched)} entries on {cards_used} {task_noun}.",
                 ""]
    for person, person_projects in groups.items():
        person_minutes = sum(item["minutes"] for project in person_projects.values()
                             for entries in project.values() for item in entries)
        if len(groups) > 1:
            lines.append(f"{person} — {_format_duration(person_minutes)}")
        for project, project_cards in person_projects.items():
            lines.append(f"Project: {project}")
            for (title, status, _doc_id), entries in project_cards.items():
                suffix = f" [{status}]" if status else ""
                lines.append(f"- {title}{suffix}")
                for item in entries:
                    detail = f"  - {item['date'] + ': ' if item['date'] else ''}{_format_duration(item['minutes'])}"
                    if item["description"]:
                        detail += f" — {item['description']}"
                    if item["extras"]:
                        detail += "; " + "; ".join(item["extras"])
                    lines.append(detail)
            lines.append("")
        if len(groups) > 1:
            lines.append("")

    if team_ranking and len(ranking) > 1 and is_admin:
        word = "highest" if extremum == "most" else "lowest"
        lines.append(f"Ranking ({word} logged time first, {len(ranking)} contributors):")
        for name, bucket in ranking[:8]:
            marker = " ←" if name == person_name else ""
            lines.append(f"  {name} — {_format_duration(bucket['minutes'])} "
                         f"({bucket['entries']} {'entry' if bucket['entries'] == 1 else 'entries'}){marker}")
        if len(ranking) > 8:
            lines.append(f"  … and {len(ranking) - 8} more contributors.")
        lines.append("")

    records = []
    record_text = {}
    for item in matched:
        key = (item["document_id"], item["title"], item["project"], item["status"])
        detail = f"{item['date'] + ': ' if item['date'] else ''}{_format_duration(item['minutes'])}"
        if item["description"]:
            detail += f" — {item['description']}"
        if item["extras"]:
            detail += "; " + "; ".join(item["extras"])
        record_text.setdefault(key, []).append(detail)
    for (document_id, title, project, status), details in record_text.items():
        prefix = f"Project: {project}; Task: {title}" + (f"; Status: {status}" if status else "")
        records.append({"document_id": document_id, "name": title,
                        "text": prefix + "; Entries: " + "; ".join(details)})

    conditions = ("loggedTime.user = " + str(person_name) if person_name else
                  f"project = {requested_project}" if requested_project else "all loggedTime entries")
    return ExactResult(
        collection="cards", total=len(matched), records=records, conditions=conditions,
        complete=True, source="planner", operation="list",
        notes=[f"Read {len(matched)} logged-time entries from {cards_used} cards; the list is complete."],
        direct_answer="\n".join(lines).strip(),
        timings={"db": 0},
    )


def _leave_types_empty(note: str):
    message = ("The leave system tracks three leave types: Full Leave, Short Leave, and Halfday Leave. "
               "No leave task records were found in 'cards' to confirm which ones are actually used.")
    return ExactResult(
        collection="cards", total=0, records=[], conditions="leave types", complete=True,
        source="planner", operation="list", notes=[note], direct_answer=message,
    )


def _leave_types_answer(question: str, workspace_id: str = None):
    """Return the leave types configured in the workspace from 'leavetypes' collection."""
    text = " ".join((question or "").split())
    if not _LEAVE_TYPES_PATTERN.search(text):
        return None

    db = mongo_service.get_db()
    existing = set(db.list_collection_names())
    types_list = []
    if "leavetypes" in existing:
        query = {"isActive": {"$ne": False}}
        if workspace_id:
            query = _and_merge(_workspace_filter("leavetypes", workspace_id, db), query)
        try:
            types_list = list(db["leavetypes"].find(query).sort("displayOrder", 1))
            if not types_list and workspace_id:
                types_list = list(db["leavetypes"].find().sort("displayOrder", 1))
        except Exception as exc:
            logger.warning("Querying 'leavetypes' failed: %s", exc)

    records = []
    lines = []
    if types_list:
        lines.append(f"The workspace has {len(types_list)} configured leave type(s):")
        for lt in types_list:
            name = lt.get("name") or lt.get("key") or "Leave"
            desc = lt.get("description") or ""
            half = " (Supports Half Day)" if lt.get("supportsHalfDay") else ""
            details = f"{name}{half}"
            if desc:
                details += f" — {desc}"
            records.append({"document_id": str(lt["_id"]), "name": name, "text": details})
            lines.append(f"- **{name}**{half}" + (f": {desc}" if desc else ""))
        message = "\n".join(lines)
        notes = [f"Read {len(records)} leave type(s) from 'leavetypes' collection."]
        coll = "leavetypes"
    else:
        message = ("The system tracks leave types configured in 'leavetypes' (such as Full Day Leave and Short Leave). "
                   "No active records were found.")
        notes = ["No active records found in 'leavetypes'."]
        coll = "leavetypes"

    return ExactResult(
        collection=coll, total=len(records), records=records, conditions="leave types",
        complete=True, source="planner", operation="list", notes=notes, direct_answer=message,
        timings={"db": 0},
    )


def _logged_time_for_project(project_filter: str):
    """Return exact logged-time totals for every contributor on a named project."""
    db = mongo_service.get_db()
    collections = set(db.list_collection_names())
    if "boards" not in collections:
        return None

    wanted = _norm(project_filter)
    tokens = wanted.split()
    boards = [board for board in db["boards"].find({}, {"name": 1, "project": 1})
              if all(token in _norm(board.get("name") or board.get("project") or "") for token in tokens)]
    board_ids = {str(board["_id"]): board for board in boards}
    # Some imported workspaces store the project label on a card title while
    # the referenced board has a generic or unrelated name. Resolve those cards
    # to their board so the whole project's logs can still be aggregated.
    title_pattern = re.compile(re.escape(project_filter), re.IGNORECASE)
    for source in settings.LOGGED_TIME_COLLECTIONS:
        if source == "activities" or source not in collections:
            continue
        for doc in db[source].find({"title": title_pattern, "board": {"$exists": True}},
                                   {"board": 1}).limit(200):
            board_id = doc.get("board")
            if board_id is not None:
                board_ids.setdefault(str(board_id), {"_id": board_id})
    if not board_ids:
        return None
    raw_board_ids = [value for board in boards for value in (board["_id"], str(board["_id"]))]
    for board in board_ids.values():
        board_id = board.get("_id")
        if board_id is not None:
            raw_board_ids.extend((board_id, str(board_id)))
    raw_board_ids = list(dict.fromkeys(raw_board_ids))
    # Fetch names for boards discovered from matching card titles.
    for board in db["boards"].find({"_id": {"$in": raw_board_ids}}, {"name": 1, "project": 1}):
        board_ids[str(board["_id"])] = board
    project_names = {key: board.get("name") or board.get("project") or key
                     for key, board in board_ids.items()}

    entries = []
    for source in settings.LOGGED_TIME_COLLECTIONS:
        if source in {"activities"} or source not in collections:
            continue
        cursor = db[source].find(
            {"board": {"$in": raw_board_ids}, "loggedTime": {"$exists": True}},
            {"title": 1, "board": 1, "loggedTime": 1},
        )
        for doc in cursor:
            for item in doc.get("loggedTime") or []:
                if not isinstance(item, dict):
                    continue
                person_id = item.get("user")
                if person_id is None:
                    continue
                entries.append({
                    "person_id": str(person_id), "person_name": None,
                    "board_id": str(doc.get("board")),
                    "title": doc.get("title") or "Untitled work item",
                    "hours": _to_number(item.get("hours")) or 0,
                    "minutes": _to_number(item.get("minutes")) or 0,
                    "date": item.get("date"), "description": item.get("description"),
                    "source": source,
                })

    # Resolve user references once so the output can be grouped by contributor.
    user_values = {entry["person_id"] for entry in entries if entry["person_id"]}
    user_ids = []
    for value in user_values:
        user_ids.append(value)
        if HEX24.match(value):
            try:
                user_ids.append(ObjectId(value))
            except Exception:
                pass
    user_names = {}
    if user_ids and "users" in collections:
        for user in db["users"].find({"_id": {"$in": user_ids}}, {"name": 1}):
            user_names[str(user["_id"])] = user.get("name")

    totals = {}
    grand_total = 0
    for entry in entries:
        person = entry["person_name"] or user_names.get(entry["person_id"]) or "Unidentified contributor"
        minutes = round(entry["hours"] * 60 + entry["minutes"])
        group = totals.setdefault(person, {"minutes": 0, "entries": []})
        group["minutes"] += minutes
        grand_total += minutes
        date = entry["date"]
        if hasattr(date, "isoformat"):
            date = date.isoformat()
        detail = f"{entry['title']} — {minutes // 60}h {minutes % 60}m"
        if date:
            detail += f", date: {date}"
        if entry["description"]:
            detail += f", description: {entry['description']}"
        detail += f" (source: {entry['source']})"
        group["entries"].append(detail)

    if len(project_names) == 1:
        only_project_name = next(iter(project_names.values()))
        project = only_project_name if wanted in _norm(only_project_name) else project_filter
    else:
        project = project_filter
    if not entries:
        summary = f"No logged-time entries found for project {project}."
    else:
        people = []
        for person, data in sorted(totals.items(), key=lambda pair: pair[0].casefold()):
            minutes = data["minutes"]
            people.append(f"{person}: {minutes // 60}h {minutes % 60}m; entries: "
                          + "; ".join(data["entries"]))
        summary = (f"Exact total logged time for project {project}: {grand_total // 60}h "
                   f"{grand_total % 60}m across {len(entries)} entries. By contributor: "
                   + " | ".join(people))

    return ExactResult(
        collection="cards + subtasknanos + subtasks", total=len(entries),
        records=[{"document_id": project, "name": project, "text": summary}],
        conditions=f"logged-time entries for project {project}", complete=True, source="planner",
        operation="list", notes=[f"Read {len(entries)} matching entries across all configured time sources."],
        timings={"db": 0},
    )


def _logged_time_by_project(question: str, history: list = None, current_user_id: str = None):
    """Collect matching time logs, resolving calculation follow-ups from prior user turns."""
    if not re.search(r"\b(?:logged\s*time|loggedtime|time\s+logged|hours)\b", question, re.IGNORECASE):
        return None
    is_total_followup = bool(history and re.search(
        r"\b(?:calculate|compute|sum|total)\b", question, re.IGNORECASE,
    ))
    personal = bool(current_user_id and re.search(r"\b(?:my|mine|myself)\b", question, re.IGNORECASE))
    person = None if personal else re.search(
        r"\b([A-Za-z][A-Za-z .'-]*?)['’]s\s+(?:all\s+)?(?:logged\s*time|loggedtime|hours|timesheet)\b",
        question, re.IGNORECASE,
    ) or re.search(
        r"^\s*([A-Za-z][A-Za-z .'-]*?)\s+(?:(?:all|total)\s+)?(?:logged\s*time|loggedtime|hours|timesheet)\b",
        question, re.IGNORECASE,
    ) or re.search(
        r"\b(?:of|for|by)\s+([A-Za-z][A-Za-z .'-]*?)(?=\s+(?:in|across|on|from|between)\b|[?.!,]|$)",
        question, re.IGNORECASE,
    )
    if person and _norm(person.group(1)) in {"logged time", "time logged", "hours", "total hours"}:
        person = None
    project_filter_match = re.search(r"\bin\s+(.+?)(?=[?.!,]|$)", question, re.IGNORECASE)
    scope_question = question
    if not person and not personal and is_total_followup:
        # Resolve the active person and project from the latest relevant user
        # query; the database is re-read below, so prior assistant text is not data.
        for turn in reversed(history or []):
            prior = turn.get("question", "")
            if not re.search(r"\b(?:logged\s*time|loggedtime|time\s+logged|hours)\b", prior, re.IGNORECASE):
                continue
            scope_question = prior
            person = re.search(
                r"^\s*([A-Za-z][A-Za-z .'-]*?)\s+(?:all\s+)?logged\s*time\b",
                prior, re.IGNORECASE,
            ) or re.search(
                r"\b(?:of|for|by)\s+([A-Za-z][A-Za-z .'-]*?)(?=\s+(?:in|across|on|from|between)\b|[?.!,]|$)",
                prior, re.IGNORECASE,
            )
            if person:
                break
    if not person and not personal and project_filter_match:
        project_filter = re.sub(r"\s+project$", "", project_filter_match.group(1).strip(), flags=re.IGNORECASE)
        if project_filter:
            return _logged_time_for_project(project_filter)
    # Any explicit person's logged-time question should use every configured
    # source, even when the user did not ask for a per-project breakdown.
    if not person and not personal:
        return None
    if personal:
        user_ids = _id_variants([current_user_id])
        user = mongo_service.get_db()["users"].find_one(
            {"_id": {"$in": user_ids}}, {"name": 1, "firstName": 1, "lastName": 1}
        )
        person_name = (user or {}).get("name") or " ".join(filter(None, (
            (user or {}).get("firstName"), (user or {}).get("lastName")
        ))) or "you"
    else:
        person_name = " ".join(person.group(1).split()).strip(" .'-")
    if not person_name:
        return None

    if not personal:
        user_ids = _ids_by_name("users", person_name)
    if not user_ids:
        # Keep explicit logged-time requests on the exact path. Falling back to
        # semantic search could surface audit events from the activities index.
        return ExactResult(
            collection="cards + subtasknanos + subtasks", total=0, records=[],
            conditions=f"logged-time entries for {person_name}", complete=True,
            source="planner", operation="list",
            notes=[f"No matching user was found for {person_name}."], timings={"db": 0},
        )
    user_id_strings = {str(value) for value in user_ids}
    db = mongo_service.get_db()
    existing_collections = set(db.list_collection_names())
    project_filter_match = re.search(r"\bin\s+(.+?)(?=[?.!,]|$)", scope_question, re.IGNORECASE)
    project_filter = project_filter_match.group(1).strip() if project_filter_match else None
    if project_filter:
        project_filter = re.sub(r"\s+projects?$", "", project_filter, flags=re.IGNORECASE).strip()
        if _norm(project_filter) in {"all", "every", "each"}:
            project_filter = None
    projects = {}
    matched = []
    # Logged time comes only from nested loggedTime entries on work items.
    for source in settings.LOGGED_TIME_COLLECTIONS:
        if source == "activities":
            continue
        if source not in existing_collections:
            continue
        cursor = db[source].find(
            {"loggedTime.user": {"$in": user_ids}},
            {"title": 1, "board": 1, "loggedTime": 1},
        )
        for doc in cursor:
            board_id = doc.get("board")
            for entry in doc.get("loggedTime") or []:
                if not isinstance(entry, dict) or str(entry.get("user")) not in user_id_strings:
                    continue
                hours = _to_number(entry.get("hours")) or 0
                minutes = _to_number(entry.get("minutes")) or 0
                matched.append({
                    "board_id": board_id,
                    "title": doc.get("title") or "Untitled work item",
                    "source": source,
                    "hours": hours,
                    "minutes": minutes,
                    "date": entry.get("date"),
                    "description": strip_html(entry.get("description")) if entry.get("description") else None,
                })

    board_ids = list({entry["board_id"] for entry in matched if entry["board_id"] is not None})
    # Imported records may store a board reference as a hex string while the
    # board document uses an ObjectId (or vice versa). Query both forms.
    board_lookup_ids = list(board_ids)
    for board_id in board_ids:
        board_id_text = str(board_id)
        if HEX24.match(board_id_text):
            try:
                board_lookup_ids.append(ObjectId(board_id_text))
            except Exception:
                pass
        board_lookup_ids.append(board_id_text)
    board_lookup_ids = list(dict.fromkeys(board_lookup_ids))
    if board_ids and "boards" in db.list_collection_names():
        for board in db["boards"].find({"_id": {"$in": board_lookup_ids}}, {"name": 1, "project": 1}):
            projects[str(board["_id"])] = board.get("name") or board.get("project") or str(board["_id"])

    if project_filter:
        filter_tokens = _norm(project_filter).split()
        matched = [entry for entry in matched if all(
            token in _norm(projects.get(str(entry["board_id"]), ""))
            or token in _norm(entry.get("title", ""))
            or token in _norm(entry.get("description", ""))
            for token in filter_tokens
        )]

    if is_total_followup:
        total_minutes = sum(round(entry["hours"] * 60 + entry["minutes"]) for entry in matched)
        scope = f" in {project_filter}" if project_filter else ""
        total_text = (f"Exact total logged time for {person_name}{scope}: "
                      f"{total_minutes // 60}h {total_minutes % 60}m across {len(matched)} entries.")
        return ExactResult(
            collection="cards + subtasknanos + subtasks",
            total=len(matched), records=[{"document_id": person_name, "name": person_name, "text": total_text}],
            conditions=f"logged-time entries for {person_name}{scope}", complete=True, source="planner",
            operation="list", notes=[total_text], timings={"db": 0},
        )

    grouped = {}
    for entry in matched:
        date = entry["date"]
        if hasattr(date, "isoformat"):
            date = date.isoformat()
        project = projects.get(str(entry["board_id"]), "Unknown project")
        group = grouped.setdefault(project, {"minutes": 0, "entries": []})
        total_minutes = round(entry["hours"] * 60 + entry["minutes"])
        group["minutes"] += total_minutes
        detail = f"{entry['title']} — {total_minutes // 60}h {total_minutes % 60}m"
        if date:
            detail += f", date: {date}"
        if entry["description"]:
            detail += f", description: {entry['description']}"
        detail += f" (source: {entry['source']})"
        group["entries"].append(detail)

    records = []
    for project, data in sorted(grouped.items(), key=lambda item: item[0].casefold()):
        minutes = data["minutes"]
        detail_text = "; ".join(data["entries"])
        records.append({
            "document_id": project,
            "name": project,
            "text": f"Project: {project}; total logged time for {person_name}: "
                    f"{minutes // 60}h {minutes % 60}m; entries: {detail_text}",
        })
    return ExactResult(
        collection="cards + subtasknanos + subtasks (grouped by boards)", total=len(records), records=records,
        conditions=f"loggedTime.user = {person_name}", complete=True, source="planner",
        operation="list", notes=[f"Grouped {len(matched)} matching logged-time entries by project."],
        timings={"db": 0},
    )
                             

def _project_team_details(result: ExactResult):
    """Enrich matched project rows with exact linked people and work-item counts."""
    db = mongo_service.get_db()
    existing_collections = set(db.list_collection_names())
    for record in result.records:
        board = mongo_service.get_document("boards", record["document_id"])
        if not board:
            continue
        board_id = board.get("_id")
        board_ids = [board_id, str(board_id)]
        if isinstance(board_id, str) and ObjectId.is_valid(board_id):
            board_ids.append(ObjectId(board_id))
        elif isinstance(board_id, ObjectId):
            board_ids.append(str(board_id))
        people = {}

        def add_person(value):
            if value is not None and not isinstance(value, dict):
                people[str(value)] = value

        owner = board.get("owner")
        if owner:
            add_person(owner)
        members = board.get("members") or []
        if not isinstance(members, (list, tuple, set)):
            members = [members]
        for value in members:
            add_person(value)

        cards = list(db["cards"].find(
            {"board": {"$in": board_ids}},
            {"assignees": 1, "loggedTime.user": 1, "status": 1, "isArchived": 1},
        )) if "cards" in existing_collections else []
        for card in cards:
            for value in card.get("assignees") or []:
                add_person(value)
            for entry in card.get("loggedTime") or []:
                if isinstance(entry, dict):
                    add_person(entry.get("user"))

        subtasks = list(db["subtasks"].find(
            {"board": {"$in": board_ids}}, {"loggedTime.user": 1, "status": 1, "isArchived": 1},
        )) if "subtasks" in existing_collections else []
        for subtask in subtasks:
            for entry in subtask.get("loggedTime") or []:
                if isinstance(entry, dict):
                    add_person(entry.get("user"))

        candidates = []
        for value in list(people.values()):
            if isinstance(value, ObjectId):
                candidates.append(value)
                candidates.append(str(value))
            elif isinstance(value, str) and HEX24.match(value):
                candidates.append(value)
                try:
                    candidates.append(ObjectId(value))
                except Exception:
                    pass
        users = {}
        if candidates and "users" in existing_collections:
            for user in db["users"].find({"_id": {"$in": candidates}}, {"name": 1, "firstName": 1, "lastName": 1, "fullName": 1}):
                name = user.get("name") or user.get("fullName") or " ".join(filter(None, (user.get("firstName"), user.get("lastName"))))
                if name:
                    users[str(user["_id"])] = name
        names = sorted({users[person_id] for person_id in people if person_id in users}, key=str.casefold)
        open_cards = sum(not card.get("isArchived") and str(card.get("status", "")).lower() not in
                         {"done", "completed", "cancelled", "canceled"} for card in cards)
        open_subtasks = sum(not item.get("isArchived") and str(item.get("status", "")).lower() not in
                            {"done", "completed", "cancelled", "canceled"} for item in subtasks)
        team_text = ", ".join(names) if names else "No linked team members found"
        record["text"] = (record.get("text", "") +
                           f"; Linked people: {len(names)} unique ({team_text}); "
                           f"cards: {len(cards)} total, {open_cards} not marked complete; "
                           f"subtasks: {len(subtasks)} total, {open_subtasks} not marked complete. "
                           "People are counted from board members, card assignees, and other linked work.")


def try_structured(question: str, history: list = None, current_user_id: str = None, is_admin: bool = False,
                   restrict_unscoped: bool = False, workspace_id: str = None):
    """-> ExactResult, or None when similarity search should handle the question.

    `restrict_unscoped` must be True for every request-serving caller: when it is set,
    the answers below that have no internal scope enforcement (user attributes, person
    lookups, employee work status, planner/catalog reads) run only for admins.
    """
    if not settings.STRUCTURED_QUERIES:
        return None

    workspace_employee_list = _workspace_employee_list_answer(question, is_admin=is_admin)
    if workspace_employee_list is not None:
        workspace_employee_list.timings = {"plan": 0, "db": 0}
        return workspace_employee_list

    workspace_employee_count = _workspace_employee_count_answer(question, is_admin=is_admin)
    if workspace_employee_count is not None:
        workspace_employee_count.timings = {"plan": 0, "db": 0}
        return workspace_employee_count

    leave_types = _leave_types_answer(question, workspace_id=workspace_id)
    if leave_types is not None:
        leave_types.context = _context(leave_types)
        leave_types.timings = {"plan": 0, **leave_types.timings}
        return leave_types

    leave_time = _leave_time_answer(question, history=history, current_user_id=current_user_id,
                                    is_admin=is_admin, workspace_id=workspace_id)
    if leave_time is not None:
        leave_time.context = _context(leave_time)
        leave_time.timings = {"plan": 0, **leave_time.timings}
        return leave_time

    logged_time = _logged_time_answer(question, history=history, current_user_id=current_user_id,
                                      is_admin=is_admin)
    if logged_time is not None:
        logged_time.context = _context(logged_time)
        logged_time.timings = {"plan": 0, **logged_time.timings}
        return logged_time

    project_type_result = _project_type_answer(
        question, workspace_id=workspace_id,
        restrict_unscoped=restrict_unscoped, is_admin=is_admin,
    )
    if project_type_result is not None:
        project_type_result.context = _context(project_type_result)
        project_type_result.timings = {"plan": 0, **project_type_result.timings}
        return project_type_result

    if restrict_unscoped and not is_admin:
        return None

    try:
        began = time.perf_counter()
        workspace_admin = _workspace_admin_answer(question, workspace_id=workspace_id)
        if workspace_admin is not None:
            workspace_admin.context = _context(workspace_admin)
            workspace_admin.timings = {"plan": 0, "db": round((time.perf_counter() - began) * 1000)}
            return workspace_admin
        uncompleted_projects = _uncompleted_projects_answer(question, workspace_id=workspace_id)
        if uncompleted_projects is not None:
            uncompleted_projects.context = _context(uncompleted_projects)
            uncompleted_projects.timings = {"plan": 0, **uncompleted_projects.timings}
            return uncompleted_projects
        team_answer = _project_or_task_team_answer(question, history=history, workspace_id=workspace_id)
        if team_answer is not None:
            team_answer.context = _context(team_answer)
            team_answer.timings = {"plan": round((time.perf_counter() - began) * 1000), **team_answer.timings}
            return team_answer
        user_attr = _user_attribute_answer(question, history=history, workspace_id=workspace_id)
        if user_attr is not None:
            user_attr.timings = {"plan": round((time.perf_counter() - began) * 1000), **user_attr.timings}
            return user_attr
        assignee_count = _assignee_count_answer(question)
        if assignee_count is not None:
            assignee_count.context = _context(assignee_count)
            assignee_count.timings = {"plan": round((time.perf_counter() - began) * 1000),
                                      **assignee_count.timings}
            return assignee_count
        person = _person_lookup(question, workspace_id=workspace_id)
        if person is not None:
            person.timings = {"plan": round((time.perf_counter() - began) * 1000), **person.timings}
            return person
        employee_work = _employee_work_status(question, history=history, workspace_id=workspace_id)
        if employee_work is not None:
            employee_work.timings = {"plan": round((time.perf_counter() - began) * 1000), **employee_work.timings}
            return employee_work
        catalog = build_catalog()
        if not catalog:
            return None
        history_key = tuple((turn["question"], turn["answer"]) for turn in (history or [])[-12:])
        key = (" ".join(question.lower().split()), history_key)
        cached = _plans.get(key)
        plan, source, planner_ok = None, "planner", True
        if cached is not None:
            plan = cached["plan"]
        elif settings.STRUCT_PLANNER == "off":
            plan = None                     # no planning call: rely on the no-LLM collection-name fallback below
        else:
            try:
                plan = validate_plan(llm_service.plan_query(
                    question, catalog_text(catalog, question=question), history=history), catalog)
            except ChatbotError as exc:
                planner_ok = False
                logger.warning("Query planner unavailable: %s", exc.message)
            if planner_ok:
                _plans.set(key, {"plan": plan})
        plan_ms = round((time.perf_counter() - began) * 1000)
        if plan is None and not history:
            name = match_collection(question, list(catalog))
            if name and catalog[name]["count"] <= settings.STRUCT_SCAN_MAX:
                plan, source = {"collection": name, "operation": "list", "filters": [], "show": []}, "name-match"
        if plan is None:
            return None
        wants_project_team = bool(
            re.search(r"\b(?:people|persons|employees|members|team|contributors|working)\b", question, re.IGNORECASE)
            and re.search(r"\b(?:project|board|working|assigned)\b", question, re.IGNORECASE)
        )
        if wants_project_team and plan.get("collection") == "boards":
            plan = {**plan, "operation": "list"}
        result = execute(plan, catalog, source, include_logged_time=wants_logged_time(question),
                         current_user_id=current_user_id, is_admin=is_admin, workspace_id=workspace_id)
        if result is None:
            return None
        if wants_project_team and plan.get("collection") == "boards":
            _project_team_details(result)
        result.context = _context(result)
        result.timings = {"plan": plan_ms, **result.timings}
        logger.info("Exact query on %s via %s: %d matching (%s) timings=%s", result.collection, source,
                    result.total, result.conditions, result.timings)
        return result
    except ChatbotError as exc:
        logger.warning("Exact query skipped: %s", exc.message)
        return None
    except Exception:
        logger.exception("Exact query failed; falling back to similarity search")
        return None


def try_current_user_work(question: str, user_id: str, history: list = None, user_name: str = None,
                          workspace_id=None, is_admin: bool = False):
    """Resolve personal work questions against the authenticated account's ID."""
    if not settings.STRUCTURED_QUERIES:
        return None
    is_personal_leave = bool(re.search(r"\b(?:my|mine|myself)\b", question, re.IGNORECASE))
    if is_personal_leave:
        leave_requests = _leave_requests_answer(
            question, current_user_id=user_id, history=history, user_name=user_name,
            workspace_id=workspace_id, is_admin=is_admin)
        if leave_requests is not None:
            leave_requests.context = _context(leave_requests)
            return leave_requests
    personal = _personal_logged_time_question(question) or any(
        isinstance(turn, dict) and _personal_logged_time_question(turn.get("question", ""))
        and is_logged_time_question(turn.get("question", ""))
        for turn in (history or [])
    )
    if not personal:
        return None
    result = _leave_time_answer(question, history=history, current_user_id=user_id)
    if result is None:
        result = _logged_time_answer(question, history=history, current_user_id=user_id)
    if result is None and _personal_logged_time_question(question):
        result = _employee_work_status(question, history=history, current_user_id=user_id)
    if result is not None:
        result.context = _context(result)
    return result
