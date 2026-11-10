"""
The full RAG pipeline for one question:

  question -> hybrid search (vector + keyword) -> multi-hop expansion -> context -> LLM -> answer

Only a few relevant records are sent to the LLM, never the whole database.
Answers are cached (keyed by question + index version), so repeated questions are free.
"""
import copy
import logging
import re
import time
from html import unescape
from urllib.parse import urlparse

from django.conf import settings

from . import (cache_service, embedding_service, embedding_store, llm_service, retrieval,
               chat_memory, structured_query, vector_index, query_correction, authorization_service)
from .exceptions import AuthorizationError, ChatbotError, InvalidRequestError, LLMError
from .llm_service import NOT_FOUND_ANSWER
from .structured_query import _sanitize_ids
from .document_converter import strip_html as _strip_html
from .document_converter import pick_display_name

logger = logging.getLogger(__name__)

MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 2, 500
_answer_cache = cache_service.TTLCache(settings.CACHE_MAX_ENTRIES, settings.CACHE_TTL_SECONDS)


# Questions that expect SEVERAL answers. They automatically retrieve more records (LIST_TOP_K).
LIST_PATTERN = re.compile(
    r"\b(?:all|every|each|list|enumerate|how many|show me|what are|who are|"
    r"last|latest|recent|when did|which\s+(?!is\b|are\b)\w{3,}s?|"
    r"what did|what has|what have|completed|complete|done|finished|total|"
    r"progress|performance|working|overview|assigned|department wise|group|grouped|grouping|sort|sorted|filter|breakdown)\b",
    re.IGNORECASE,
)



def is_list_question(question: str) -> bool:
    return bool(LIST_PATTERN.search(question))


def _multi_topic_leave_question(question: str) -> bool:
    """Keep policy and taken-leave details together in semantic retrieval when both are asked."""
    text = question.lower()
    asks_leave = re.search(r"\b(?:leave|leaves|vacation|time off)\b", text)
    asks_policy = re.search(r"\b(?:credit|credited|crediting|accrual|accrued|allowance|balance)\b", text)
    asks_usage = re.search(r"\b(?:take|takes|taken|request|requests|history|date|dates|expiry|expire|expires)\b", text)
    return bool(asks_leave and asks_policy and asks_usage)


GREETING_WORDS = {
    "hello", "hi", "hey", "hiya", "greetings", "namaste", "hola",
    "good morning", "good afternoon", "good evening", "good day",
    "hey there", "hello there", "hi there"
}

HOW_ARE_YOU_WORDS = {
    "how are you", "how are you doing", "how is it going", "hows it going",
    "how do you do", "what is up", "whats up", "wassup"
}

THANKS_WORDS = {
    "thanks", "thank you", "thank you so much", "thanks a lot", "thx",
    "many thanks", "appreciate it"
}

GOODBYE_WORDS = {
    "bye", "goodbye", "see you", "see ya", "take care", "cya", "good night", "have a good day"
}

CAPABILITIES_WORDS = {
    "who are you", "what are you", "what can you do", "help", "help me",
    "what do you do", "what is this", "what is this app"
}


SELF_IDENTITY_PATTERNS = re.compile(
    r"\b(?:who\s+am\s+i|what\s+is\s+my\s+name|what\s+am\s+i\s+called|"
    r"my\s+name\s+is|tell\s+me\s+my\s+name|identify\s+me|tell\s+me\s+about\s+myself|"
    r"my\s+profile|my\s+account|my\s+details|my\s+information|"
    r"what\s+is\s+my\s+email|my\s+email\s+address|tell\s+me\s+my\s+email|"
    r"what\s+is\s+my\s+role|my\s+role|tell\s+me\s+my\s+role|"
    r"what\s+do\s+you\s+know\s+about\s+me|am\s+i\s+an\s+admin|"
    r"am\s+i\s+an\s+employee|tell\s+me\s+about\s+me)\b",
    re.IGNORECASE,
)

_NAME_WORDS = re.compile(r"\bname\b", re.IGNORECASE)
_EMAIL_WORDS = re.compile(r"\bemail\b|\bemail\s+address\b", re.IGNORECASE)
_ROLE_WORDS = re.compile(r"\brole\b|\badmin\b|\bemployee\b", re.IGNORECASE)


def check_self_identity(question: str, scope) -> str | None:
    """Answer questions about the user's own identity (name, email, role) from the verified account."""
    if not SELF_IDENTITY_PATTERNS.search(question):
        return None
    if scope.user_id.startswith("guest:") or not scope.user_name:
        return None

    asks_name = bool(_NAME_WORDS.search(question))
    asks_email = bool(_EMAIL_WORDS.search(question))
    asks_role = bool(_ROLE_WORDS.search(question))

    # "who am i" with no specific field word is a general identity question.
    general = not (asks_name or asks_email or asks_role)

    parts = []
    if general or asks_name:
        parts.append(f"Your name is {scope.user_name}.")
    if (general or asks_email) and scope.user_email:
        parts.append(f"Your email is {scope.user_email}.")
    if (general or asks_role) and scope.user_role:
        parts.append(f"Your role is {scope.user_role}.")

    if not parts:
        parts.append(f"Your name is {scope.user_name}.")

    return " ".join(parts)


COMPANY_IDENTITY_PATTERNS = re.compile(
    r"\b(?:what\s+is\s+my\s+company|my\s+company|what\s+is\s+my\s+workspace|my\s+workspace|"
    r"which\s+company|which\s+workspace|my\s+organization|my\s+organization\s+name|"
    r"what\s+organization\s+am\s+i\s+in|my\s+company\s+name|"
    r"company\s+name\s+of\s+([A-Za-z][A-Za-z .'-]*?)|"
    r"([A-Za-z][A-Za-z .'-]*?)['’]s\s+company|"
    r"what\s+is\s+([A-Za-z][A-Za-z .'-]*?)['’]s\s+company)\b",
    re.IGNORECASE,
)

_COMPANY_PERSON_PATTERNS = (
    r"what\s+is\s+([A-Za-z][A-Za-z .'-]{1,40}?)['’]s\s+company\b",
    r"company\s+name\s+of\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?=\s*(?:$|[?.!]|company|workspace|organization))",
    r"([A-Za-z][A-Za-z .'-]{1,40}?)['’]s\s+company\b",
)


def _person_name_from_company_question(question: str) -> str | None:
    """Extract a person's name from 'company name of <person>' or '<person>'s company' questions."""
    _COMPANY_STOP_WORDS = {'company', 'companies', 'workspace', 'workspaces',
                           'organization', 'organizations', 'corporation', 'corp', 'inc', 'llc'}
    for pattern in _COMPANY_PERSON_PATTERNS:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            name = name.strip(" .'-")
            # Remove trailing stop words that may have been captured.
            words = name.split()
            while words and words[-1].lower() in _COMPANY_STOP_WORDS:
                words.pop()
            name = ' '.join(words).strip(" .'-")
            if name and len(name) >= 2:
                return name
    return None


def check_company_identity(question: str, scope) -> str | None:
    """Answer 'what is my company' / 'my workspace' / '<person>'s company' questions from verified accounts."""
    if not COMPANY_IDENTITY_PATTERNS.search(question):
        return None
    if scope.user_id.startswith("guest:"):
        return None

    # Check if asking about a specific person's company.
    person_name = _person_name_from_company_question(question)
    if person_name:
        # If asking about yourself, use the already-resolved scope (no DB lookup needed).
        if scope.user_name and person_name.casefold() == scope.user_name.casefold():
            if scope.user_company:
                return f"{person_name}'s company is {scope.user_company}."
            return f"{person_name} doesn't have a company/workspace recorded."
        try:
            from . import mongo_service
            db = mongo_service.get_db()
            # Direct name lookup across common name fields.
            user = db["users"].find_one(
                {
                    "$or": [
                        {"name": {"$regex": person_name, "$options": "i"}},
                        {"fullName": {"$regex": person_name, "$options": "i"}},
                        {"full_name": {"$regex": person_name, "$options": "i"}},
                        {"firstName": {"$regex": person_name, "$options": "i"}},
                        {"first_name": {"$regex": person_name, "$options": "i"}},
                        {"displayName": {"$regex": person_name, "$options": "i"}},
                        {"display_name": {"$regex": person_name, "$options": "i"}},
                        {"username": {"$regex": person_name, "$options": "i"}},
                    ]
                },
                {"lastActiveWorkspace": 1, "name": 1, "email": 1},
            )
            if not user:
                return f"I couldn't find a user named '{person_name}'."
            workspace_id = user.get("lastActiveWorkspace")
            if not workspace_id:
                return f"{person_name} doesn't have a company/workspace recorded."
            sid = str(workspace_id)
            workspace = db["workspaces"].find_one({"_id": sid})
            if not workspace:
                try:
                    from bson import ObjectId
                    workspace = db["workspaces"].find_one({"_id": ObjectId(sid)})
                except Exception:
                    workspace = None
            if not workspace:
                return f"{person_name}'s workspace could not be resolved."
            company = pick_display_name(workspace)
            if not company:
                company = workspace.get("name") or workspace.get("project") or "Unknown company"
            return f"{person_name}'s company is {company}."
        except Exception as exc:
            logger.warning("Company lookup failed for %s: %s", person_name, exc)
            return f"I couldn't look up the company for '{person_name}'."

    # "my company" / "my workspace" questions.
    if not scope.user_company:
        return None
    return f"Your company is {scope.user_company}."


def check_conversational(question: str) -> str | None:
    cleaned = " ".join(re.sub(r"[^a-z0-9\s]", "", question.lower()).split())
    if not cleaned:
        return None
    if cleaned in GREETING_WORDS:
        return "Hello! How can I help you today? You can ask me about employees, workspaces, projects, tasks, departments, or other data in the database."
    if cleaned in HOW_ARE_YOU_WORDS:
        return "I'm doing well, thank you! How can I assist you with your projects, employees, or database records today?"
    if cleaned in THANKS_WORDS:
        return "You're welcome! Let me know if you need any other information from the database."
    if cleaned in GOODBYE_WORDS:
        return "Goodbye! Have a great day ahead. Feel free to come back whenever you need database insights."
    if cleaned in CAPABILITIES_WORDS:
        return ("I am your MongoDB database assistant. You can ask me to list employees, "
                "check assigned tasks, view departments, inspect projects, or count records.")
    return None


def strip_greeting(text: str) -> str:
    cleaned = re.sub(
        r"^(?:hello|hi|hey|good\s+(?:morning|afternoon|evening)|greetings|namaste)\b[\s,!:\.\-]*",
        "", text, flags=re.IGNORECASE
    ).strip()
    return cleaned or text


def validate_question(question) -> str:
    if not isinstance(question, str):
        raise InvalidRequestError("The question must be text.")
    question = " ".join(question.split())
    if len(question) < MIN_QUESTION_CHARS:
        raise InvalidRequestError("Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidRequestError(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
    return question


def _group(hits: list) -> list:
    """Merge chunks of the same document; best-scoring document first."""
    groups = {}
    for hit in hits:
        groups.setdefault(hit.doc_key, []).append(hit)
    ordered = sorted(groups.values(), key=lambda g: max(h.score for h in g), reverse=True)
    return [sorted(g, key=lambda h: h.chunk_index) for g in ordered]


_LOGGED_TIME_LINE = re.compile(
    r"^\s*(?:Logged\s+Time|Time\s+Logged|LoggedTime|Work\s+Logs?|Time\s+Logs?|Timesheets?)\b",
    re.IGNORECASE,
)


def _without_logged_time_text(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _LOGGED_TIME_LINE.match(line))


def build_context(main_groups: list, related_groups: list, budget: int = None,
                  include_logged_time: bool = True):
    """Build the prompt context under a character budget. Returns (context, included_main, included_related)."""
    budget, used = budget or settings.MAX_CONTEXT_CHARS, 0
    blocks, main_done, related_done = [], [], []
    label_of = {}

    def add(label, group, note=""):
        nonlocal used
        text = "\n...\n".join(_strip_html(h.text) for h in group)
        if not include_logged_time:
            text = _without_logged_time_text(text)
        prefix = f"[{label}]{note}\n"
        remaining = budget - used - (2 if blocks else 0)
        if remaining <= len(prefix):
            return False
        if len(prefix) + len(text) > remaining:
            if blocks:
                return False
            text = text[:remaining - len(prefix)].rstrip()
        block = prefix + text
        blocks.append(block)
        used += len(block) + 2
        return True

    for number, group in enumerate(main_groups, 1):
        if add(str(number), group):
            main_done.append(group)
            label_of[group[0].doc_key] = number
    for number, group in enumerate(related_groups, 1):
        via = group[0].via or {}
        origin = label_of.get(via.get("from_doc"))
        note = f" (linked from record {origin} via '{via.get('field')}')" if origin else ""
        if add(f"R{number}", group, note):
            related_done.append(group)
    return "\n\n".join(blocks), main_done, related_done


def _source(group, related=False) -> dict:
    head = group[0]
    source = {"collection": head.collection, "document_id": head.primary_id, "name": head.display_name,
              "score": round(max(h.score for h in group), 4), "related": related}
    if related and head.via:
        source["via"] = {"from": list(head.via["from_doc"]), "field": head.via["field"],
                         "direction": head.via["direction"]}
    return source


SOURCE_CAP = 25


def _retrieval_query(question: str, history: list) -> str:
    """Use recent prior questions as a narrow anchor for elliptical follow-ups, never prior answers."""
    if not history:
        return question
    # A few prior questions resolve pronouns such as "which ones?" or "give me all"
    # even when the referenced topic appeared a turn or two earlier. The current
    # wording is kept last and clearly separate to limit retrieval drift.
    prior_questions = [turn["question"] for turn in history[-3:]]
    context = " | ".join(" ".join(item.split()) for item in prior_questions if item)
    return f"Follow-up context: {context}\nCurrent question (focus retrieval on this): {question}"[-2000:]


def _prior_list_reference(question: str, history: list) -> str | None:
    """Resolve a direct ordinal question from the previous answer's displayed list."""
    match = re.fullmatch(
        r"(?:what(?:'s| is| was)\s+(?:the\s+)?|which\s+(?:one\s+)?(?:was|is)\s+(?:the\s+)?|name\s+the\s+)"
        r"(first|second|third|last)(?:\s+(?:one|item|category|person|task|project))?[?.!]*",
        question.strip(), flags=re.IGNORECASE,
    )
    if not match or not history:
        return None

    items = []
    for line in history[-1]["answer"].splitlines():
        item = re.match(r"\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$", line)
        if item:
            items.append(item.group(1))
    if not items:
        return None

    ordinal = match.group(1).lower()
    index = {"first": 0, "second": 1, "third": 2, "last": -1}[ordinal]
    if index >= len(items):
        return None
    return items[index]


_EMPTY_BULLET_LINE = re.compile(
    r"^\s*(?:[•*+\-]|\d+[.)])\s*[\s,.;:•·\-\u00a0\u200b-\u200d\ufeff]*$"
)


def _strip_blank_bullet_lines(text: str) -> str:
    lines = []
    for line in unescape(str(text or "")).splitlines():
        if _EMPTY_BULLET_LINE.match(line):
            continue
        if not line.strip() and (not lines or not lines[-1].strip()):
            continue
        lines.append(line.rstrip())
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _remember(session_id, question, result, edit_index=None, user_id=None):
    if session_id:
        if edit_index is None:
            chat_memory.add(session_id, question, result["answer"], owner_id=user_id)
        else:
            chat_memory.replace(session_id, edit_index, question, result["answer"], owner_id=user_id)


def _plain_answer(exact) -> str:
    """The exact result without the LLM, used when the language model is unavailable."""
    head = (f"The language model is unavailable right now, so this is the plain result read directly from the "
            f"database: {exact.total} matching record(s) in '{exact.collection}' (conditions: {exact.conditions}).")
    if exact.operation == "count":
        return head
    items = "\n".join(f"- {r['name'] or r['text'][:160]}" for r in exact.records)
    more = "" if exact.complete else f"\n\nShowing the first {len(exact.records)} of {exact.total}."
    return f"{head}\n\n{items}{more}"


def _field_from_text(text: str, label: str) -> str:
    match = re.search(rf"\b{re.escape(label)}:\s*([^;]+)", text or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _unique_records(records: list) -> list:
    """Collapse duplicate display rows while keeping the richest copy."""
    unique, seen = [], {}
    for record in records or []:
        text = _strip_html(record.get("text", "") or "")
        name = _strip_html(str(record.get("name") or "")).strip()
        status = _field_from_text(text, "status").casefold()
        project = _field_from_text(text, "project").casefold()
        key = (("name", name.casefold(), status, project) if name
               else ("id", str(record.get("document_id") or text.casefold())))
        current = seen.get(key)
        if current is None:
            seen[key] = len(unique)
            unique.append(record)
        elif len(text) > len(_strip_html(unique[seen[key]].get("text", "") or "")):
            unique[seen[key]] = record
    return unique


_BLANK_TEXT = re.compile(r"[\u00a0\u200b-\u200d\ufeff\u2022\u00b7]+")
_PLACEHOLDER_VALUES = {"yes", "no", "true", "false", "none", "null", "n/a", "na", "-", "--"}


def _meaningful_field(value: str) -> bool:
    text = unescape(str(value or ""))
    text = _strip_html(text)
    text = _BLANK_TEXT.sub(" ", text)
    text = " ".join(text.split()).strip()
    if not re.search(r"[A-Za-z0-9]", text):
        return False
    return text.casefold() not in _PLACEHOLDER_VALUES


def _format_board_answer(exact) -> str:
    """Render board/card exact records as clean, scannable grouped markdown.

    No box-drawing (keeps output readable on any screen), no raw URLs, no LLM
    truncation — every matching record is shown once.
    """
    records = _unique_records(exact.records or [])
    if not records:
        return f"No {exact.collection} records found."

    def _status_of(text):
        m = re.search(r"\bstatus:\s*([^;]+)", text or "", re.IGNORECASE)
        return (m.group(1).strip() if m else "Unknown")

    STATUS_EMOJI = {
        "planning": "🟡", "plan": "🟡",
        "in-progress": "🔵", "in progress": "🔵", "in_progress": "🔵",
        "done": "🟢", "completed": "🟢", "complete": "🟢", "closed": "🟢",
        "blocked": "🔴",
    }

    def _emoji(status):
        s = str(status).strip().lower().replace("-", " ").replace("_", " ")
        return STATUS_EMOJI.get(s, "⚪")

    groups = {}
    for r in records:
        groups.setdefault(_status_of(r.get("text", "")), []).append(r)

    order = ["in progress", "planning", "done", "blocked"]

    def sort_key(status):
        s = str(status).strip().lower().replace("-", " ").replace("_", " ")
        try:
            return (order.index(s), s)
        except ValueError:
            return (99, s)

    lines = [f"📊 {len(records)} {exact.collection.title()} Records", ""]
    for status in sorted(groups, key=sort_key):
        items = groups[status]
        label = status.title() if status else "Other"
        lines.append(f"### {_emoji(status)} {label} ({len(items)})")
        lines.append("")
        for r in items:
            name = r.get("name") or r.get("document_id") or "Untitled"
            text = _strip_html(r.get("text", "") or "")
            fields = {}
            for part in text.split(";"):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    fields[k.strip().lower()] = v.strip()
            lines.append(f"**{_emoji(status)} {name}** — `{label}`")
            bullets = []

            def add_bullet(value):
                value = unescape(str(value or ""))
                value = _strip_html(value)
                value = _BLANK_TEXT.sub(" ", value)
                value = " ".join(value.split()).strip()
                if not _meaningful_field(value) or value.casefold() in {name.casefold(), label.casefold()}:
                    return
                bullet = f"• {value}"
                if bullet not in bullets:
                    bullets.append(bullet)

            for key in ("description", "scope", "next steps", "client", "tools", "deadline"):
                if key in fields:
                    add_bullet(fields[key])
            if not bullets:
                shown = 0
                for k, v in fields.items():
                    if k in {"status", "links", "title", "name", "document id", "document_id"}:
                        continue
                    before = len(bullets)
                    add_bullet(v)
                    if len(bullets) > before:
                        shown += 1
                    if shown >= 3:
                        break
            for b in bullets[:4]:
                lines.append(b)
            urls = re.findall(r"https?://[^\s;]+", text)
            if urls:
                labels = []
                for u in urls[:3]:
                    u = u.rstrip(",.;:)}]")
                    low = u.lower()
                    if "zoom.us" in low:
                        labels.append("[🎥 Meeting](" + u + ")")
                    elif "docs.google.com/document" in low:
                        labels.append("[📄 Doc](" + u + ")")
                    elif "figma.com" in low:
                        labels.append("[🎨 Figma](" + u + ")")
                    elif "swagger" in low:
                        labels.append("[🔌 API](" + u + ")")
                    elif "hubspot" in low:
                        labels.append("[⚙️ HubSpot](" + u + ")")
                    else:
                        labels.append("[🔗 " + urlparse(u).netloc + "](" + u + ")")
                lines.append("🔗 " + " · ".join(labels))
            lines.append("")
    return "\n".join(lines).rstrip()


def _clean_html_text(text: str) -> str:
    """Strip HTML tags while preserving linebreaks and decoding entities."""
    if not text or "<" not in text:
        return text
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text)
    s = re.sub(r"<[^>]+>", "", s)
    s = unescape(s)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.splitlines()]
    return "\n".join(line for line in lines if line)


def _exact_answer(question: str, exact, history=None) -> dict:
    """Answer from an exact database query (complete list / exact count)."""
    timings = dict(exact.timings)
    mark = time.perf_counter()
    degraded = False
    if exact.direct_answer:
        # Complete data (for example every logged-time entry) is returned byte-for-byte;
        # the LLM is never allowed to shorten it or drop rows.
        answer = _clean_html_text(_sanitize_ids(exact.direct_answer))
    elif exact.collection in {"boards", "cards"} and exact.records:
        # Render board/card records directly as clean grouped markdown: full data,
        # no LLM truncation, no raw URLs.
        answer = _sanitize_ids(_format_board_answer(exact))
    else:
        try:
            answer = llm_service.generate(question, exact.context, max_tokens=settings.LIST_MAX_TOKENS,
                                          history=history)
        except LLMError as exc:
            logger.warning("LLM unavailable for an exact result (%s); returning the plain result.", exc.message)
            answer, degraded = _plain_answer(exact), True
    answer = _strip_blank_bullet_lines(_sanitize_ids(answer))
    timings["llm"] = round((time.perf_counter() - mark) * 1000)
    not_found = not degraded and answer.lower().startswith(NOT_FOUND_ANSWER.lower().rstrip("."))
    visible_records = _unique_records(exact.records)
    sources = [] if not_found else [
        {"collection": exact.collection, "document_id": r["document_id"], "name": r["name"],
         "score": None, "related": False} for r in visible_records[:SOURCE_CAP]]
    result = {"question": question, "answer": answer, "sources": sources, "cached": False, "mode": "structured",
            "collection": exact.collection, "total_matching": exact.total, "records_used": len(visible_records),
            "conditions": exact.conditions, "list_mode": True, "degraded": degraded, "timings_ms": timings}
    return result


def ask(
    question,
    top_k=None,
    collections=None,
    hops=None,
    session_id=None,
    edit_index=None,
    user_id=None,
    email=None,
    history=None,
) -> dict:
    question = validate_question(question)
    if session_id is not None and (not isinstance(session_id, str) or not re.fullmatch(r"[A-Fa-f0-9-]{1,64}", session_id)):
        raise InvalidRequestError("'session_id' must be a valid session identifier.")
    scope = authorization_service.resolve_scope(user_id, email=email)
    chat_owner_id = scope.chat_owner_id

    # Self-identity questions are about the current account only, not another user's data,
    # so answer them before the mentions-another-user guard.
    self_identity = check_self_identity(question, scope)
    if self_identity is not None:
        result = {
            "question": question,
            "answer": self_identity,
            "sources": [],
            "cached": False,
            "mode": "self_identity",
            "timings_ms": {"total": 0}
        }
        _remember(session_id, question, result, edit_index, chat_owner_id)
        return result

    company_identity = check_company_identity(question, scope)
    if company_identity is not None:
        result = {
            "question": question,
            "answer": company_identity,
            "sources": [],
            "cached": False,
            "mode": "company_identity",
            "timings_ms": {"total": 0}
        }
        _remember(session_id, question, result, edit_index, chat_owner_id)
        return result

    if not scope.is_admin and (
        structured_query.is_person_lookup_question(question) or structured_query.is_person_attribute_question(question)
    ) and authorization_service.mentions_another_user(question, scope.user_id, scope.user_name):
        raise AuthorizationError("You are not authenticated , to access information")
    if session_id and not chat_memory.can_access(session_id, owner_id=chat_owner_id):
        raise InvalidRequestError("This chat session does not belong to your account.")
    if history is None:
        state = chat_memory.get_state(session_id, owner_id=chat_owner_id) if session_id else {
            "turns": [], "offset": 0,
        }
        history, history_offset = state["turns"], state["offset"]
    else:
        history = history[-chat_memory.MAX_TURNS:]
        history_offset = 0
    if edit_index is not None:
        if (not isinstance(edit_index, int) or isinstance(edit_index, bool)
                or not history_offset <= edit_index < history_offset + len(history)):
            raise InvalidRequestError(
                f"This turn is outside the {chat_memory.MAX_TURNS}-turn edit history. "
                "Refresh the chat and edit a more recent turn."
            )
        history = history[:edit_index - history_offset]

    conversational = check_conversational(question)
    if conversational is not None:
        result = {
            "question": question,
            "answer": conversational,
            "sources": [],
            "cached": False,
            "mode": "conversational",
            "timings_ms": {"total": 0}
        }
        _remember(session_id, question, result, edit_index, chat_owner_id)
        return result

    prior_item = _prior_list_reference(question, history)
    if prior_item is not None:
        ordinal = re.search(r"\b(first|second|third|last)\b", question, re.IGNORECASE).group(1).lower()
        result = {
            "question": question,
            "answer": f"The {ordinal} one listed was {prior_item}.",
            "sources": [],
            "cached": False,
            "mode": "history_reference",
            "timings_ms": {"total": 0},
        }
        _remember(session_id, question, result, edit_index, chat_owner_id)
        return result

    query_text = query_correction.normalize_query(question)
    query_text = query_correction.correct_query(strip_greeting(query_text))
    top_k = top_k or settings.TOP_K
    list_mode = bool(settings.LIST_TOP_K) and is_list_question(query_text)
    if list_mode:
        top_k = max(top_k, settings.LIST_TOP_K)     # never fewer than the user's mode, more for list questions
    cache_history = history
    normalized_question = question.casefold()
    # An immediate repeat should use the same context key as the original request.
    while (cache_history and
           " ".join(cache_history[-1]["question"].split()).casefold() == normalized_question):
        cache_history = cache_history[:-1]
    history_key = tuple((turn["question"], turn["answer"]) for turn in cache_history)
    cache_key = (scope.cache_key, session_id, normalized_question, top_k, tuple(sorted(collections or [])), hops,
                 history_key, vector_index.current_version())
    cached = _answer_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        _remember(session_id, question, result, edit_index, chat_owner_id)
        return result

    # Personal work queries use the verified account ID directly, so "my tasks"
    # does not depend on the name appearing in the question or on vector links.
    if not scope.user_id.startswith("guest:"):
        personal_work = structured_query.try_current_user_work(
            query_text, scope.user_id, history=history
        )
        if personal_work is not None:
            result = _exact_answer(question, personal_work, history)
            _remember(session_id, question, result, edit_index, chat_owner_id)
            return result

    # Leave-time questions (Full Leave / Short Leave / Halfday Leave logged time)
    # are read from nested loggedTime entries on leave task cards. Privacy is
    # enforced inside _leave_time_answer: a non-admin account only ever sees its
    # own logged time. Route these before the admin-only structured_candidate check.
    # Also routes superlative leave-count queries ("who takes most leaves").
    if not scope.user_id.startswith("guest:"):
        if (structured_query._LEAVE_TIME_QUESTION_PATTERN.search(query_text)
                or structured_query._LEAVE_COUNT_SUPERLATIVE_PATTERN.search(query_text)) and \
                (structured_query._LEAVE_TYPE_PATTERN.search(query_text)
                 or re.search(r"\bleaves?\b", query_text, re.IGNORECASE)):
            leave_time_exact = structured_query.try_structured(
                query_text,
                history=history,
                current_user_id=scope.user_id,
                is_admin=scope.is_admin,
            )
            if leave_time_exact is not None:
                result = _exact_answer(question, leave_time_exact, history)
                _remember(session_id, question, result, edit_index, chat_owner_id)
                return result

    # Leave types questions (e.g. "types of leaves", "what leave types exist")
    # scan card titles for configured leave types. No privacy restriction needed:
    # this only reports which leave task types exist, not any user's data.
    if not scope.user_id.startswith("guest:"):
        if structured_query._LEAVE_TYPES_PATTERN.search(query_text):
            leave_types_exact = structured_query.try_structured(query_text, history=history)
            if leave_types_exact is not None:
                result = _exact_answer(question, leave_types_exact, history)
                _remember(session_id, question, result, edit_index, chat_owner_id)
                return result

    # Logged-time / timesheet questions always read only nested `loggedTime` entries on
    # 'cards' and return every entry without LLM truncation. Privacy is enforced inside
    # _logged_time_answer: a non-admin account only ever sees its own logged time.
    logged_time_question = (
        structured_query.is_logged_time_question(query_text)
        or structured_query.is_logged_time_followup(query_text, history)
    )
    if logged_time_question:
        exact = structured_query.try_structured(
            query_text,
            history=history,
            current_user_id=None if scope.user_id.startswith("guest:") else scope.user_id,
            is_admin=scope.is_admin,
        )
        if exact is not None:
            result = _exact_answer(question, exact, history)
            _remember(session_id, question, result, edit_index, chat_owner_id)
            return result

    structured_candidate = (
        (settings.STRUCTURED_QUERIES or logged_time_question) and (not collections or logged_time_question)
        and not _multi_topic_leave_question(query_text) and scope.embedding_ids is None
        and (logged_time_question or (
            scope.is_admin
            and (list_mode or structured_query.mentions_collection(query_text)
                 or structured_query.is_person_lookup_question(query_text)
                 or structured_query.is_person_attribute_question(query_text))
        ))
    )
    cache_result = True
    if structured_candidate:
        exact = structured_query.try_structured(query_text, history=history)
        if exact is not None:
            result = _exact_answer(question, exact, history)
            if exact.source != "name-match":
                _answer_cache.set(cache_key, copy.deepcopy(result))
            _remember(session_id, question, result, edit_index, chat_owner_id)
            return result
        cache_result = False

    timings, started = {}, time.perf_counter()

    def mark(stage):
        nonlocal started
        now = time.perf_counter()
        timings[stage] = round((now - started) * 1000)
        started = now

    retrieval_query = query_correction.expand_query(_retrieval_query(query_text, history))
    hits = retrieval.search(retrieval_query, top_k=top_k, collections=collections,
                            allowed_ids=scope.embedding_ids, filter_key=scope.cache_key)
    mark("search")
    if not hits:
        fallback_msg = (
            f"{NOT_FOUND_ANSWER} You can ask me about team members, active projects, tasks, "
            "departments, logged hours, sales leads, or company announcements!"
        )
        result = {"question": question, "answer": fallback_msg, "sources": [], "cached": False,
                  "timings_ms": timings}
        _remember(session_id, question, result, edit_index, chat_owner_id)
        return result

    related = retrieval.expand(hits, hops=hops, allowed_ids=scope.embedding_ids)
    mark("expand")
    budget = max(settings.MAX_CONTEXT_CHARS, settings.LIST_CONTEXT_CHARS) if list_mode else settings.MAX_CONTEXT_CHARS
    context, main_done, related_done = build_context(
        _group(hits), _group(related), budget,
        include_logged_time=structured_query.wants_logged_time(question),
    )
    answer = llm_service.generate(question, context, max_tokens=settings.LIST_MAX_TOKENS if list_mode else None,
                                  history=history)
    answer = _strip_blank_bullet_lines(answer)
    mark("llm")

    is_pure_refusal = (
        len(answer.strip()) < 90 and
        answer.lower().startswith(NOT_FOUND_ANSWER.lower().rstrip("."))
    )
    sources = [] if is_pure_refusal else (
        [_source(g) for g in main_done] + [_source(g, related=True) for g in related_done])
    result = {"question": question, "answer": answer, "sources": sources, "cached": False,
              "records_used": len(main_done), "list_mode": list_mode, "timings_ms": timings}
    if cache_result:
        _answer_cache.set(cache_key, copy.deepcopy(result))
    _remember(session_id, question, result, edit_index, chat_owner_id)
    logger.info("ask: %d hits, %d related, timings=%s", len(hits), len(related), timings)
    return result


def search(query, top_k=None, collections=None, hybrid=None, expand=False, hops=None, user_id=None,
           email=None) -> dict:
    """Semantic (and keyword) search without the LLM."""
    query = validate_question(query)
    search_query = query_correction.expand_query(query_correction.correct_query(query))
    started = time.perf_counter()
    scope = authorization_service.resolve_scope(user_id, email=email)
    if not scope.is_admin and authorization_service.mentions_another_user(query, scope.user_id, scope.user_name):
        raise AuthorizationError("You are not authenticated , to access information")
    hits = retrieval.search(search_query, top_k=top_k, collections=collections, hybrid=hybrid,
                            allowed_ids=scope.embedding_ids, filter_key=scope.cache_key)
    result = {"query": query, "results": [h.to_dict() for h in hits]}
    if expand:
        result["related"] = [h.to_dict() for h in retrieval.expand(
            hits, hops=hops, allowed_ids=scope.embedding_ids)]
    result["timings_ms"] = {"total": round((time.perf_counter() - started) * 1000)}
    return result


def health() -> dict:
    from . import mongo_service

    info = {"mongo": mongo_service.ping(), "embedding_model": settings.EMBEDDING_MODEL_ID,
            "llm_model": settings.LLM_MODEL, "llm": llm_service.status(), "hybrid_search": settings.HYBRID_SEARCH,
            "cache": _answer_cache.stats(), "index": {"ready": False}, "embeddings": 0}
    if info["mongo"]:
        try:
            info["embeddings"] = embedding_store.count()
            index = vector_index.get_index()
            info["index"] = {"ready": True, "vectors": len(index.ids), "built_at": index.built_at}
        except ChatbotError as exc:
            info["index"] = {"ready": False, "message": exc.message}
    info["status"] = "ok" if info["mongo"] and info["index"]["ready"] else "degraded"
    return info


def warm_up():
    try:
        query_correction.prepare_vocabulary()
    except Exception as exc:
        logger.info("Query spelling warm-up skipped: %s", exc)
    try:
        structured_query.build_catalog()      # first list question is then fast
    except Exception as exc:
        logger.info("Catalog warm-up skipped: %s", exc)
    vector_index.get_index()
    if settings.EMBEDDING_PROVIDER == "local":
        embedding_service.get_model()
    try:
        llm_service.warm_up()            # loads a local Ollama model into memory (no-op for hosted APIs)
    except ChatbotError as exc:
        logger.info("LLM warm-up skipped: %s", exc.message)
