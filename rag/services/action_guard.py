"""Deterministic warning gate for requests that would mutate or expose sensitive data."""
import re


_MUTATION = (
    r"(?:delete|deleting|remove|removing|erase|erasing|drop|dropping|truncate|truncating|wipe|wiping|"
    r"purge|purging|destroy|destroying|edit|editing|update|updating|modify|modifying|change|changing|"
    r"alter|altering|overwrite|overwriting|reset|resetting|grant|granting|revoke|revoking|"
    r"disable|disabling|enable|enabling|assign|assigning|transfer|transferring|promote|demote|making|make)"
)
_TARGET = (
    r"(?:all|every|each|entire|bulk|this|that|these|those|them|it|"
    r"data|records?|documents?|database|collections?|users?|employees?|accounts?|"
    r"tasks?|projects?|boards?|cards?|fields?|files?|titles?|dates?|descriptions?|status|entries|roles?|permissions?|access|"
    r"passwords?|credentials?|tokens?|secrets?|settings?)"
)
_ACTION_PREFIX = (
    r"(?:^\s*(?:please\s+)?|\b(?:please|can you|could you|would you|i want(?: you)? to|"
    r"i need(?: you)? to|go ahead and|execute|run|perform)\b.{0,80})"
)
_INSTRUCTION_PREFIX = (
    r"\b(?:how\s+(?:do|can|could)\s+i|how\s+to|steps?\s+to|instructions?\s+to|"
    r"write|generate|create|give me|provide)\b.{0,100}"
)
_CREDENTIAL_EXFILTRATION = re.compile(
    r"\b(?:show|reveal|export|download|dump|extract|send|print|list)\b.{0,60}\b"
    r"(?:passwords?|api[ _-]?keys?|tokens?|secrets?|credentials?|private[ _-]?keys?)\b"
    r"|\b(?:passwords?|api[ _-]?keys?|tokens?|secrets?|credentials?|private[ _-]?keys?)\b.{0,60}"
    r"\b(?:show|reveal|export|download|dump|extract|send|print|list)\b",
    re.IGNORECASE,
)
_SENSITIVE_EXPORT = re.compile(
    r"\b(?:show|reveal|export|download|dump|extract|send|print|list)\b.{0,80}\b"
    r"(?:personal information|employee data|user data|emails?|phone numbers?|salary|bank details|addresses)\b"
    r"|\b(?:personal information|employee data|user data|emails?|phone numbers?|salary|bank details|addresses)\b.{0,80}"
    r"\b(?:show|reveal|export|download|dump|extract|send|print|list)\b",
    re.IGNORECASE,
)
_PRIVILEGE_CHANGE = re.compile(
    rf"(?:{_ACTION_PREFIX}|{_INSTRUCTION_PREFIX})(?:give|grant|revoke|promote|demote|make|set)\b"
    r".{0,60}\b(?:admin|role|permission|access)\b",
    re.IGNORECASE,
)
_BULK_EXPORT = re.compile(
    r"\b(?:export|download|dump|extract)\b.{0,50}\b(?:all|every|entire|whole|bulk)\b.{0,50}"
    r"\b(?:data|records?|documents?|users?|employees?|accounts?|tasks?|projects?|collections?)\b"
    r"|\b(?:all|every|entire|whole|bulk)\b.{0,50}\b(?:data|records?|documents?|users?|employees?|accounts?|"
    r"tasks?|projects?|collections?)\b.{0,50}\b(?:export|download|dump|extract)\b",
    re.IGNORECASE,
)
_MUTATION_REQUEST = re.compile(
    rf"(?:{_ACTION_PREFIX}|{_INSTRUCTION_PREFIX}){_MUTATION}\b.{{0,100}}\b{_TARGET}\b"
    rf"|^\s*(?:all|every|each|entire|whole|bulk)\b.{{0,60}}\b{_TARGET}\b"
    rf".{{0,60}}\b(?:should|must|need to be)\s+{_MUTATION}\b",
    re.IGNORECASE,
)
_BULK = re.compile(r"\b(?:bulk|all|every|each|entire|whole|mass)\b", re.IGNORECASE)
_DESTRUCTIVE = re.compile(r"\b(?:delete|remove|erase|drop|truncate|wipe|purge|destroy)\w*\b", re.IGNORECASE)
_SECURITY = re.compile(
    r"\b(?:role|permission|access|admin|password|credential|token|secret|api[ _-]?key|security setting)\b",
    re.IGNORECASE,
)

WARNING_MESSAGE = (
    "Safety warning: this request is blocked. I can’t perform or provide instructions for deleting, "
    "editing, bulk-changing, or exposing sensitive company data or access settings. No data was changed. "
    "Use your organization’s approved, audited workflow for these actions."
)


def warning_for_request(text: str) -> dict | None:
    """Return a warning for action requests; ordinary explanations remain allowed."""
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return None
    if re.search(r"\bupdate me on\b", normalized, re.IGNORECASE):
        return None

    if _CREDENTIAL_EXFILTRATION.search(normalized) or _SENSITIVE_EXPORT.search(normalized):
        category = "sensitive_data"
    elif _BULK_EXPORT.search(normalized):
        category = "bulk_export"
    elif _PRIVILEGE_CHANGE.search(normalized):
        category = "sensitive_change"
    elif not _MUTATION_REQUEST.search(normalized):
        return None
    elif _BULK.search(normalized):
        category = "bulk_change"
    elif _DESTRUCTIVE.search(normalized):
        category = "destructive_change"
    elif _SECURITY.search(normalized):
        category = "sensitive_change"
    else:
        category = "data_change"

    return {"blocked": True, "category": category, "message": WARNING_MESSAGE}
