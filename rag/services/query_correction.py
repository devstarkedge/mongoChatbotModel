"""Fast, conservative query repair using terms already present in indexed documents."""
import logging
import re
import threading
from collections import Counter, defaultdict

from . import embedding_store

logger = logging.getLogger(__name__)

_vocabulary = None
_by_length = None
_by_prefix = None
_lock = threading.Lock()
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")
_MASKED = re.compile(r"(?<!\w)([A-Za-z0-9][A-Za-z0-9*#?•]{2,})(?!\w)")
_MASK_CHARS = re.compile(r"[*#?•]+")
_MORPH_SUFFIXES = ("", "s", "es", "ed", "ing", "er", "ers", "ment", "ments", "tion", "tions")


def _distance_within(left: str, right: str, limit: int) -> int | None:
    """Return edit distance up to limit, treating a neighboring transposition as one edit."""
    if abs(len(left) - len(right)) > limit:
        return None
    previous_previous = None
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            value = min(current[-1] + 1, previous[j] + 1,
                        previous[j - 1] + (left_char != right_char))
            if (previous_previous is not None and i > 1 and j > 1
                    and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]):
                value = min(value, previous_previous[j - 2] + 1)
            current.append(value)
        previous_previous, previous = previous, current
    return previous[-1] if previous[-1] <= limit else None


def _load_vocabulary():
    global _vocabulary, _by_length, _by_prefix
    if _vocabulary is not None:
        return _vocabulary, _by_length, _by_prefix
    with _lock:
        if _vocabulary is None:
            counts = Counter()
            try:
                cursor = embedding_store.collection().find({}, {"text": 1})
                for doc in cursor:
                    counts.update(token.lower() for token in _TOKEN.findall(doc.get("text", ""))
                                  if len(token) >= 3)
            except Exception:
                logger.exception("Could not build the query spelling vocabulary")
            by_length = defaultdict(list)
            by_prefix = defaultdict(list)
            for word in counts:
                by_length[len(word)].append(word)
                by_prefix[word[:2]].append(word)
            _vocabulary, _by_length, _by_prefix = counts, dict(by_length), dict(by_prefix)
    return _vocabulary, _by_length, _by_prefix


def _best_spelling(word: str, vocabulary: Counter, by_length: dict, by_prefix: dict) -> str | None:
    limit = 2 if len(word) >= 7 else 1
    candidates = []
    seen = set()
    pools = [by_prefix.get(word[:2], ()),
             [candidate for size in range(len(word) - limit, len(word) + limit + 1)
              for candidate in by_length.get(size, ())]]
    for pool in pools:
        for candidate in pool:
            if candidate in seen:
                continue
            seen.add(candidate)
            if abs(len(word) - len(candidate)) > limit:
                continue
            distance = _distance_within(word, candidate, limit)
            if distance is not None:
                candidates.append((distance, candidate))
        # A one-edit match cannot be beaten; for two-edit matches, inspect the broader
        # length bucket in case another word is only one edit away.
        if candidates and min(distance for distance, _ in candidates) == 1:
            break
    if not candidates:
        return None
    nearest = min(distance for distance, _ in candidates)
    best = [candidate for distance, candidate in candidates if distance == nearest]
    if len(best) == 1:
        return best[0]
    # Frequency is only a tie breaker at the same edit distance; never prefer a farther word.
    ranked = sorted(((vocabulary[candidate], candidate) for candidate in best), reverse=True)
    if ranked[0][0] >= max(3, ranked[1][0] * 2):
        return ranked[0][1]
    return None


def _masked_candidates(token: str, vocabulary: Counter) -> list:
    masks = list(_MASK_CHARS.finditer(token))
    if not masks or (len(masks) == 1 and masks[0].group() == "?" and masks[0].end() == len(token)):
        return []
    pieces = _MASK_CHARS.split(token.lower())
    visible = "".join(pieces)
    if len(visible) < 2:
        return []
    prefix = pieces[0]
    suffix = pieces[-1]
    matches = [word for word in vocabulary
               if len(word) >= len(visible) and word.startswith(prefix) and word.endswith(suffix)]
    matches.sort(key=lambda word: (-vocabulary[word], abs(len(word) - len(token)), word))
    return matches[:3]


_NORMALIZATIONS = (
    (r"\bwat\b\s+is\b", "what is"),
    (r"\bwat\b(?!\s+is\b)", "what is"),
    (r"\bwhats\b(?!\s+is\b)", "what is"),
    (r"\bstatu(s)?\b", "status"),
    (r"\bbord(s)?\b", "board"),
    (r"\bloged\b", "logged"),
    (r"\bloging\b", "logging"),
    (r"\btoal\b", "total"),
    (r"\bestimaton\b", "estimation"),
    (r"\bestimations\b", "estimations"),
    (r"\bworkd\b", "worked"),
    (r"\bgogle\b", "Google"),
    (r"\bgsheet\b", "GSheet"),
    (r"\bwhatsapp\b", "WhatsApp"),
    (r"\bwhatsap\b", "WhatsApp"),
    (r"\bintregation\b", "integration"),
    (r"\bintregration\b", "integration"),
    (r"\bprogres\b", "progress"),
    (r"\bseo\b", "SEO"),
    (r"\bhr\b", "HR"),
    (r"\bit\b", "IT"),
)

_NORMALIZED = [(re.compile(p, re.IGNORECASE), r) for p, r in _NORMALIZATIONS]


def normalize_query(query: str) -> str:
    """Apply minimal, high-confidence fixes to spelling/grammar; preserve entities."""
    result = query
    for pattern, replacement in _NORMALIZED:
        result = pattern.sub(replacement, result)
    return result


def correct_query(query: str) -> str:
    """Repair clear typos and partially masked words using indexed corpus terms."""
    vocabulary, by_length, by_prefix = _load_vocabulary()
    if not vocabulary:
        return query
    masked_replacements = {}
    for match in _MASKED.finditer(query):
        token = match.group(1)
        candidates = _masked_candidates(token, vocabulary)
        if candidates:
            masked_replacements[token] = " ".join(candidates)

    def repair_mask(match):
        token = match.group(1)
        return masked_replacements.get(token, token)

    repaired = _MASKED.sub(repair_mask, query)

    def repair_spelling(match):
        original = match.group(0)
        word = original.lower()
        if len(word) < 4 or word in vocabulary:
            return original
        corrected = _best_spelling(word, vocabulary, by_length, by_prefix)
        if not corrected:
            return original
        return corrected.capitalize() if original[0].isupper() else corrected

    return _TOKEN.sub(repair_spelling, repaired)


def expand_query(query: str) -> str:
    """Add a few corpus-backed inflections so grammar differences match indexed wording."""
    vocabulary, _, _ = _load_vocabulary()
    if not vocabulary:
        return query
    additions = []
    seen = {token.lower() for token in _TOKEN.findall(query)}
    for match in _TOKEN.finditer(query):
        word = match.group(0).lower()
        if len(word) < 4:
            continue
        roots = {word}
        if word.endswith("ies") and len(word) > 5:
            roots.add(word[:-3] + "y")
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                roots.add(word[:-len(suffix)])
        forms = {root + suffix for root in roots for suffix in _MORPH_SUFFIXES if root + suffix in vocabulary}
        forms.discard(word)
        for form in sorted(forms, key=lambda item: (-vocabulary[item], item))[:2]:
            if form not in seen:
                additions.append(form)
                seen.add(form)
    return query if not additions else f"{query} {' '.join(additions)}"


def clear_vocabulary_cache():
    global _vocabulary, _by_length, _by_prefix
    with _lock:
        _vocabulary = None
        _by_length = None
        _by_prefix = None


def prepare_vocabulary():
    """Load corpus terms during background startup so the first user query stays responsive."""
    _load_vocabulary()
