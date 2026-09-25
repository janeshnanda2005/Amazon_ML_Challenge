"""
Normalization for business names and addresses.

Goal: collapse superficial noise (case, punctuation, common abbreviations)
*before* similarity scoring and blocking, so that features and blocking keys
reflect real disagreement rather than formatting differences.

Kept deliberately conservative and rule-based (no external lookups, per the
challenge's fair-play rules) - just deterministic string rewriting.
"""

import re

# Common business-name abbreviation -> expansion. Expand rather than
# contract, since expansions are more standardized than the many possible
# abbreviated spellings.
NAME_ABBREVIATIONS = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "plc": "public limited company",
    "&": "and",
}

# Common address-token abbreviation -> expansion.
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "no": "number",
    "nr": "near",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _expand_tokens(tokens, abbrev_map):
    return [abbrev_map.get(tok, tok) for tok in tokens]


def normalize_name(raw_name: str) -> str:
    """Lowercase, strip punctuation, expand legal-suffix abbreviations."""
    cleaned = _basic_clean(raw_name)
    tokens = cleaned.split()
    tokens = _expand_tokens(tokens, NAME_ABBREVIATIONS)
    return " ".join(tokens)


def normalize_address(raw_address: str) -> str:
    """Lowercase, strip punctuation, expand common address abbreviations."""
    cleaned = _basic_clean(raw_address)
    tokens = cleaned.split()
    tokens = _expand_tokens(tokens, ADDRESS_ABBREVIATIONS)
    return " ".join(tokens)


def tokenize(text: str) -> list:
    """Whitespace tokenize an already-normalized string."""
    if not text:
        return []
    return text.split()


def core_name_tokens(raw_name: str, generic_tokens: set) -> list:
    """
    Tokens of a normalized name with generic/legal-suffix tokens removed -
    used for blocking so that e.g. "ABC Corporation" and "ABC Company"
    still share the discriminative token "abc".
    """
    tokens = tokenize(normalize_name(raw_name))
    core = [t for t in tokens if t not in generic_tokens]
    # Never return an empty token list if everything was generic - fall
    # back to the full token list so the record still gets a blocking key.
    return core if core else tokens


def extract_postal_token(raw_address: str) -> str:
    """
    Best-effort extraction of a postal/PIN code from an address string:
    the last standalone run of 4-6 digits. Returns "" when none is found
    (missing PIN codes are an expected noise pattern per the challenge).
    """
    if not raw_address:
        return ""
    matches = re.findall(r"\b\d{4,6}\b", str(raw_address))
    return matches[-1] if matches else ""
