"""Neutralize adversarial prompt injections hidden in transaction notes.

Four layers, applied in order:

  1. NORMALIZE  - strip the obfuscation an attacker uses to slip past a naive
                  regex: zero-width joiners, unicode confusables, RTL overrides,
                  leetspeak, runs of whitespace, chat-template control tokens.
  2. DETECT     - match a pattern library against the *normalized* text, so
                  "1gnore  prev1ous" is caught by the same rule as the plain form.
  3. NEUTRALIZE - replace the note wholesale rather than trying to excise the bad
                  span, and raise `injection_attempt`, which the risk model reads
                  as an aggravating fraud signal. An attacker trying to talk the
                  classifier down becomes evidence against their own transaction.
  4. CONTAIN    - `wrap_untrusted` fences whatever survives inside a nonce-tagged
                  block that never appears in the instruction region of a prompt.

Layer 4 plus schema-constrained decoding means even a missed injection cannot
change the *shape* of what the model emits - only, at worst, one record's verdict.
"""
from __future__ import annotations

import re
import secrets
import unicodedata
from dataclasses import dataclass

# Characters an attacker uses to break up a keyword without changing how it reads.
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD,
     0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069],
    None,
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Chat-template / control tokens that could open a new turn inside the context.
_CONTROL_TOKENS = re.compile(
    r"<\|[^|>]{0,40}\|>|<\/?(?:s|system|user|assistant|im_start|im_end)>|\[/?INST\]|###\s*(?:system|instruction)",
    re.IGNORECASE,
)

# A role prefix is an attack wherever it sits, not only at line start: normalization
# collapses newlines, so an attacker hiding "system:" behind a code fence or a pipe
# would otherwise slip past a `^`-anchored pattern.
_ROLE_PREFIX = re.compile(
    r"(?:^|[`~|\-*>\]\)\n])\s*(system|assistant|user|developer)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("override_instructions", re.compile(
        r"\b(ignore|disregard|forget|overrid\w*|bypass)\b[^.]{0,40}?"
        r"\b(previous|prior|above|earlier|all|any|everything|preceding)\b", re.IGNORECASE)),
    ("forget_everything", re.compile(r"\bforget\s+(everything|all)\b", re.IGNORECASE)),
    ("new_instructions", re.compile(r"\b(new|updated|revised)\s+(instructions?|rules?|prompt)\b", re.IGNORECASE)),
    ("role_reassignment", re.compile(r"\byou\s+are\s+(now|a|an)\b", re.IGNORECASE)),
    ("role_prefix", _ROLE_PREFIX),
    ("control_token", _CONTROL_TOKENS),
    ("force_verdict", re.compile(
        r"\b(classify|mark|treat|label|report|consider|flag)\b[^.]{0,40}?"
        r"\b(as\s+)?(safe|legitimate|legit|benign|valid|authorised|authorized|approved|not\s+fraud)\b", re.IGNORECASE)),
    ("suppress_flag", re.compile(r"\b(do\s*n[o']?t|never)\b[^.]{0,30}\b(flag|report|block|decline|escalat\w*)\b", re.IGNORECASE)),
    ("force_schema", re.compile(r"is_fraud\s*[=:]\s*(false|true|0|1)|confidence\s*[=:]\s*[01]?\.?\d+", re.IGNORECASE)),
    ("whitelist_claim", re.compile(r"\b(whitelist\w*|pre[- ]?approved|verified\s+merchant|internal\s+transfer)\b", re.IGNORECASE)),
    ("json_injection", re.compile(r"respond\s+only\s+with|output\s+(only|exactly)|return\s*\{", re.IGNORECASE)),
    ("instruction_fence", re.compile(r"#{3,}\s*\w+\s*#{3,}", re.IGNORECASE)),
]

REDACTION = "[REDACTED: prompt-injection attempt detected and neutralized]"
MAX_NOTE_CHARS = 240


@dataclass(frozen=True)
class SanitizedNote:
    """Outcome of sanitizing one note."""
    text: str                 # safe to place in a prompt
    injection_attempt: bool
    patterns: tuple[str, ...]
    original_len: int

    @property
    def pattern_str(self) -> str:
        return "|".join(self.patterns)


def normalize(raw: str) -> str:
    """Fold away obfuscation so detection sees the attacker's intent, not their spelling."""
    if not raw:
        return ""
    # NFKC maps confusables and fullwidth forms onto their ASCII equivalents.
    text = unicodedata.normalize("NFKC", str(raw)).translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    # Collapse the "I G N O R E   A L L" trick. Wider gaps separate words, so
    # split on them first and de-space only within each chunk - otherwise the
    # whole phrase fuses into one token and destroys the word boundaries that
    # the detection patterns rely on.
    chunks = re.split(r"[\s ]{2,}|\t|\n", text)
    collapsed = [re.sub(r"(?<=\b\w)[  ]+(?=\w\b)", "", c) for c in chunks]
    text = " ".join(c for c in collapsed if c)
    text = re.sub(r"[\s ]+", " ", text)
    return text.strip()


def _detect(text: str) -> tuple[str, ...]:
    """Match patterns against both the normalized and de-leetspeaked forms."""
    candidates = {text, text.translate(_LEET)}
    hits: list[str] = []
    for name, pattern in INJECTION_PATTERNS:
        if any(pattern.search(c) for c in candidates):
            hits.append(name)
    return tuple(hits)


def sanitize_note(raw: object) -> SanitizedNote:
    """Normalize, detect, and neutralize a single transaction note."""
    original = "" if raw is None else str(raw)
    text = normalize(original)
    if not text:
        return SanitizedNote("(no note)", False, (), len(original))

    patterns = _detect(text)
    if patterns:
        return SanitizedNote(REDACTION, True, patterns, len(original))

    # Even a clean note gets defanged: no control tokens, no role prefixes, bounded length.
    text = _CONTROL_TOKENS.sub(" ", text)
    text = _ROLE_PREFIX.sub(" ", text)
    text = re.sub(r"[{}<>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_NOTE_CHARS:
        text = text[:MAX_NOTE_CHARS] + "..."
    return SanitizedNote(text or "(no note)", False, (), len(original))


def wrap_untrusted(text: str, nonce: str | None = None) -> tuple[str, str]:
    """Fence untrusted text in a nonce-tagged block that the model is told to treat as data.

    The nonce is unguessable per-call, so a note cannot forge a closing tag and
    escape back into the instruction region.
    """
    nonce = nonce or secrets.token_hex(4)
    body = text.replace(nonce, "")
    return f"<<UNTRUSTED_DATA_{nonce}>>\n{body}\n<</UNTRUSTED_DATA_{nonce}>>", nonce
