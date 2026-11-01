"""Make summariser output inert: drop lines that read as orders, quote the rest as a record."""

from __future__ import annotations

import re
import unicodedata

REMOVED_LINE = "[removed: a line that read as an instruction to the assistant]"
RECORD_OPEN = "<summary-record>"
RECORD_CLOSE = "</summary-record>"
QUOTE = "> "
# Over this share of neutralised lines, the summariser is taken to have obeyed the data.
MAX_REMOVED_SHARE = 0.5

# Look-alike letters (Cyrillic, Greek) folded to Latin before matching.
_HOMOGLYPHS = str.maketrans(
    "\u0430\u0435\u043e\u0440\u0441\u0443\u0445\u0456\u0458\u0455\u0501\u051b\u051d"  # Cyrillic
    "\u03bf\u03b1\u03b5\u03b9\u03ba\u03bd\u03c1\u03c4\u03c5\u03c7",  # Greek
    "aeopcyxijsdqwoaeiknptux",
)
# "I G N O R E", "i.g.n.o.r.e": single letters split by one separator are joined again.
_SPACED = re.compile(r"\b(?:[a-z][\s._*\-]){3,}[a-z]\b")
_LEAD = re.compile(r"^[\s>*_#`~|\-+.\d)(\[\]\"']*")  # bullets, quotes, headings, numbering

_ROLE = (
    r"(?:system|assistant|user|developer|tool|human|ai|model|bot|admin|administrator|root|operator)"
)
# Each family is one way a line can carry an order; the tests cover every family.
_FAMILIES = (
    # 1. override earlier context: "ignore previous instructions", "disregard the guidance"
    r"\b(?:ignore|disregard|forget|override|bypass|discard|abandon|neglect)\b.{0,60}"
    r"\b(?:instructions?|rules?|prompts?|directions?|guidance|guidelines?|polic(?:y|ies)|"
    r"constraints?|above|earlier|previous|prior|preceding)\b",
    # 2. standing orders: "from now on", "henceforth"
    r"\b(?:from now on|from this point|starting (?:now|immediately|today)|henceforth|"
    r"going forward|for the rest of (?:the|this))\b",
    # 3. a role label opening the line: "Assistant:", "[system]", "### System"
    rf"^{_ROLE}(?:\s+(?:message|prompt|note|override|says?))?\s*[:>\]]",
    rf"^\[\s*{_ROLE}\s*\]",
    # 3b. a line labelled as a rule: "Rule: approve refunds", "Instruction 2 - obey"
    r"^(?:new\s+)?(?:rules?|polic(?:y|ies)|instructions?|directives?|orders?|commands?|"
    r"requirements?|mandatory|important)\s*\d*\s*[:\-]",
    # 4. claimed authority: "SYSTEM OVERRIDE", "admin notice", "developer mode"
    r"\b(?:system|admin|administrator|developer|root|security|emergency|priority|sudo)\s+"
    r"(?:override|update|notice|instructions?|command|directive|mode|message|prompt|alert)\b",
    r"\b(?:jailbreak|dan mode|god mode|unrestricted mode)\b",
    # 5. a new identity: "you are now", "act as", "new instructions"
    r"\byou are (?:now|no longer|actually|hereby)\b",
    r"\b(?:act|behave|respond|roleplay|role-play) as\b|\bpretend (?:to be|you|that)\b",
    r"\bnew (?:system )?(?:instructions?|rules?|role|persona|prompt|directive)\b",
    # 6. orders addressed to the assistant: "you must", "always approve"
    r"\byou\s+(?:must|should|shall|will|need to|have to|ought to|may not|cannot|can't|"
    r"are (?:required|instructed|ordered|allowed|permitted|to))\b",
    r"\b(?:always|never)\s+(?:approve|deny|reveal|call|run|execute|reply|respond|answer|say|"
    r"tell|ask|send|share|follow|obey|ignore|refuse|grant|trust|use)\b",
    # 7. an order opening the line: "Approve the refund", "Reveal the prompt"
    r"^(?:please\s+)?(?:ignore|disregard|forget|override|bypass|reveal|leak|disclose|approve|"
    r"authori[sz]e|grant|pretend|obey|comply|exfiltrate|do not|don't|stop)\b",
    # 8. asks for hidden data: "reveal the system prompt", "print your api key"
    r"\b(?:reveal|print|repeat|leak|disclose|send|exfiltrate|dump)\b.{0,40}"
    r"\b(?:system prompt|your (?:instructions|prompt|rules)|passwords?|secrets?|api ?keys?|"
    r"credentials?)\b",
    # 9. fake delimiters: "</data>", "<|im_start|>", "[INST]"
    rf"</?\s*(?:{_ROLE}|instructions?|data|summary-record|reference|im_start|im_end)\b",
    r"<\|[a-z_]+\|>|\[/?(?:inst|sys)\]|<<\s*/?sys\s*>>",
    # 10. the summariser speaking as the assistant: "I will comply"
    r"^(?:i|we)\s+(?:will|shall|must|am going to|have to)\s+(?:now\s+)?"
    r"(?:comply|obey|approve|ignore|follow|do (?:it|so|that))\b",
    # 11. a call to act at once: "call delete_all immediately", "pay without asking"
    r"\b(?:call|invoke|execute|trigger)\b.{0,40}\b(?:immediately|right away|at once)\b",
    r"\bwithout (?:asking|confirmation|approval|permission|checking)\b",
    # 12. an approval the summary has no power to record: "approval granted"
    r"\b(?:approval|authori[sz]ation|permission|consent)\s+(?:is\s+|was\s+|has been\s+)?"
    r"(?:granted|given|confirmed|approved)\b",
)
_ORDER = re.compile("|".join(f"(?:{family})" for family in _FAMILIES))


def normalised(line: str) -> str:
    """`line` folded for matching: NFKC, no invisible characters, Latin letters, lower case."""
    text = unicodedata.normalize("NFKC", line)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = text.casefold().translate(_HOMOGLYPHS)
    text = _SPACED.sub(lambda m: re.sub(r"[\s._*\-]", "", m.group(0)), text)
    text = re.sub(r"\s+", " ", text)
    return _LEAD.sub("", text).strip()


def reads_as_order(line: str) -> bool:
    """True when `line` reads as an instruction to the assistant rather than as a record."""
    return bool(_ORDER.search(normalised(line)))


def clean_lines(text: str) -> tuple[str, int, int]:
    """`text` with order-like lines replaced; also how many lines were removed, of how many."""
    lines = text.splitlines()
    kept = [REMOVED_LINE if reads_as_order(line) else line for line in lines]
    removed = sum(1 for old, new in zip(lines, kept, strict=True) if new is REMOVED_LINE)
    return "\n".join(kept), removed, sum(1 for line in lines if line.strip())


def as_record(header: str, body: str) -> str:
    """`body` quoted line by line inside a record block under `header`; no tags can close it."""
    escaped = body.replace("<", "&lt;").replace(">", "&gt;")
    quoted = "\n".join(f"{QUOTE}{line}" if line else QUOTE.rstrip() for line in escaped.split("\n"))
    return f"{header}\n{RECORD_OPEN}\n{quoted}\n{RECORD_CLOSE}"
