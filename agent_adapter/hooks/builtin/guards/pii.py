"""PIIGuard: find emails, phone numbers, cards, IBANs and secrets with regexes; no model needed."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from agent_adapter.core.errors import ConfigError
from agent_adapter.hooks.builtin.guardrail import Stage
from agent_adapter.hooks.builtin.guards.base import DEFAULT_STAGES, Action, GuardVerdict, TextGuard
from agent_adapter.hooks.builtin.logging import is_secret_field

PIIKind = Literal["email", "phone", "card", "iban", "secret"]
PII_KINDS: tuple[str, ...] = ("email", "phone", "card", "iban", "secret")

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
)
# A run of digits with spaces, dots, dashes or brackets; card or phone is decided by its digits.
_NUMBER = re.compile(r"(?<![\w+])\+?\(?\d(?:[ .()-]{0,2}\d)*(?!\w)")
_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_DIGITS = re.compile(r"\d+")
# Where an IBAN may start: a country code and two check digits, any case.
_IBAN_START = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{2}")
_IBAN_MIN, _IBAN_MAX = 15, 34
_SECRETS = (
    r"\bAKIA[0-9A-Z]{16}\b",  # cloud access key id
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",  # code-host token
    r"\bxox[abposr]-[A-Za-z0-9-]{10,}",  # chat-app token
    r"\bsk-[A-Za-z0-9_-]{20,}",  # model-provider key
    r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}",  # payment-provider key
    r"\bAIza[0-9A-Za-z_-]{35}\b",
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i:\b(?:api[_-]?key|secret|token|passw(?:or)?d|passcode)\b)[\"']?\s*[:=]\s*[\"']?[^\s\"']{8,}",
)
# Secret words the log redaction does not cover: one-time and card codes.
_MORE_SECRET_WORDS = frozenset(
    {"passcode", "passcodes", "pin", "pincode", "otp", "totp", "cvv", "cvc"}
)
_NAME_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SECRET = re.compile("|".join(f"(?:{p})" for p in _SECRETS))


@dataclass(frozen=True)
class PIIMatch:
    """One finding: its kind and where it sits in the text."""

    kind: str
    start: int
    end: int


def luhn_ok(digits: str) -> bool:
    """True when a digit string passes the Luhn checksum used by payment cards."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2:
            number = number * 2 - 9 if number > 4 else number * 2
        total += number
    return bool(digits) and total % 10 == 0


def iban_ok(text: str) -> bool:
    """True when an IBAN-like id passes its mod-97 check."""
    compact = text.replace(" ", "").upper()
    if not compact.isascii() or not compact.isalnum():
        return False
    moved = compact[4:] + compact[:4]
    number = "".join(str(int(char, 36)) for char in moved)
    return int(number) % 97 == 1


def find_pii(
    text: str,
    kinds: Iterable[str] = PII_KINDS,
    *,
    patterns: Mapping[str, re.Pattern[str]] | None = None,
) -> list[PIIMatch]:
    """Every finding in `text` of the given kinds (plus `patterns`), in order, not overlapping."""
    wanted = set(kinds)
    found: list[PIIMatch] = []
    if "secret" in wanted:
        found += [PIIMatch("secret", m.start(), m.end()) for m in _SECRET.finditer(text)]
    if "email" in wanted:
        found += [PIIMatch("email", m.start(), m.end()) for m in _EMAIL.finditer(text)]
    if "iban" in wanted:
        found += list(_ibans(text))
    if wanted & {"card", "phone"}:
        found += [match for m in _NUMBER.finditer(text) for match in _numbers(m, wanted)]
    for kind, pattern in (patterns or {}).items():
        found += [PIIMatch(kind, m.start(), m.end()) for m in pattern.finditer(text) if m[0]]
    return _first_wins(found)


def redact_pii(text: str, matches: Iterable[PIIMatch], placeholder: str = "[{kind}]") -> str:
    """`text` with each match replaced by `placeholder` (`{kind}` is the upper-case kind)."""
    parts: list[str] = []
    last = 0
    for match in sorted(matches, key=lambda m: m.start):
        parts += [text[last : match.start], placeholder.format(kind=match.kind.upper())]
        last = match.end
    return "".join(parts) + text[last:]


class PIIGuard(TextGuard):
    """Blocks, redacts or approves payloads holding personal data or secrets (regexes only)."""

    name = "pii_guard"

    def __init__(
        self,
        *,
        kinds: Iterable[str] = PII_KINDS,
        patterns: Mapping[str, str | re.Pattern[str]] | None = None,
        on: Iterable[Stage] | Stage = DEFAULT_STAGES,
        action: Action = "block",
        placeholder: str = "[{kind}]",
    ) -> None:
        super().__init__(on=on, action=action)
        chosen = tuple(dict.fromkeys(kinds))
        unknown = sorted(set(chosen) - set(PII_KINDS))
        if unknown:
            raise ConfigError(f"unknown PII kinds {unknown}; choose from {list(PII_KINDS)}")
        try:
            compiled = {
                kind: re.compile(p) if isinstance(p, str) else p
                for kind, p in (patterns or {}).items()
            }
        except re.error as exc:
            raise ConfigError(f"PIIGuard pattern does not compile: {exc.msg}") from None
        if not chosen and not compiled:
            raise ConfigError("PIIGuard needs at least one kind or pattern")
        try:
            placeholder.format(kind="KIND")
        except (KeyError, IndexError, ValueError):
            raise ConfigError("placeholder may only use the {kind} field") from None
        self.kinds = chosen
        self.patterns = compiled
        self.placeholder = placeholder

    async def review(self, text: str, stage: str) -> GuardVerdict:
        matches = find_pii(text, self.kinds, patterns=self.patterns)
        if not matches:
            return GuardVerdict(True)
        found = tuple(sorted({m.kind for m in matches}))
        redacted = redact_pii(text, matches, self.placeholder)
        return GuardVerdict(
            False, f"found {', '.join(found)}", redacted, kinds=found, count=len(matches)
        )

    def review_field(self, keys: tuple[str, ...], text: str) -> GuardVerdict | None:
        if "secret" in self.kinds and any(secret_key(key, text) for key in keys):
            secret = self.placeholder.format(kind="SECRET")
            return GuardVerdict(False, "found secret", secret, kinds=("secret",), count=1)
        return None


def secret_key(name: str, value: str) -> bool:
    """True when a key name says its value (or everything below it) is a secret."""
    if is_secret_field(name, value):
        return True
    return any(word.lower() in _MORE_SECRET_WORDS for word in _NAME_WORDS.findall(name))


def _numbers(match: re.Match[str], wanted: set[str]) -> list[PIIMatch]:
    """Cards in a run of digit groups (any span of whole groups), else the run as a phone."""
    raw = match[0]
    digits = re.sub(r"\D", "", raw)
    if "card" in wanted:
        cards = _cards(match)
        if cards:
            return cards
    if "phone" in wanted and 10 <= len(digits) <= 15 and _phone_shaped(raw, digits):
        return [PIIMatch("phone", match.start(), match.end())]
    return []


def _cards(match: re.Match[str]) -> list[PIIMatch]:
    """Luhn-valid spans of 13-19 digits made of whole groups, longest first, left to right."""
    groups = [(m.start(), m.end(), m[0]) for m in _DIGITS.finditer(match[0])]
    found: list[PIIMatch] = []
    first = 0
    while first < len(groups):
        end = _card_end(groups, first)
        if end is None:
            first += 1
            continue
        start, stop = match.start() + groups[first][0], match.start() + groups[end - 1][1]
        found.append(PIIMatch("card", start, stop))
        first = end
    return found


def _card_end(groups: list[tuple[int, int, str]], first: int) -> int | None:
    for end in range(len(groups), first, -1):
        digits = "".join(group[2] for group in groups[first:end])
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            return end
    return None


def _ibans(text: str) -> Iterator[PIIMatch]:
    """IBAN-like ids: from each start, the longest stretch (single spaces allowed) that checks."""
    for start in _IBAN_START.finditer(text):
        compact, index = start[0], start.end()
        ends: list[tuple[int, int]] = []
        while index < len(text) and len(compact) < _IBAN_MAX:
            char = text[index]
            if char.isascii() and char.isalnum():
                compact += char
                index += 1
                ends.append((len(compact), index))
            elif char == " " and index + 1 < len(text) and text[index + 1].isalnum():
                index += 1
            else:
                break
        for length, end in reversed(ends):
            if length >= _IBAN_MIN and iban_ok(compact[:length]):
                yield PIIMatch("iban", start.start(), end)
                break


def _phone_shaped(raw: str, digits: str) -> bool:
    """A leading + or separators, and not a decimal amount or an IPv4 address."""
    if raw == digits or _IPV4.fullmatch(raw):
        return False
    return not (raw.count(".") == 1 and raw.replace(".", "") == digits)


def _first_wins(found: list[PIIMatch]) -> list[PIIMatch]:
    """Drop findings that overlap an earlier-starting (or longer) one."""
    kept: list[PIIMatch] = []
    for match in sorted(found, key=lambda m: (m.start, -(m.end - m.start))):
        if not kept or match.start >= kept[-1].end:
            kept.append(match)
    return kept
