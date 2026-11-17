"""`SpokenForm`: text as a voice should read it: money, numbers, dates, times and short forms."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

from agent_adapter.voice.observers import CallObserver
from agent_adapter.voice.quality._numbers import (
    digit_words,
    number_words,
    ordinal_words,
    year_words,
)

if TYPE_CHECKING:
    from agent_adapter.voice.session import VoiceSession

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)  # fmt: skip
_MONTH_OF = {m.lower(): i for i, m in enumerate(MONTHS, 1)}
_MONTH_OF |= {m[:3].lower(): i for i, m in enumerate(MONTHS, 1)} | {"sept": 9}


class Currency(NamedTuple):
    """How to say an amount: "rupee", "rupees", "paisa", "paise", read in lakh and crore."""

    one: str
    many: str
    cent: str
    cents: str
    indian: bool = False


_RUPEE = Currency("rupee", "rupees", "paisa", "paise", True)
_DOLLAR = Currency("dollar", "dollars", "cent", "cents")
_EURO = Currency("euro", "euros", "cent", "cents")
_POUND = Currency("pound", "pounds", "penny", "pence")
CURRENCIES = {
    "₹": _RUPEE, "Rs": _RUPEE, "$": _DOLLAR, "USD": _DOLLAR,
    "€": _EURO, "EUR": _EURO, "£": _POUND, "GBP": _POUND,
}  # fmt: skip
SHORT_FORMS = {
    "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miz", "Prof.": "Professor",
    "e.g.": "for example", "i.e.": "that is", "etc.": "et cetera", "vs.": "versus",
    "approx.": "approximately", "Ltd.": "Limited", "Inc.": "Incorporated", "&": "and",
}  # fmt: skip
UNITS = {
    "km": "kilometres", "kg": "kilograms", "cm": "centimetres", "mm": "millimetres",
    "ml": "millilitres", "mins": "minutes", "min": "minutes", "hrs": "hours", "hr": "hours",
    "secs": "seconds", "sec": "seconds", "km/h": "kilometres per hour", "mph": "miles per hour",
    "GB": "gigabytes", "MB": "megabytes", "kWh": "kilowatt hours",
}  # fmt: skip
SCALE_WORDS = {"k": "thousand", "K": "thousand", "M": "million", "mn": "million", "bn": "billion"}
SCALE_WORDS |= {w: w for w in ("thousand", "lakh", "lakhs", "crore", "crores", "million")}
SCALE_WORDS["billion"] = "billion"


def _alternatives(words: object) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))  # type: ignore[call-overload]


_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SCALE = r"(?:\s?(k|K|M|mn|bn|thousand|lakhs?|crores?|million|billion)\b)?"
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_SLASH_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b")
_MONTHS = _alternatives(_MONTH_OF)
_YEAR = r"(?:,?\s+(\d{4}))?"
_MONTH_DAY = re.compile(rf"\b({_MONTHS})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b{_YEAR}", re.I)
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTHS})\b{_YEAR}", re.I)
_AMPM = r"([AaPp])\.?[Mm]\.?(?!\w)"
_TIME = re.compile(rf"\b(\d{{1,2}}):(\d{{2}})(?:\s?{_AMPM})?")
_HOUR_AMPM = re.compile(rf"\b(\d{{1,2}})\s?{_AMPM}")
_PHONE = re.compile(r"(?<![\w,.])(\+?)(\d[\d\s-]{6,}\d)(?![\w,.]*\d)")
_PERCENT = re.compile(rf"({_NUM})\s?%")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_UNIT = re.compile(rf"\b({_NUM})\s?({_alternatives(UNITS)})(?![\w/])")
_YEAR_AFTER = re.compile(r"\b(in|since|from|until|till|by|year)\s+(1[89]\d\d|20\d\d)\b", re.I)
_NUMBER_NO = re.compile(r"\bNo\.\s?(?=\d)")
_RANGE = re.compile(r"(?<![\w.-])(\d+(?:\.\d+)?)[-\u2013](\d+(?:\.\d+)?)(?![\w.-]*\d)")
_NUMBER = re.compile(rf"(?<![\w.-])(-)?({_NUM})(?![\w.]*\d)")
_SHORT_END = r"(?=[\s,;:)]|$)"


class SpokenForm(CallObserver):
    """Rewrites text before text-to-speech: "Rs 4,50,000" -> "four lakh fifty thousand rupees".

    Handles money (rupees with lakh and crore, dollars, euros, pounds), plain and decimal
    numbers, ranges, percentages, ordinals, dates, times, phone numbers (digit by digit),
    units and short forms. `locale="en-IN"` reads plain numbers in lakh and crore and slash
    dates day first; "en-US" reads millions and month first. `pronounce` maps a word (case
    sensitive) to how to say it: {"VoxCat": "vox cat", "SQL": "sequel"}. `currencies` adds
    symbols or codes: {"AED": Currency("dirham", "dirhams", "fils", "fils")}; a code of
    capital letters also reads after the amount ("30 USD").

    Call it on a string, or pass it as a `VoiceSession` observer: it then rewrites every line
    the session's player speaks. Pure Python; well under a millisecond a sentence.
    """

    def __init__(
        self,
        *,
        locale: str = "en-US",
        pronounce: Mapping[str, str] | None = None,
        short_forms: Mapping[str, str] | None = None,
        currencies: Mapping[str, Currency] | None = None,
    ) -> None:
        self.locale = locale
        self.indian = locale.lower().endswith("-in")
        self.pronounce = dict(pronounce or {})
        self.short_forms = SHORT_FORMS | dict(short_forms or {})
        self._pronounce = _word_pattern(self.pronounce, r"(?!\w)")
        self._short = _word_pattern(self.short_forms, _SHORT_END)
        self.currencies = CURRENCIES | dict(currencies or {})
        marks = _alternatives(self.currencies)
        codes = _alternatives(c for c in self.currencies if c.isalpha() and c.isupper())
        self._money_before = re.compile(rf"(?<![\w.])({marks})\.?\s?({_NUM}){_SCALE}")
        self._money_after = re.compile(rf"\b({_NUM}){_SCALE}\s?({codes})\b")

    def started(self, session: VoiceSession) -> None:
        hooks = getattr(session.speaker, "text_hooks", None)
        if isinstance(hooks, list) and self not in hooks:
            hooks.append(self)

    def __call__(self, text: str) -> str:
        if self._pronounce is not None:
            text = self._pronounce.sub(lambda m: self.pronounce[m[0]], text)
        text = self._money_before.sub(lambda m: self._money(m[1], m[2], m[3]), text)
        text = self._money_after.sub(lambda m: self._money(m[3], m[1], m[2]), text)
        text = _ISO_DATE.sub(lambda m: self._date(m[3], m[2], m[1]) or m[0], text)
        text = _SLASH_DATE.sub(self._slash_date, text)
        text = _MONTH_DAY.sub(lambda m: self._date(m[2], m[1], m[3]) or m[0], text)
        text = _DAY_MONTH.sub(lambda m: self._date(m[1], m[2], m[3]) or m[0], text)
        text = _TIME.sub(_time, text)
        text = _HOUR_AMPM.sub(lambda m: f"{number_words(int(m[1]))} {m[2].upper()} M", text)
        text = _PHONE.sub(_phone, text)
        text = _PERCENT.sub(lambda m: f"{self._number(m[1])} percent", text)
        text = _ORDINAL.sub(lambda m: ordinal_words(int(m[1])), text)
        text = _UNIT.sub(self._unit, text)
        text = _YEAR_AFTER.sub(lambda m: f"{m[1]} {year_words(int(m[2]))}", text)
        text = _NUMBER_NO.sub("number ", text)
        text = _RANGE.sub(lambda m: f"{self._number(m[1])} to {self._number(m[2])}", text)
        text = _NUMBER.sub(lambda m: ("minus " if m[1] else "") + self._number(m[2]), text)
        if self._short is not None:
            text = self._short.sub(lambda m: self.short_forms[m[0]], text)
        return text

    def _number(self, raw: str, *, indian: bool | None = None) -> str:
        whole, _, fraction = raw.replace(",", "").partition(".")
        if len(whole) > 1 and whole.startswith("0"):
            return digit_words(raw)  # "007" is a code, not an amount
        words = number_words(int(whole), indian=self.indian if indian is None else indian)
        return f"{words} point {digit_words(fraction)}" if fraction.strip("0") else words

    def _money(self, code: str, amount: str, scale: str | None) -> str:
        one, many, cent, cents, indian = self.currencies[code]
        if scale:
            return f"{self._number(amount, indian=indian)} {SCALE_WORDS[scale]} {many}"
        whole, _, fraction = amount.replace(",", "").partition(".")
        units = int(whole)
        said = f"{number_words(units, indian=indian)} {one if units == 1 else many}"
        hundredths = int((fraction + "00")[:2]) if fraction else 0
        if hundredths:
            said += f" and {number_words(hundredths)} {cent if hundredths == 1 else cents}"
        return said

    def _date(self, day: str, month: str, year: str | None) -> str | None:
        number = int(month) if month.isdigit() else _MONTH_OF[month.lower()]
        if not (1 <= number <= 12 and 1 <= int(day) <= 31):
            return None
        name, nth = MONTHS[number - 1], ordinal_words(int(day))
        said = f"{nth} {name}" if self.indian else f"{name} {nth}"
        if year:
            said += f"{'' if self.indian else ','} {year_words(int(year))}"
        return said

    def _slash_date(self, m: re.Match[str]) -> str:
        first, second, year = m[1], m[2], m[3]
        day_first = self.indian or int(first) > 12
        day, month = (first, second) if day_first else (second, first)
        return self._date(day, month, year if len(year) == 4 else f"20{year}") or m[0]

    def _unit(self, m: re.Match[str]) -> str:
        unit = UNITS[m[2]]
        if m[1] == "1":
            words = unit.split()
            plural = words.index("per") - 1 if "per" in words else len(words) - 1
            words[plural] = words[plural].removesuffix("s")
            unit = " ".join(words)
        return f"{self._number(m[1])} {unit}"


def _time(m: re.Match[str]) -> str:
    hour, minute, half = int(m[1]), int(m[2]), m[3]
    if hour > 23 or minute > 59:
        return m[0]
    if half is None and hour > 12:
        hour, half = hour - 12, "p"
    elif half is None and hour == 0:
        hour, half = 12, "a"
    said = number_words(hour)
    if minute:
        said += f" oh {number_words(minute)}" if minute < 10 else f" {number_words(minute)}"
    elif half is None:
        said += " o'clock"
    return f"{said} {half.upper()} M" if half else said


def _phone(m: re.Match[str]) -> str:
    if not m[1] and sum(c.isdigit() for c in m[2]) < 8:
        return m[0]
    groups = [digit_words(group) for group in re.split(r"[\s-]+", m[2]) if group]
    return ("plus " if m[1] else "") + ", ".join(groups)


def _word_pattern(words: Mapping[str, str], end: str) -> re.Pattern[str] | None:
    return re.compile(rf"(?<!\w)(?:{_alternatives(words)}){end}") if words else None
