"""Numbers as English words, both ways: Western (million) and Indian (lakh, crore) scales."""

from __future__ import annotations

ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
)  # fmt: skip
TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
WESTERN = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
INDIAN = ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand"))
_ORDINAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth",
}  # fmt: skip


def _below_100(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] + (f"-{ONES[ones]}" if ones else "")


def _below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    parts = [f"{ONES[hundreds]} hundred"] if hundreds else []
    if rest:
        parts.append(_below_100(rest))
    return " ".join(parts)


def number_words(n: int, *, indian: bool = False) -> str:
    """4500 -> "four thousand five hundred"; with `indian`, 450000 -> "four lakh fifty thousand"."""
    if n < 0:
        return "minus " + number_words(-n, indian=indian)
    if n < 1000:
        return _below_1000(n) if n else "zero"
    parts = []
    for size, word in INDIAN if indian else WESTERN:
        if n >= size:
            count, n = divmod(n, size)
            parts.append(f"{number_words(count, indian=indian)} {word}")
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def ordinal_words(n: int) -> str:
    """21 -> "twenty-first"."""
    words = number_words(n)
    head, sep, last = words.rpartition("-") if "-" in words[-8:] else words.rpartition(" ")
    if last in _ORDINAL:
        last = _ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def year_words(year: int) -> str:
    """2026 -> "twenty twenty-six", 2005 -> "two thousand five", 1900 -> "nineteen hundred"."""
    if not 1000 <= year <= 9999 or 2000 <= year <= 2009:
        return number_words(year)
    high, low = divmod(year, 100)
    if low == 0:
        return f"{_below_100(high)} hundred"
    return f"{_below_100(high)} {'oh ' + ONES[low] if low < 10 else _below_100(low)}"


def digit_words(digits: str) -> str:
    """ "007" -> "zero zero seven"."""
    return " ".join(ONES[int(d)] for d in digits if d.isdigit())


_VALUES = {word: i for i, word in enumerate(ONES)} | {
    word: i * 10 for i, word in enumerate(TENS) if i > 1
}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 10**5, "lakhs": 10**5, "crore": 10**7}
_SCALES |= {"crores": 10**7, "million": 10**6, "billion": 10**9}
NUMBER_WORDS = frozenset(_VALUES) | frozenset(_SCALES)


def words_number(words: list[str]) -> int | None:
    """["forty", "five", "hundred"] -> 4500; None when a word is not a number word."""
    total = current = 0
    seen = False
    for raw in words:
        for word in raw.lower().split("-"):
            if word in _VALUES:
                current += _VALUES[word]
            elif word == "hundred":
                current = (current or 1) * 100
            elif word in _SCALES:
                total += (current or 1) * _SCALES[word]
                current = 0
            elif word != "and":
                return None
            seen = True
    return total + current if seen else None
