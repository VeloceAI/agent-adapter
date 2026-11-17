import pytest

from agent_adapter.core.types import ApprovalRequest
from agent_adapter.voice.quality import (
    Currency,
    KnownNames,
    SpokenForm,
    UnsureWords,
    Word,
    number_words,
)
from agent_adapter.voice.quality._numbers import ordinal_words, words_number, year_words
from agent_adapter.voice.quality._unsure import words_of

US = SpokenForm()
IN = SpokenForm(locale="en-IN")


@pytest.mark.parametrize(
    ("n", "western", "indian"),
    [
        (0, "zero", "zero"),
        (4500, "four thousand five hundred", "four thousand five hundred"),
        (450000, "four hundred fifty thousand", "four lakh fifty thousand"),
        (12_500_000, "twelve million five hundred thousand", "one crore twenty-five lakh"),
        (-21, "minus twenty-one", "minus twenty-one"),
    ],
)
def test_number_words_in_both_scales(n, western, indian):
    assert number_words(n) == western
    assert number_words(n, indian=True) == indian


def test_ordinals_years_and_words_back_to_numbers():
    assert [ordinal_words(n) for n in (1, 2, 3, 12, 20, 21, 100)] == [
        "first", "second", "third", "twelfth", "twentieth", "twenty-first", "one hundredth",
    ]  # fmt: skip
    assert [year_words(y) for y in (2026, 2005, 1900, 1905)] == [
        "twenty twenty-six", "two thousand five", "nineteen hundred", "nineteen oh five",
    ]  # fmt: skip
    assert words_number(["forty", "five", "hundred"]) == 4500
    assert words_number(["one", "lakh", "twenty", "thousand"]) == 120_000
    assert words_number(["twenty-one"]) == 21
    assert words_number(["five", "apples"]) is None


@pytest.mark.parametrize(
    ("text", "us", "india"),
    [
        ("Pay ₹4,50,000 now.", "Pay four lakh fifty thousand rupees now.", None),
        ("Rs. 2,500 or Rs 99.50", "two thousand five hundred rupees or ninety-nine rupees "
         "and fifty paise", None),
        ("$12.50 and $1", "twelve dollars and fifty cents and one dollar", None),
        ("$3.5M, 30 USD, £5", "three point five million dollars, thirty dollars, "
         "five pounds", None),
        ("On 2026-11-18.", "On November eighteenth, twenty twenty-six.",
         "On eighteenth November twenty twenty-six."),
        ("Due 04/11/2026", "Due April eleventh, twenty twenty-six",
         "Due fourth November twenty twenty-six"),
        ("March 3rd, 2025 and 5 May.", "March third, twenty twenty-five and May fifth.",
         "third March twenty twenty-five and fifth May."),
        ("At 14:30, 9:05 am, 12:00 or 9 pm", "At two thirty P M, nine oh five A M, "
         "twelve o'clock or nine P M", None),
        ("Call +91 98765 43210", "Call plus nine one, nine eight seven six five, "
         "four three two one zero", None),
        ("3.5% of 1,250,000", "three point five percent of one million two hundred fifty thousand",
         "three point five percent of twelve lakh fifty thousand"),
        ("the 21st, 5 km, 1 km, 1 km/h", "the twenty-first, five kilometres, one kilometre, "
         "one kilometre per hour", None),
        ("in 2026, 10-20 days, No. 5, -4", "in twenty twenty-six, ten to twenty days, "
         "number five, minus four", None),
        ("Dr. Rao, e.g. code 007 & more", "Doctor Rao, for example code zero zero seven "
         "and more", None),
        ("COVID-19, v3.5.1, A320", "COVID-19, v3.5.1, A320", None),
    ],
)  # fmt: skip
def test_spoken_form(text, us, india):
    assert US(text) == us
    assert IN(text) == (india or us)


def test_own_currencies_read_before_and_after_the_amount():
    form = SpokenForm(currencies={"AED": Currency("dirham", "dirhams", "fils", "fils")})
    assert form("AED 1 or 20.50 AED") == "one dirham or twenty dirhams and fifty fils"


def test_pronunciation_dictionary_and_own_short_forms():
    form = SpokenForm(
        pronounce={"VoxCat": "vox cat", "SQL": "sequel"}, short_forms={"Apt.": "Apartment"}
    )
    assert (
        form("VoxCat speaks SQL, not sql, at Apt. 4")
        == "vox cat speaks sequel, not sql, at Apartment four"
    )


def test_known_names_fix_misheard_words_and_report_guesses():
    names = KnownNames(["Rahul", "Priyanka", "Acme Corp", "Hitesh"])
    assert names("Hi, I am Rahool from Acmy Corp.") == "Hi, I am Rahul from Acme Corp."
    assert [(f.heard, f.name) for f in names.last_fixes] == [
        ("Rahool", "Rahul"),
        ("Acmy Corp", "Acme Corp"),
    ]
    assert names("call priyanka") == "call Priyanka" and names.last_fixes == []  # capitals only
    assert names("the rail is late, well thanks") == "the rail is late, well thanks"
    assert names("Acmy, Corp") == "Acmy, Corp"  # punctuation splits a name
    assert KnownNames([])("Rahool") == "Rahool"


def test_words_of_reads_provider_word_lists():
    heard = words_of(
        [{"word": "a", "confidence": 0.5}, {"punctuated_word": "B.", "confidence": 1}, Word("c"), 7]
    )
    assert heard == [Word("a", 0.5), Word("B.", 1.0), Word("c"), Word("7")]
    assert words_of("two words") == [Word("two"), Word("words")]


def request(arguments, run_id="r1"):
    return ApprovalRequest(
        approval_id="a", run_id=run_id, call_id="c", tool="refund", arguments=arguments
    )


def test_unsure_numbers_and_names_in_the_arguments_are_read_back_once():
    unsure = UnsureWords()
    heard = [
        Word("Refund", 0.99),
        Word("4,500", 0.41),
        Word("to", 0.98),
        Word("Rahul.", 0.6),
        Word("Now", 0.5),
    ]
    found = unsure.hear(heard, run_id="r1")
    assert [(i.text, i.kind) for i in found] == [("4,500", "number"), ("Rahul", "name")]
    assert (
        unsure.readback(request({"amount": 4500.0, "user": "Rahul"}))
        == "Did you say 4,500 and Rahul?"
    )
    assert unsure.readback(request({"amount": 4500.0, "user": "Rahul"})) is None


def test_unsure_number_words_dates_and_arguments_that_do_not_match():
    unsure = UnsureWords()
    unsure.hear(
        [
            Word("forty", 0.5),
            Word("five", 0.9),
            Word("hundred", 0.9),
            Word("November", 0.4),
            Word("18", 0.99),
        ]
    )
    assert unsure.readback(request({"amount": 45})) is None  # 45 is not 4500
    line = unsure.readback(request({"amount": "4500", "on": "2026-11-18"}))
    assert line == "Did you say forty five hundred and November?"


def test_sure_words_guessed_names_and_options():
    unsure = UnsureWords(match_arguments=False, template="Just to check: {items}.")
    unsure.hear("send 4500 to Rahul")  # no confidence: taken as sure
    assert unsure.unsure() == []
    unsure.hear("pay Rahul", guessed=[("Rahul", 0.7)])
    assert unsure.readback(request({}, run_id="other")) == "Just to check: Rahul."
    numbers_only = UnsureWords(kinds=["number"])
    assert numbers_only.hear([Word("Rahul", 0.1)], guessed=[("Rahul", 0.5)]) == []
    aged = UnsureWords(max_age_s=-1)
    aged.hear([Word("12", 0.1)], run_id="r1")
    assert aged.unsure("r1") == []
    aged.clear("r1")
    assert aged.unsure("r1") == []
