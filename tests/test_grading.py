import pytest

from jspace.grading import extract_answer, first_number, grade, last_boxed, normalize_answer


@pytest.mark.parametrize("pred, gold, dataset", [
    ("$26.00", "26", "gsm8k"),
    ("14 bags.", "14", "gsm8k"),
    ("8,000.", "8000", "gsm8k"),
    ("204", "204", "aime24"),
    ("\\boxed{4}", "4", "math500"),
    ("\\frac{1}{2}", "\\dfrac{1}{2}", "math500"),
    ("0.5", "\\frac{1}{2}", "math500"),
])
def test_correct(pred, gold, dataset):
    assert grade(pred, gold, dataset)


@pytest.mark.parametrize("pred, gold, dataset", [
    ("", "125", "gsm8k"),
    ("10 points.", "9", "gsm8k"),
    ("no idea", "42", "aime24"),
    ("\\frac{1}{3}", "\\frac{1}{2}", "math500"),
    ("", "\\frac{1}{2}", "math500"),
])
def test_incorrect(pred, gold, dataset):
    assert not grade(pred, gold, dataset)


@pytest.mark.xfail(strict=True, reason=(
    "known limitation, kept so the published results stay reproducible: "
    "math-verify parses unwrapped '(15, -29)' as just -29"))
def test_tuple_with_spaces():
    assert grade("(15, -29)", "(15,-29)", "math500")


def test_last_boxed_handles_nesting():
    assert last_boxed("x \\boxed{1} then \\boxed{\\frac{a}{b}}") == "\\frac{a}{b}"
    assert last_boxed("\\boxed{unclosed") is None
    assert last_boxed("nothing here") is None


def test_cot_prefers_box_over_marker():
    text = "Work...\n\n### Final Answer:\n$$\n\\boxed{\\frac{3}{4}}\n$$"
    assert extract_answer(text, cot=True) == "\\frac{3}{4}"


def test_cot_falls_back_to_marker_skipping_decoration():
    text = "So the total is 18.\n\n**Final answer:**\n$$\n18\n$$"
    assert extract_answer(text, cot=True) == "18"
    assert extract_answer("Final answer: 7 apples", cot=True) == "7 apples"


def test_cot_babble_extracts_nothing():
    assert extract_answer("the the the the the", cot=True) == ""


def test_no_cot_grades_only_first_line():
    assert extract_answer(" 42.\nActually wait, it's 43.", cot=False) == "42."
    assert extract_answer("   ", cot=False) == ""


def test_normalize_and_first_number():
    assert normalize_answer("$18.0$.") == "18"
    assert normalize_answer("\\left( x \\right)") == "(x)"
    assert first_number("$1,200.50 total") == 1200.5
    assert first_number("-3 degrees") == -3
    assert first_number("none") is None
