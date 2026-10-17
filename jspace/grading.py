"""Answer extraction and grading.

GSM8K and AIME golds are integers: grade by numeric equality on the first
number in the prediction. MATH-500 golds are LaTeX: grade with math-verify's
symbolic equivalence, falling back to a normalised string match.
"""

import re

from math_verify import parse, verify


def last_boxed(text):
    """Contents of the last \\boxed{...}, brace-balanced; None if absent/unclosed."""
    i = text.rfind("\\boxed{")
    if i == -1:
        return None
    start = i + len("\\boxed{")
    depth = 1
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return None


def extract_answer(text, cot):
    """Pull the graded answer out of a generation.

    CoT: the last \\boxed{} wins. Qwen3 ignores the requested 'Final answer:'
    format on competition math and writes '### Final Answer' followed by a
    display-math block, so a marker-then-same-line rule misreads it. Fallback
    is the first non-decoration line after the last 'final answer' marker.
    Nothing found -> '' (a babbling ablated run should never grade correct).

    No-CoT: first line of the continuation only. The prefill forces the answer
    out first; anything after it is post-hoc and not graded.
    """
    if cot:
        boxed = last_boxed(text)
        if boxed is not None:
            return boxed.strip()
        hits = list(re.finditer(r"final answer\s*:?", text, flags=re.I))
        if hits:
            for line in text[hits[-1].end():].splitlines():
                core = line.strip(" $*#:.").strip()
                if core:
                    return core
        return ""
    head = text.strip()
    return head.splitlines()[0].strip() if head else ""


def normalize_answer(a):
    """Strip $, commas, \\boxed, \\left/\\right, whitespace; '18.0' -> '18'."""
    a = a.strip().rstrip(".").replace("$", "").replace(",", "")
    m = re.search(r"\\boxed\{(.*)\}", a)
    if m:
        a = m.group(1)
    a = a.replace("\\left", "").replace("\\right", "").replace(" ", "")
    try:
        f = float(a)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return a


def first_number(s):
    """First number in s as a float ('$1,200.50 total' -> 1200.5), else None."""
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _mv_parse(s):
    s = str(s).strip()
    return parse(s) or parse(f"${s}$")


def grade(pred, gold, dataset):
    if dataset in ("gsm8k", "aime24"):
        a, b = first_number(pred), first_number(gold)
        return a is not None and a == b
    try:
        g, p = _mv_parse(gold), _mv_parse(pred)
        if g and p:
            return bool(verify(g, p))
    except Exception:
        pass
    return normalize_answer(pred) == normalize_answer(gold)
