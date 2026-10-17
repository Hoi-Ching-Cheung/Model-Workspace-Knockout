"""The committed results must be reproducible from the committed code."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from jspace.grading import grade
from jspace.runner import ARMS, CONDITIONS, FIELDS

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["gsm8k", "math500", "aime24"]


def rows(name):
    with open(ROOT / "results" / f"{name}.csv") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("name", DATASETS)
def test_schema_and_labels(name):
    rs = rows(name)
    assert list(rs[0].keys()) == FIELDS
    assert {r["condition"] for r in rs} <= set(CONDITIONS)
    assert {r["arm"] for r in rs} <= set(ARMS)
    assert len({(r["id"], r["condition"], r["arm"]) for r in rs}) == len(rs)


@pytest.mark.parametrize("name", DATASETS)
def test_regrading_reproduces_scores(name):
    for r in rows(name):
        assert int(grade(r["pred"], r["gold"], r["dataset"])) == int(r["correct"]), r


@pytest.mark.parametrize("name", DATASETS)
def test_every_ablation_is_paired_with_clean(name):
    by_cell = {}
    for r in rows(name):
        by_cell.setdefault((r["arm"], r["condition"]), set()).add(r["id"])
    for (arm, cond), ids in by_cell.items():
        assert ids == by_cell[(arm, "clean")], (arm, cond)


def test_frozen_sample():
    s = json.loads((ROOT / "artifacts" / "sample_ids.json").read_text())
    assert len(s["gsm8k_test_indices"]) == 150
    assert len(s["aime24_ids"]) == 30
    levels = Counter(r["level"] for r in rows("math500")
                     if r["condition"] == "clean" and r["arm"] == "no_cot")
    assert levels == {str(l): 30 for l in range(1, 6)}
