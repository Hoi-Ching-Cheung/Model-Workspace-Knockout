"""Runner bookkeeping, with the model calls stubbed out."""

import csv
from contextlib import contextmanager

import pytest
import torch

from jspace import runner
from jspace.generation import LoopStop
from jspace.runner import FIELDS, RunConfig, run_grid

BANDS = {"light": [27, 28], "medium": [26, 27, 28, 29], "heavy": list(range(24, 33))}


class RecordingAblator:
    def __init__(self):
        self.calls = []

    @contextmanager
    def ablate(self, band, mode, exclude_ids=None, k=None):
        self.calls.append((tuple(band), mode, k))
        yield


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(runner, "build_prompt", lambda tok, q, cot: q)
    monkeypatch.setattr(runner, "clean_exclude_ids", lambda *a: [0, 1, 2])
    monkeypatch.setattr(runner, "generate",
                        lambda m, tok, p, cap: (f"Final answer: {p[-1]}", 5, False, False))
    return RunConfig(hf_model=None, tokenizer=None, ablator=RecordingAblator(), bands=BANDS)


DATA = [{"dataset": "gsm8k", "id": f"gsm8k/{i}", "question": f"q{i}", "answer": str(i)}
        for i in range(4)]


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def test_rows_written_and_graded(tmp_path, cfg):
    out = tmp_path / "r.csv"
    run_grid(DATA, "clean", "cot", out, cfg)
    rows = read(out)
    assert list(rows[0].keys()) == FIELDS
    assert [r["correct"] for r in rows] == ["1"] * 4
    assert {r["config"] for r in rows} == {"clean"}


def test_resume_skips_finished_rows(tmp_path, cfg):
    out = tmp_path / "r.csv"
    run_grid(DATA[:2], "heavy", "no_cot", out, cfg)
    run_grid(DATA, "heavy", "no_cot", out, cfg)
    run_grid(DATA, "heavy", "no_cot", out, cfg)
    assert sorted(r["id"] for r in read(out)) == sorted(d["id"] for d in DATA)
    assert len(cfg.ablator.calls) == 4


@pytest.mark.parametrize("condition, band, mode", [
    ("clean", (), "none"),
    ("light", (27, 28), "jspace"),
    ("heavy", tuple(range(24, 33)), "jspace"),
    ("random", tuple(range(24, 33)), "random"),   # control sits on the heavy band
])
def test_condition_routing(tmp_path, cfg, condition, band, mode):
    run_grid(DATA[:1], condition, "cot", tmp_path / "r.csv", cfg)
    assert cfg.ablator.calls == [(band, mode, 25)]


def test_unknown_condition_rejected(tmp_path, cfg):
    with pytest.raises(ValueError):
        run_grid(DATA, "direct", "cot", tmp_path / "r.csv", cfg)


def test_loopstop():
    loop = torch.tensor([list(range(10)) * 100])          # 1000 tokens, period 10
    fresh = torch.arange(1000).unsqueeze(0)
    assert LoopStop()(loop, None)
    assert not LoopStop()(fresh, None)
    assert not LoopStop()(loop[:, :500], None)              # under min_len


class ByteTokenizer:
    """Just enough tokenizer for the real prompt -> generate -> grade path."""
    eos_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, **kw):
        return "".join(m["content"] for m in msgs)

    def __call__(self, text, return_tensors="pt", **kw):
        from transformers import BatchEncoding
        return BatchEncoding({"input_ids": torch.tensor([[1 + b % 63 for b in text.encode()]])})

    def decode(self, ids, **kw):
        return "".join(chr(48 + int(i) % 10) for i in ids)


def test_end_to_end_on_tiny_model(tmp_path, tiny_qwen, ablator):
    cfg = RunConfig(hf_model=tiny_qwen, tokenizer=ByteTokenizer(), ablator=ablator,
                    bands={"light": [1], "medium": [1, 2], "heavy": [0, 1, 2, 3]},
                    k=4, max_tokens_no_cot=6, max_tokens_cot={"gsm8k": 8})
    out = tmp_path / "r.csv"
    for cond in ["clean", "heavy", "random"]:
        for arm in ["cot", "no_cot"]:
            run_grid(DATA[:2], cond, arm, out, cfg)
    rs = read(out)
    assert len(rs) == 12
    assert {r["n_tokens"] for r in rs if r["arm"] == "no_cot"} == {"6"}
    assert all(r["hit_cap"] == "1" for r in rs)
