"""Resumable experiment runner: one CSV row per (problem, condition, arm)."""

import csv
import os
from dataclasses import dataclass, field

from tqdm.auto import tqdm

from .generation import build_prompt, clean_exclude_ids, generate
from .grading import extract_answer, grade

FIELDS = ["id", "dataset", "level", "condition", "arm", "pred", "gold",
          "correct", "n_tokens", "hit_cap", "looped", "config"]

CONDITIONS = ("clean", "random", "light", "medium", "heavy")
ARMS = ("cot", "no_cot")


@dataclass
class RunConfig:
    hf_model: object
    tokenizer: object
    ablator: object
    bands: dict                      # {'light': [...], 'medium': [...], 'heavy': [...]}
    k: int = 25                      # concepts removed per position
    n_exclude: int = 3               # clean top-N output tokens spared
    max_tokens_no_cot: int = 128
    max_tokens_cot: dict = field(default_factory=lambda: {
        "gsm8k": 1024, "math500": 4096, "aime24": 6144})

    def band_and_mode(self, condition):
        if condition == "clean":
            return [], "none"
        if condition == "random":    # control sits on the heavy band
            return self.bands["heavy"], "random"
        return self.bands[condition], "jspace"


def _done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as fh:
        return {(r["id"], r["condition"], r["arm"]) for r in csv.DictReader(fh)}


def run_grid(data, condition, arm, csv_path, cfg):
    """Run every problem in `data` under one condition x arm, appending to
    csv_path. Rows already present are skipped, so it's safe to re-run after a
    Colab disconnect."""
    if condition not in CONDITIONS or arm not in ARMS:
        raise ValueError(f"unknown condition/arm: {condition!r}, {arm!r}")
    band, mode = cfg.band_and_mode(condition)
    cot = arm == "cot"
    done = _done(csv_path)
    todo = [ex for ex in data if (ex["id"], condition, arm) not in done]

    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fh.tell() == 0:
            w.writeheader()
        for ex in tqdm(todo, desc=f"{condition} x {arm}"):
            ptxt = build_prompt(cfg.tokenizer, ex["question"], cot)
            # the random control needs exclusions too: its norm comes from the treatment
            excl = (clean_exclude_ids(cfg.hf_model, cfg.tokenizer, ptxt, cfg.n_exclude)
                    if mode != "none" else None)
            cap = cfg.max_tokens_cot[ex["dataset"]] if cot else cfg.max_tokens_no_cot
            with cfg.ablator.ablate(band, mode, exclude_ids=excl, k=cfg.k):
                text, n, hit_cap, looped = generate(cfg.hf_model, cfg.tokenizer, ptxt, cap)
            pred = extract_answer(text, cot)
            w.writerow({
                "id": ex["id"], "dataset": ex["dataset"], "level": ex.get("level", ""),
                "condition": condition, "arm": arm, "pred": pred, "gold": ex["answer"],
                "correct": int(grade(pred, ex["answer"], ex["dataset"])),
                "n_tokens": n, "hit_cap": int(hit_cap), "looped": int(looped),
                "config": "clean" if condition == "clean" else "frozen_n3"})
            fh.flush()
