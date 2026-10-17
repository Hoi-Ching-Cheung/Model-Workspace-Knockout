"""The frozen 330-problem sample: 150 GSM8K, 150 MATH-500 (30/level), 30 AIME 2024."""

import json
import os
import random

from datasets import load_dataset

AIME_DATASET = "HuggingFaceH4/aime_2024"   # both 2024 exams, 30 problems


def _draw_sample(gsm8k, math500, aime, seed, n_gsm8k, n_per_level):
    rng = random.Random(seed)
    gsm_idx = sorted(rng.sample(range(len(gsm8k)), n_gsm8k))
    math_ids = []
    for lvl in sorted(set(math500["level"])):
        pool = [ex["unique_id"] for ex in math500 if ex["level"] == lvl]
        math_ids += sorted(rng.sample(pool, min(n_per_level, len(pool))))
    return {"gsm8k_test_indices": gsm_idx,
            "math500_unique_ids": math_ids,
            "aime24_ids": [ex["id"] for ex in aime]}


def load_sample(path, seed=288, n_gsm8k=150, n_per_level=30):
    """Uniform records {dataset, id, question, answer, [level]}.

    Problem ids are drawn once and frozen to `path`; later calls reload them,
    so every condition is scored on the same problems.
    """
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    aime = load_dataset(AIME_DATASET, split="train")

    if os.path.exists(path):
        with open(path) as f:
            sample = json.load(f)
    else:
        sample = _draw_sample(gsm8k, math500, aime, seed, n_gsm8k, n_per_level)
        with open(path, "w") as f:
            json.dump(sample, f, indent=1)

    data_gsm8k = [{
        "dataset": "gsm8k", "id": f"gsm8k/{i}",
        "question": gsm8k[i]["question"],
        # gold is the integer after '####'
        "answer": gsm8k[i]["answer"].split("####")[-1].strip().replace(",", ""),
    } for i in sample["gsm8k_test_indices"]]

    math_by_id = {ex["unique_id"]: ex for ex in math500}
    data_math = [{
        "dataset": "math500", "id": uid, "level": math_by_id[uid]["level"],
        "question": math_by_id[uid]["problem"], "answer": math_by_id[uid]["answer"],
    } for uid in sample["math500_unique_ids"]]

    aime_by_id = {ex["id"]: ex for ex in aime}
    data_aime = [{
        "dataset": "aime24", "id": f"aime24/{i}",
        "question": aime_by_id[i]["problem"],
        "answer": str(aime_by_id[i]["answer"]).strip(),
    } for i in sample["aime24_ids"]]

    return data_gsm8k, data_math, data_aime
