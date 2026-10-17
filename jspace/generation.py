"""Prompting and greedy generation for the two answering modes."""

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

COT_SUFFIX = ("\n\nReason step by step, then give the final answer "
              "on a new line in the form 'Final answer: <answer>'.")
NO_COT_SUFFIX = "\n\nGive only the final answer, no working."


def build_prompt(tokenizer, question, cot):
    """Chat-templated prompt (thinking disabled). No-CoT prefills 'Final answer:'
    so the answer tokens come out first, with no room to reason in text."""
    if cot:
        msgs = [{"role": "user", "content": question + COT_SUFFIX}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    msgs = [{"role": "user", "content": question + NO_COT_SUFFIX},
            {"role": "assistant", "content": "Final answer:"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, continue_final_message=True,
        enable_thinking=False)


def _four_grams(ids):
    return {tuple(ids[i:i + 4]) for i in range(len(ids) - 3)}


class LoopStop(StoppingCriteria):
    """Stop when the last `window` tokens mostly repeat the window before them.

    Ablated CoT runs sometimes fall into a loop and would otherwise burn the
    whole token budget. `fired` records whether this happened.
    """

    def __init__(self, window=200, thresh=0.9, min_len=600):
        self.window, self.thresh, self.min_len = window, thresh, min_len
        self.fired = False

    def __call__(self, input_ids, scores, **kw):
        ids = input_ids[0]
        if ids.shape[0] < self.min_len:
            return False
        tail = _four_grams(ids[-self.window:].tolist())
        prev = _four_grams(ids[-2 * self.window:-self.window].tolist())
        self.fired = len(tail & prev) / max(len(tail), 1) >= self.thresh
        return self.fired


@torch.no_grad()
def generate(hf_model, tokenizer, prompt_text, max_new_tokens):
    """Greedy decode. Returns (text, n_new_tokens, hit_cap, looped)."""
    device = next(hf_model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    stopper = LoopStop()
    out = hf_model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
        stopping_criteria=StoppingCriteriaList([stopper]),
        pad_token_id=tokenizer.eos_token_id)
    gen = out[0, inputs.input_ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text, len(gen), len(gen) >= max_new_tokens, stopper.fired


@torch.no_grad()
def clean_exclude_ids(hf_model, tokenizer, prompt_text, n):
    """Top-n next-token ids from a clean forward pass at the last prompt
    position. Call outside any ablation context."""
    device = next(hf_model.parameters()).device
    ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    return hf_model(ids).logits[0, -1].topk(n).indices
