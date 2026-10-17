"""Checks that the ablation does what it claims before trusting a null result.

Two measurements:
- multihop accuracy: two-hop questions whose bridge entity is never written
  ("animal that spins webs" -> spider -> 8). J-space ablation should hurt;
  a norm-matched random ablation shouldn't.
- prompt-token top-1 match: how often the ablated model's argmax agrees with
  the clean model's over wikitext. A dial for overall ablation strength.
"""

import torch

from .generation import clean_exclude_ids, generate

MULTIHOP = [
    ("The number of legs on the animal that spins webs is", ["8", "eight"]),
    ("The capital of the country shaped like a boot is", ["rome"]),
    ("The language spoken in the country whose capital is Madrid is", ["spanish"]),
    ("The continent containing the country whose capital is Tokyo is", ["asia"]),
    ("The number of legs on the insect that makes honey is", ["6", "six"]),
    ("The capital of the country where the Eiffel Tower stands is", ["paris"]),
    ("The currency used in the country whose capital is Rome is the", ["euro", "euros"]),
    ("The colour of the gas giant with the great red spot is mostly", ["red"]),
    ("The capital of the country famous for tulips and windmills is", ["amsterdam"]),
    ("The language spoken in the country whose capital is Lisbon is", ["portuguese"]),
    ("The continent containing the country whose capital is Cairo is", ["africa"]),
    ("The currency used in the country whose capital is Tokyo is the", ["yen"]),
]

NO_EXCLUSIONS = torch.tensor([], dtype=torch.long)


def _chat(tokenizer, q):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def multihop_acc(hf_model, tokenizer, ablator, band, mode, k=0, n=0, items=MULTIHOP):
    hits = 0
    for q, aliases in items:
        ptxt = _chat(tokenizer, "Answer with one word. " + q)
        excl = (clean_exclude_ids(hf_model, tokenizer, ptxt, n)
                if mode != "none" and n > 0 else NO_EXCLUSIONS)
        with ablator.ablate(band, mode, exclude_ids=excl, k=k):
            text, *_ = generate(hf_model, tokenizer, ptxt, 12)
        hits += any(a in text.lower() for a in aliases)
    return hits / len(items)


@torch.no_grad()
def pt_match(hf_model, tokenizer, ablator, texts, band, mode, k, max_length=96):
    device = next(hf_model.parameters()).device
    agree = total = 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=max_length).input_ids.to(device)
        clean = hf_model(ids).logits.argmax(-1)
        with ablator.ablate(band, mode, exclude_ids=NO_EXCLUSIONS, k=k):
            ablated = hf_model(ids).logits.argmax(-1)
        agree += (clean == ablated).sum().item()
        total += clean.numel()
    return agree / total
