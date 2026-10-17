import pytest
import torch

from jspace.ablation import (
    concept_scores,
    effective_unembed,
    remove_jspace,
    remove_random_matched,
)

from .conftest import D_MODEL, VOCAB


@pytest.fixture
def setup():
    g = torch.Generator().manual_seed(0)
    h = torch.randn(2, 5, D_MODEL, generator=g)
    J = torch.eye(D_MODEL) + 0.1 * torch.randn(D_MODEL, D_MODEL, generator=g)
    W = torch.randn(VOCAB, D_MODEL, generator=g)
    return h, J, W


def test_selected_concepts_are_silenced(setup):
    h, J, W = setup
    k = 6
    top = concept_scores(h, J, W).topk(k, dim=-1).indices
    pre = concept_scores(h, J, W).gather(-1, top)
    post = concept_scores(remove_jspace(h, J, W, k), J, W).gather(-1, top)
    assert post.abs().max() < 1e-4 * pre.abs().max()


def test_excluded_tokens_are_never_selected(setup):
    h, J, W = setup
    k = 4
    top1 = concept_scores(h, J, W).argmax(-1).unique()
    post = concept_scores(remove_jspace(h, J, W, k, exclude_ids=top1), J, W)
    pre = concept_scores(h, J, W)
    # the protected tokens keep a substantial score; without exclusion they'd be ~0
    protected = pre[..., top1].abs().max()
    assert post[..., top1].abs().max() > 1e-2 * protected
    unprotected = concept_scores(remove_jspace(h, J, W, k), J, W)
    assert unprotected.gather(-1, pre.argmax(-1, keepdim=True)).abs().max() < 1e-3


def test_full_rank_removal_zeroes_the_state(setup):
    h, J, W = setup
    assert remove_jspace(h, J, W, k=D_MODEL).abs().max() < 1e-4


def test_edit_is_a_projection(setup):
    """Removed component lies in the concept span: h - h' is orthogonal to h'."""
    h, J, W = setup
    h2 = remove_jspace(h, J, W, k=5)
    assert ((h - h2) * h2).sum(-1).abs().max() < 1e-3
    assert (h2.norm(dim=-1) <= h.norm(dim=-1) + 1e-5).all()


def test_dtype_preserved(setup):
    h, J, W = setup
    assert remove_jspace(h.bfloat16(), J, W, k=3).dtype == torch.bfloat16


def test_random_control_matches_norm():
    g = torch.Generator().manual_seed(2)
    h = torch.randn(1, 7, D_MODEL, generator=g)
    Q, _ = torch.linalg.qr(torch.randn(D_MODEL, 5, generator=g))
    target = torch.rand(1, 7, 1, generator=g) * 3
    removed = (h - remove_random_matched(h, Q, target)).norm(dim=-1, keepdim=True)
    torch.testing.assert_close(removed, target, rtol=1e-4, atol=1e-5)


def test_effective_unembed_folds_norm_gain(tiny_qwen):
    W = effective_unembed(tiny_qwen, dtype=torch.float32)
    expected = tiny_qwen.lm_head.weight * tiny_qwen.model.norm.weight
    torch.testing.assert_close(W, expected)


# --- hooks on a real HF module stack ---------------------------------------

def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def test_ablation_changes_output_and_unhooks(tiny_qwen, ablator, ids):
    clean = logits(tiny_qwen, ids)
    with ablator.ablate([1, 2], "jspace", exclude_ids=[], k=8):
        ablated = logits(tiny_qwen, ids)
    assert not torch.allclose(clean, ablated)
    torch.testing.assert_close(logits(tiny_qwen, ids), clean)


def test_hooks_removed_on_exception(tiny_qwen, ablator, ids):
    clean = logits(tiny_qwen, ids)
    with pytest.raises(RuntimeError):
        with ablator.ablate([0, 1, 2, 3], "random", exclude_ids=[], k=4):
            raise RuntimeError
    torch.testing.assert_close(logits(tiny_qwen, ids), clean)


def test_none_mode_is_identity(tiny_qwen, ablator, ids):
    clean = logits(tiny_qwen, ids)
    with ablator.ablate([0, 1], "none"):
        torch.testing.assert_close(logits(tiny_qwen, ids), clean)


def test_random_control_moves_same_distance_as_treatment(ablator):
    h = torch.randn(1, 6, D_MODEL)
    treat = ablator.edit(h, 2, "jspace", 5, torch.tensor([], dtype=torch.long))
    ctrl = ablator.edit(h, 2, "random", 5, torch.tensor([], dtype=torch.long))
    torch.testing.assert_close((h - ctrl).norm(dim=-1), (h - treat).norm(dim=-1),
                               rtol=1e-4, atol=1e-5)
    assert not torch.allclose(ctrl, treat)


def test_random_basis_is_orthonormal_and_reproducible(tiny_qwen, jacobians, ablator):
    from jspace.ablation import Ablator
    Q = ablator.random_basis(layer=3, k=6)
    torch.testing.assert_close(Q.T @ Q, torch.eye(6), atol=1e-5, rtol=0)
    fresh = Ablator(tiny_qwen, jacobians, seed=288, dtype=torch.float32)
    torch.testing.assert_close(fresh.random_basis(3, 6), Q)
    assert not torch.allclose(fresh.random_basis(2, 6), Q)


def test_ablation_applies_during_generate(tiny_qwen, ablator, ids):
    """KV-cached decoding goes through the same hooks."""
    kw = dict(max_new_tokens=12, do_sample=False, pad_token_id=0)
    clean = tiny_qwen.generate(ids, **kw)
    with ablator.ablate([0, 1, 2, 3], "jspace", exclude_ids=[], k=24):
        ablated = tiny_qwen.generate(ids, **kw)
    assert not torch.equal(clean, ablated)


@pytest.mark.parametrize("kwargs", [dict(mode="bogus", exclude_ids=[], k=1),
                                    dict(mode="jspace", exclude_ids=None, k=1),
                                    dict(mode="random", exclude_ids=[], k=None)])
def test_bad_arguments_rejected(ablator, kwargs):
    with pytest.raises(ValueError):
        with ablator.ablate([0], **kwargs):
            pass
