"""J-space ablation and its norm-matched random control.

At each position of each layer in a band, every vocabulary token is scored
through that layer's Jacobian lens. The top-k scoring concepts are mapped back
into the layer's residual basis and their joint span is projected out. The
clean model's top-N next tokens are excluded from selection, so the edit
removes what the model is "thinking about" without directly deleting what it
is about to say.
"""

from contextlib import contextmanager

import torch


def effective_unembed(hf_model, dtype=torch.bfloat16):
    """Unembedding with the final RMSNorm gain folded in: [vocab, d_model], fp32.

    Rounded through `dtype` first, matching the precision used in the runs.
    """
    W_U = hf_model.get_output_embeddings().weight.float()
    g = hf_model.model.norm.weight.float()
    return (W_U * g).to(dtype).float()


def concept_scores(h, J, W):
    """Lens logits for residual h at a layer with Jacobian J: [..., vocab]."""
    return (h @ J.T) @ W.T


def remove_jspace(h, J, W, k, exclude_ids=None):
    """Project the top-k lens concepts at every position out of h."""
    orig_dtype = h.dtype
    h, J = h.float(), J.float()

    scores = concept_scores(h, J, W)                     # [B, S, V]
    if exclude_ids is not None and len(exclude_ids):
        scores.index_fill_(-1, exclude_ids, float("-inf"))
    top = scores.topk(k, dim=-1).indices                 # [B, S, k]

    vs = W[top] @ J                                      # [B, S, k, d]
    Q, _ = torch.linalg.qr(vs.mT)                        # [B, S, d, k]
    coeff = h.unsqueeze(-2) @ Q                          # [B, S, 1, k]
    h = h - (coeff @ Q.mT).squeeze(-2)
    return h.to(orig_dtype)


def remove_random_matched(h, Q, norm):
    """Remove a component of the given norm along fixed random directions Q."""
    orig_dtype = h.dtype
    h = h.float()
    delta = (h @ Q) @ Q.T
    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * norm
    return (h - delta).to(orig_dtype)


class Ablator:
    """Hooks J-space (or random-control) ablation into a HF decoder.

    jacobians: {layer: J} from a fitted `jlens.JacobianLens`.
    """

    MODES = ("none", "jspace", "random")

    def __init__(self, hf_model, jacobians, seed=288, dtype=torch.bfloat16):
        self.model = hf_model
        self.device = next(hf_model.parameters()).device
        self.J = {l: J.to(self.device, dtype) for l, J in jacobians.items()}
        self.W = effective_unembed(hf_model, dtype).to(self.device)
        self.d_model = self.W.shape[1]
        self.seed = seed
        self._rand_Q = {}

    def random_basis(self, layer, k):
        """Orthonormal [d, k] random directions, fixed per (layer, k)."""
        if (layer, k) not in self._rand_Q:
            g = torch.Generator().manual_seed(self.seed + 1000 * k + layer)
            Q, _ = torch.linalg.qr(torch.randn(self.d_model, k, generator=g))
            self._rand_Q[(layer, k)] = Q.to(self.device)
        return self._rand_Q[(layer, k)]

    def edit(self, h, layer, mode, k, exclude_ids):
        """One layer's edit. The random control is norm-matched on the fly to
        what the treatment would remove from this exact state; the two arms'
        generations diverge, so replaying recorded norms wouldn't make sense."""
        h_treat = remove_jspace(h, self.J[layer], self.W, k, exclude_ids)
        if mode == "jspace":
            return h_treat
        norm = (h.float() - h_treat.float()).norm(dim=-1, keepdim=True)
        return remove_random_matched(h, self.random_basis(layer, k), norm)

    @contextmanager
    def ablate(self, band, mode="jspace", exclude_ids=None, k=None):
        """Context manager: every forward pass inside it is ablated over `band`."""
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "none" or not band:
            yield
            return
        if exclude_ids is None or k is None:
            raise ValueError("jspace/random ablation needs exclude_ids and k")
        exclude_ids = torch.as_tensor(exclude_ids, dtype=torch.long, device=self.device)

        def hook_for(layer):
            def fn(module, inp, out):
                is_t = torch.is_tensor(out)
                h = self.edit(out if is_t else out[0], layer, mode, k, exclude_ids)
                return h if is_t else (h, *out[1:])
            return fn

        handles = []
        try:
            for l in band:
                handles.append(
                    self.model.model.layers[l].register_forward_hook(hook_for(l)))
            yield
        finally:
            for hnd in handles:
                hnd.remove()
