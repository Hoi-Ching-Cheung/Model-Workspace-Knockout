import pytest
import torch
import transformers

from jspace.ablation import Ablator

N_LAYERS, D_MODEL, VOCAB = 4, 32, 64


@pytest.fixture(scope="session")
def tiny_qwen():
    """Randomly initialised 4-layer Qwen3: same module layout as Qwen3-4B."""
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8)
    return transformers.Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def jacobians():
    """Stand-in lens: near-identity transport at every layer."""
    g = torch.Generator().manual_seed(1)
    return {l: torch.eye(D_MODEL) + 0.1 * torch.randn(D_MODEL, D_MODEL, generator=g)
            for l in range(N_LAYERS)}


@pytest.fixture
def ablator(tiny_qwen, jacobians):
    return Ablator(tiny_qwen, jacobians, seed=288, dtype=torch.float32)


@pytest.fixture
def ids():
    return torch.tensor([[1, 5, 9, 13, 2, 40, 7, 21]])
