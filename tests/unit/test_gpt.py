import torch
from nanochat.gpt import norm
from nanochat.gpt import MLP
from nanochat.gpt import GPTConfig
import pytest
from tests.conftest import gpt_config
from nanochat.gpt import CausalSelfAttention


def test_norm():
    x = torch.randn(10, 10)
    assert norm(x).shape == (10, 10)

def test_mlp(gpt_config):
    mlp = MLP(gpt_config)
    x = torch.randn(10, gpt_config.n_embd)
    assert mlp.forward(x).shape == (10, gpt_config.n_embd)

def test_causal_self_attention(gpt_config):
    causal_self_attention = CausalSelfAttention(gpt_config, 0)
    x = torch.randn(10, gpt_config.n_embd)
    assert causal_self_attention.forward(x).shape == (10, gpt_config.n_embd)



if __name__ == "__main__":
    config = GPTConfig()
    # test_mlp(config)
    test_norm()
