import torch
import pytest
from nanochat.gpt import norm, MLP, GPTConfig, CausalSelfAttention, has_ve
from tests.conftest import create_rope_cos_sin, MockKVCache


def test_norm():
    x = torch.randn(10, 10)
    assert norm(x).shape == (10, 10)


def test_mlp(gpt_config):
    mlp = MLP(gpt_config)
    x = torch.randn(10, gpt_config.n_embd)
    assert mlp.forward(x).shape == (10, gpt_config.n_embd)


def test_causal_self_attention_prefill_no_ve(small_config):
    """Test CausalSelfAttention in prefill mode without value embedding."""
    # Use layer 0 with n_layer=2, which should not have VE
    layer_idx = 0
    assert not has_ve(layer_idx, small_config.n_layer), "Layer 0 should not have VE for this test"
    
    attn = CausalSelfAttention(small_config, layer_idx)
    
    # Test dimensions
    B, T = 2, 16
    head_dim = small_config.n_embd // small_config.n_head
    
    # Create inputs
    x = torch.randn(B, T, small_config.n_embd)
    ve = None  # No value embedding for this layer
    cos, sin = create_rope_cos_sin(T, head_dim)
    cos_sin = (cos, sin)
    window_size = (-1, -1)  # No window
    kv_cache = None  # Prefill mode
    
    # Forward pass
    y = attn.forward(x, ve, cos_sin, window_size, kv_cache)
    
    # Assertions
    assert y.shape == (B, T, small_config.n_embd)
    assert torch.isfinite(y).all(), "Output should be finite"
    assert not torch.equal(y, x), "Output should be different from input"


def test_causal_self_attention_prefill_with_ve(small_config):
    """Test CausalSelfAttention in prefill mode with value embedding."""
    # Use layer 1 with n_layer=2, which should have VE
    layer_idx = 1
    assert has_ve(layer_idx, small_config.n_layer), "Layer 1 should have VE for this test"
    
    attn = CausalSelfAttention(small_config, layer_idx)
    
    # Test dimensions
    B, T = 2, 16
    head_dim = small_config.n_embd // small_config.n_head
    
    # Create inputs
    x = torch.randn(B, T, small_config.n_embd)
    ve = torch.randn(B, 1, small_config.n_kv_head, head_dim)  # Value embedding
    cos, sin = create_rope_cos_sin(T, head_dim)
    cos_sin = (cos, sin)
    window_size = (-1, -1)  # No window
    kv_cache = None  # Prefill mode
    
    # Forward pass
    y = attn.forward(x, ve, cos_sin, window_size, kv_cache)
    
    # Assertions
    assert y.shape == (B, T, small_config.n_embd)
    assert torch.isfinite(y).all(), "Output should be finite"
    assert not torch.equal(y, x), "Output should be different from input"


def test_causal_self_attention_decode_with_cache(small_config):
    """Test CausalSelfAttention in decode mode with KV cache."""
    layer_idx = 0
    attn = CausalSelfAttention(small_config, layer_idx)
    
    # Test dimensions
    B = 2
    T_cache = 20  # Cache size
    T_new = 1     # Single token generation
    head_dim = small_config.n_embd // small_config.n_head
    
    # Create KV cache
    kv_cache = MockKVCache(B, T_cache, small_config.n_layer, small_config.n_kv_head, head_dim)
    
    # Simulate some existing cache content
    cache_pos = 10
    kv_cache.cache_seqlens.fill_(cache_pos)
    
    # Create inputs for single token
    x = torch.randn(B, T_new, small_config.n_embd)
    ve = None  # No VE for layer 0
    # For decode, cos/sin should be for the current token position only
    cos_full, sin_full = create_rope_cos_sin(cache_pos + T_new, head_dim)
    cos = cos_full[:, cache_pos:cache_pos + T_new, :, :]  # Slice to current position
    sin = sin_full[:, cache_pos:cache_pos + T_new, :, :]
    cos_sin = (cos, sin)
    window_size = (-1, -1)  # No window
    
    # Forward pass
    y = attn.forward(x, ve, cos_sin, window_size, kv_cache)
    
    # Assertions
    assert y.shape == (B, T_new, small_config.n_embd)
    assert torch.isfinite(y).all(), "Output should be finite"
    
    # Check that cache was advanced (only for last layer)
    if layer_idx == small_config.n_layer - 1:
        assert kv_cache.cache_seqlens[0].item() == cache_pos + T_new
    else:
        assert kv_cache.cache_seqlens[0].item() == cache_pos


def test_causal_self_attention_with_window(small_config):
    """Test CausalSelfAttention with sliding window."""
    layer_idx = 0
    attn = CausalSelfAttention(small_config, layer_idx)
    
    # Test dimensions
    B, T = 2, 16
    head_dim = small_config.n_embd // small_config.n_head
    window_size = (8, -1)  # Left window of 8
    
    # Create inputs
    x = torch.randn(B, T, small_config.n_embd)
    ve = None
    cos, sin = create_rope_cos_sin(T, head_dim)
    cos_sin = (cos, sin)
    kv_cache = None  # Prefill mode
    
    # Forward pass
    y = attn.forward(x, ve, cos_sin, window_size, kv_cache)
    
    # Assertions
    assert y.shape == (B, T, small_config.n_embd)
    assert torch.isfinite(y).all(), "Output should be finite"


def test_causal_self_attention_shapes_consistency(small_config):
    """Test that CausalSelfAttention maintains shape consistency across different configs."""
    for layer_idx in range(small_config.n_layer):
        attn = CausalSelfAttention(small_config, layer_idx)
        
        B, T = 1, 8
        head_dim = small_config.n_embd // small_config.n_head
        
        x = torch.randn(B, T, small_config.n_embd)
        ve = torch.randn(B, 1, small_config.n_kv_head, head_dim) if has_ve(layer_idx, small_config.n_layer) else None
        cos, sin = create_rope_cos_sin(T, head_dim)
        cos_sin = (cos, sin)
        window_size = (-1, -1)
        kv_cache = None
        
        y = attn.forward(x, ve, cos_sin, window_size, kv_cache)
        assert y.shape == (B, T, small_config.n_embd)


if __name__ == "__main__":
    config = GPTConfig()
    small_config = GPTConfig(sequence_len=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=128)
    
    # Run a quick test
    test_causal_self_attention_prefill_no_ve(small_config)
