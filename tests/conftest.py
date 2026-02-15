import pytest
import torch
from nanochat.gpt import GPTConfig


@pytest.fixture
def gpt_config():
    """Fixture providing a GPTConfig instance with default values."""
    return GPTConfig()


@pytest.fixture
def small_config():
    """Fixture providing a small GPTConfig for faster tests."""
    return GPTConfig(
        sequence_len=32,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128
    )


def create_rope_cos_sin(seq_len, head_dim, base=10000.0):
    """Create RoPE cos/sin tensors for testing.
    
    Looking at apply_rotary_emb, it expects x shape (B, T, H, D) based on the gpt.py usage.
    The function splits the last dim in half, so cos/sin should broadcast to (B, T, H, D//2).
    Shape (1, seq_len, 1, D//2) should work for broadcasting.
    """
    # Create frequency components for half the head dimension
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 1).float() / half_dim))
    
    # Create position indices
    t = torch.arange(seq_len).float()
    
    # Compute angles: outer product of positions and frequencies
    angles = torch.outer(t, inv_freq)  # (seq_len, head_dim//2)
    
    # Compute cos/sin
    cos = torch.cos(angles)  # (seq_len, head_dim//2)
    sin = torch.sin(angles)  # (seq_len, head_dim//2)
    
    # Add batch and head dimensions for broadcasting: (1, seq_len, 1, head_dim//2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    return cos, sin


class MockKVCache:
    """Mock KV cache for testing."""
    
    def __init__(self, batch_size, max_seq_len, n_layers, n_kv_head, head_dim):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        
        # Initialize cache tensors
        self.k_caches = [
            torch.zeros(batch_size, max_seq_len, n_kv_head, head_dim)
            for _ in range(n_layers)
        ]
        self.v_caches = [
            torch.zeros(batch_size, max_seq_len, n_kv_head, head_dim)
            for _ in range(n_layers)
        ]
        
        # Track current sequence lengths for each batch item
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32)
    
    def get_layer_cache(self, layer_idx):
        """Get k_cache and v_cache for a specific layer."""
        return self.k_caches[layer_idx], self.v_caches[layer_idx]
    
    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens
