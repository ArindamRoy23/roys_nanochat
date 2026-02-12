import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
import torch


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    window_pattern: str = "SSSL"


# -----------------------------------------------------------------------------
def norm(x):
    return F.rms_norm(x, (x.size(-1), ))

def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):


    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

# -----------------------------------------------------------------------------


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx 
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # 768 * (6 * 128) [Shape: (768, 768)]
        self.c_q = nn.Linear(self.n_embd,
                             self.n_head * self.head_dim,
                             bias=False)
        self.c_k = nn.Linear(self.n_embd,
                             self.n_kv_head * self.head_dim,
                             bias=False)
        self.c_v = nn.Linear(self.n_embd,
                             self.n_kv_head * self.head_dim,
                             bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(
            self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(
                layer_idx, config.n_layer) else None
    
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """
        Default config dimension:
        BSZ: 1
        T: 2048
        C: 768
        x: Input tensor of shape (BSZ, 2048, 768) 
        ve: Value Embedding tensor of shape (BSZ, 1, 6, 128)
        cos_sin: Rotary Embeddings tensor of shape (1, 2048, 768)
        window_size: Window size of 2048
        kv_cache: Key-Value cache tensor of shape (BSZ, 2048, 6, 128)
        """
        B, T, C = x.size() # (BSZ, 2048, 768)

        # (BSZ, 2048, 768)*(768, 768)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim) # (BSZ, 2048, 6, 128)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim) # (BSZ, 2048, 6, 128)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim) # (BSZ, 2048, 6, 128)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim) # (BSZ, 1, 6, 128)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        
        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm