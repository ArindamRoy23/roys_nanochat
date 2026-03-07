import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
import torch
from nanochat.flash_attention import flash_attn
from nanochat.common import get_dist_info


@dataclass
class GPTConfig:
    '''
    GPTConfig is a dataclass that contains the configuration for the GPT model.
    '''
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads #TODO: Rename to n_q_head
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    window_pattern: str = "SSSL"


# -----------------------------------------------------------------------------
def norm(x):
    """
    norm is a function that normalizes the input tensor.
    """
    return F.rms_norm(x, (x.size(-1), ))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):

    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
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
            ---
            x: Input tensor of shape (BSZ, 2048, 768) 
            ve: Value Embedding tensor of shape (BSZ, 1, 6, 128)
            cos_sin: Rotary Embeddings tensor of shape (1, 2048, 768)
            window_size: Window size of 2048
            kv_cache: Key-Value cache tensor of shape (BSZ, 2048, 6, 128)
            c_proj: Projection weights tensor of shape (768, 768)
        """
        B, T, C = x.size()  # (BSZ, 2048, 768)

        # (BSZ, 2048, 768)*(768, 768)
        q = self.c_q(x).view(B, T, self.n_head,
                             self.head_dim)  # (BSZ, 2048, 6, 128)
        k = self.c_k(x).view(B, T, self.n_kv_head,
                             self.head_dim)  # (BSZ, 2048, 6, 128)
        v = self.c_v(x).view(B, T, self.n_kv_head,
                             self.head_dim)  # (BSZ, 2048, 6, 128)

        if ve is not None:
            # ve should be (B, 1, n_kv_head, head_dim) and broadcast to (B, T, n_kv_head, head_dim)
            gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q,
                                           k,
                                           v,
                                           causal=True,
                                           window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# Transformer Block
class Block(nn.Module):
    """
    Block is a transformer block that contains a causal self-attention layer and a MLP layer.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """
        Forward pass through the block.

        x: Input tensor of shape (BSZ, T, C)
        ve: Value Embedding tensor of shape (BSZ, 1, n_kv_head, head_dim)
        cos_sin: Rotary Embeddings tensor of shape (1, T, C)
        window_size: Window size of 2048
        kv_cache: Key-Value cache tensor of shape (BSZ, T, n_kv_head, head_dim)

        Returns:
            Output tensor of shape (BSZ, T, C)
        """
        x = self.attn(x, ve, cos_sin, window_size, kv_cache)
        x = self.mlp(x)
        return x


class GPT(nn.Module):

    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) //
                             pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        self.transformer = nn.ModuleDict({
            "wte":
            nn.Embedding(padded_vocab_size, config.n_embd),
            "h":
            nn.ModuleList([
                Block(config, layer_idx) for layer_idx in range(config.n_layer)
            ]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(
            config.n_layer))  # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(
            config.n_layer))  # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i):
            nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10  # 10X over-compute should be enough, TODO make nicer?
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len,
                                                      head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Steps: 
            S1 - 
                Initialize the weights of the embedding and unembedding layers.
            S2 - 
                Initialize the weights of the transformer blocks.
            S3 - 
                Initialize the weights of the residual lambdas and x0 lambdas.
            S4 - 
                Initialize the weights of the value embeddings. #TODO: Why is this separate from the transformer blocks?
            S5 - 
                Initialize the rotary embeddings.
            S6 - 
                Convert the weights to bfloat16 if on GPU.
        """
        # S1
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # S2
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s,
                                   s)  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(
                block.attn.c_proj.weight)  # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # S3
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.0)

        # S4
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # S5
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len,
                                                      head_dim)
        self.cos, self.sin = cos, sin

        # S6
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self,
                                      seq_len,
                                      head_dim,
                                      base=10000,
                                      device=None):
        """
        Precompute the rotary embeddings for the given sequence length and head dimension.
        """
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0,
                                     head_dim,
                                     2,
                                     dtype=torch.float32,
                                     device=device)
        inv_freq = 1.0 / (base**(channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[
            None, :, None, :]  # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(
            c in "SL" for c in
            pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel()
                                 for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() +
                           value_embeds_numel + self.resid_lambdas.numel() +
                           self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel()
                                   for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(
            p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self,
                        unembedding_lr=0.004,
                        embedding_lr=0.2,
                        matrix_lr=0.02,
                        weight_decay=0.0,
                        adam_betas=(0.8, 0.95),
                        scalar_lr=0.5):
        """
        Setup the optimizer for the model.
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups/
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params) + len(lm_head_params) + len(
                value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768)**-0.5
        print(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw',
                 params=lm_head_params,
                 lr=unembedding_lr * dmodel_lr_scale,
                 betas=adam_betas,
                 eps=1e-10,
                 weight_decay=0.0),
            dict(kind='adamw',
                 params=embedding_params,
                 lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas,
                 eps=1e-10,
                 weight_decay=0.0),
            dict(kind='adamw',
                 params=value_embeds_params,
                 lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas,
                 eps=1e-10,
                 weight_decay=0.0),
            dict(kind='adamw',
                 params=resid_params,
                 lr=scalar_lr * 0.01,
                 betas=adam_betas,
                 eps=1e-10,
                 weight_decay=0.0),
            dict(kind='adamw',
                 params=x0_params,
                 lr=scalar_lr,
                 betas=(0.96, 0.95),
                 eps=1e-10,
                 weight_decay=0.0),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind='muon',
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                ))

        # Factory = DistMuonAdamW if ddp else MuonAdamW
        # optimizer = Factory(param_groups)

        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
    
    def forward(self, idx, targets , kv_cache=None, loss_reduce='mean'):
        B, T = idx.size() # Batch_size, Token_length
        
        # T0 is the starting position of the current forward pass
        # If kv_cache is None, we are at the beginning, so T0 = 0
        # If kv_cache is not None, we are continuing from a previous forward pass, so T0 = kv_cache.get_pos()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
        
        # Get the token embeddings
        x = self.transformer.wte(idx) # (B, T, C) : (16, 1024, 768)
        # Normalize the token embeddings (B, T, C) : (16, 1024, 768)
        x = norm(x) 
        x0 = x 

        # For each layer in the transformer
        # We add the residual scaling factor and the x0 scaling factor to the input
        # x = resid_lambda * x + x0_lambda * x0
        # Then we pass the input through the transformer block
        # x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x) # (B, T, C) : (16, 1024, 768)

        # Cap the logits to prevent overflow
        softcap = 15

        # Get the logits from the transformer
        logits = self.lm_head(x) # (B, T, vocab_size) : (16, 1024, 50257)
        
        
        # Get the logits from the transformer
        logits = logits[..., :self.config.vocab_size]

        
        

            
        