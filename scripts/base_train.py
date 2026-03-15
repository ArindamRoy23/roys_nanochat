"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import json
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    autodetect_device_type,
    get_peak_flops,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
from nanochat.flash_attention import HAS_FA3
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint

# from nanochat.loss_eval import evaluate_bpb
# from nanochat.engine import E58ngine
# from scripts.base_eval import evaluate_core
# from nanochat.dataloader import (
#     tokenizing_distributed_data_loader_bos_bestfit,
#     tokenizing_distributed_data_loader_with_state_bos_bestfit,
# )
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument(
    "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
# FP8 training
parser.add_argument(
    "--fp8",
    action="store_true",
    help="enable FP8 training (requires H100+ GPU and torchao)",
)
parser.add_argument(
    "--fp8-recipe",
    type=str,
    default="tensorwise",
    choices=["rowwise", "tensorwise"],
    help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)",
)
# Model architecture
parser.add_argument(
    "--depth", type=int, default=20, help="depth of the Transformer model"
)
parser.add_argument(
    "--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio"
)
parser.add_argument(
    "--head-dim", type=int, default=128, help="target head dimension for attention"
)
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument(
    "--window-pattern",
    type=str,
    default="SSSL",
    help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')",
)
# Training horizon (only one used, in order of precedence)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=-1,
    help="explicit number of optimization steps (-1 = disable)",
)
parser.add_argument(
    "--target-flops",
    type=float,
    default=-1.0,
    help="calculate num_iterations to reach target_flops (-1 = disable)",
)
parser.add_argument(
    "--target-param-data-ratio",
    type=float,
    default=10.5,
    help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)",
)
# Optimization
parser.add_argument(
    "--device-batch-size",
    type=int,
    default=32,
    help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.",
)
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=-1,
    help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)",
)
parser.add_argument(
    "--embedding-lr",
    type=float,
    default=0.3,
    help="learning rate for embedding parameters (Adam)",
)
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=0.004,
    help="learning rate for unembedding parameters (Adam)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.2,
    help="cautious weight decay for the Muon optimizer (for weights)",
)
parser.add_argument(
    "--matrix-lr",
    type=float,
    default=0.02,
    help="learning rate for matrix parameters (Muon)",
)
parser.add_argument(
    "--scalar-lr",
    type=float,
    default=0.5,
    help="learning rate for scalars (resid_lambdas, x0_lambdas)",
)
parser.add_argument(
    "--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding"
)
parser.add_argument(
    "--adam-beta2",
    type=float,
    default=0.95,
    help="Adam beta2 for embedding/unembedding",
)
parser.add_argument(
    "--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup"
)
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=0.5,
    help="ratio of iterations for LR warmdown",
)
parser.add_argument(
    "--final-lr-frac",
    type=float,
    default=0.0,
    help="final LR as fraction of initial LR",
)
parser.add_argument(
    "--resume-from-step",
    type=int,
    default=-1,
    help="resume training from this step (-1 = disable)",
)
# Evaluation
parser.add_argument(
    "--eval-every",
    type=int,
    default=250,
    help="evaluate val bpb every N steps (-1 = disable)",
)
parser.add_argument(
    "--eval-tokens",
    type=int,
    default=40 * 524288,
    help="number of tokens to evaluate val loss on",
)
parser.add_argument(
    "--core-metric-every",
    type=int,
    default=2000,
    help="evaluate CORE metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--core-metric-max-per-task",
    type=int,
    default=500,
    help="examples per task for CORE metric",
)
parser.add_argument(
    "--sample-every",
    type=int,
    default=2000,
    help="sample from model every N steps (-1 = disable)",
)
parser.add_argument(
    "--save-every",
    type=int,
    default=-1,
    help="save checkpoints every N steps (-1 = only at end)",
)
# Output
parser.add_argument(
    "--model-tag",
    type=str,
    default=None,
    help="override model tag for checkpoint directory name",
)
args = parser.parse_args()
user_config = vars(args).copy()  # for logging


# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat", name=args.run, config=user_config)
)

# Flash Attention status
if HAS_FA3:
    print0(
        "✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome."
    )
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(
            f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible."
        )
        print0(
            "WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns."
        )
    print0("!" * 80)


# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")


# -----------------------------------------------------------------------------
# Initialize the Model


def build_model_meta(depth):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


# Build the model, move to device, init the weights
model = build_model_meta(
    args.depth
)  # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(
    device=device
)  # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights()  # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # free up this memory after the copy


# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility

        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            # FP8 requires both in_features and out_features divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(
            model, config=fp8_config, module_filter_fn=fp8_module_filter
        )
        num_fp8_layers = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
        num_skipped = (
            sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        )
        print0(
            f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)"
        )


# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if "Float8" in type(module).__name__:
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)


# -----------------------------------------------------------------------------
# Compile the model

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(
    model, dynamic=False
)  # the inputs to model will never change shape so dynamic=False is safe
