"""
Mamba for Regression using the `fla` (flash-linear-attention) package.

This script builds a Mamba-based regression model on top of fla's Triton-
accelerated Mamba implementation.  It is designed for *continuous-valued*
sequential inputs (not token IDs), so we bypass fla's embedding layer and
feed projected features directly via `inputs_embeds`.

Requirements
------------
    pip install torch einops transformers
    pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps

Triton backend
--------------
fla ships its own Triton kernels for conv1d and selective-scan operations.
To guarantee you are running on the **Triton** backend rather than the
CUDA `causal-conv1d` / `mamba-ssm` kernels:

    pip uninstall causal-conv1d mamba-ssm -y   # remove CUDA fallbacks

fla will then automatically use its pure-Triton code-paths.  You can
verify this at runtime – the script prints a confirmation banner.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────
# 0. Verify that the Triton backend will be used
# ──────────────────────────────────────────────────────────────────────

def _check_triton_backend() -> None:
    """Warn loudly if the CUDA causal-conv1d package is installed,
    because fla will prefer it over its own Triton conv1d kernels."""
    try:
        import causal_conv1d  # noqa: F401
        print(
            "\n⚠  `causal-conv1d` is installed – fla may use the CUDA "
            "backend instead of Triton.\n"
            "   Run `pip uninstall causal-conv1d -y` to force the Triton path.\n"
        )
    except ImportError:
        print("✓ causal-conv1d not found – fla will use its Triton conv1d backend.")

    try:
        import mamba_ssm  # noqa: F401
        print(
            "⚠  `mamba-ssm` is installed – fla may use the CUDA selective-scan "
            "kernel.\n"
            "   Run `pip uninstall mamba-ssm -y` to force the Triton path.\n"
        )
    except ImportError:
        print("✓ mamba-ssm not found – fla will use its Triton selective-scan backend.")

    import triton  # noqa: F401
    print(f"✓ Triton version: {triton.__version__}\n")


# ──────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MambaRegressionConfig:
    """All knobs for the model **and** training in one place."""

    # --- Mamba backbone ------------------------------------------------
    hidden_size: int = 128          # internal Mamba dimension
    state_size: int = 16            # SSM state dimension (N in the paper)
    num_hidden_layers: int = 4      # number of stacked Mamba blocks
    expand: int = 2                 # expansion factor → intermediate_size = expand * hidden_size
    conv_kernel: int = 4            # causal conv1d kernel width
    norm_eps: float = 1e-5

    # --- Input / output ------------------------------------------------
    input_dim: int = 10             # number of input features per timestep
    num_targets: int = 3            # number of regression outputs
    mlp_hidden: int = 128           # hidden width of the regression head

    # --- Training ------------------------------------------------------
    seq_len: int = 128              # sequence length
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    num_epochs: int = 20
    warmup_steps: int = 100
    dtype: str = "float32"          # "float32" | "bfloat16"
    device: str = "cuda"

    # --- Data (synthetic) ---------------------------------------------
    num_train: int = 4096
    num_val: int = 512


# ──────────────────────────────────────────────────────────────────────
# 2. Model
# ──────────────────────────────────────────────────────────────────────

class MambaForRegression(nn.Module):
    """
    Mamba backbone (from fla) → mean-pool over time → MLP regression head.

    Architecture
    ------------
    input (B, T, input_dim)
        → Linear projection to hidden_size
        → N × MambaBlock  (via fla's MambaModel, using inputs_embeds)
        → RMSNorm
        → mean-pool across time → (B, hidden_size)
        → MLP → (B, num_targets)
    """

    def __init__(self, cfg: MambaRegressionConfig):
        super().__init__()
        self.cfg = cfg

        # --- fla imports (deferred so the backend check runs first) -----
        from fla.models.mamba.configuration_mamba import MambaConfig as FLAMambaConfig
        from fla.models.mamba.modeling_mamba import MambaModel as FLAMambaModel

        # Build the fla config.  We set vocab_size=1 because we never use
        # the token-embedding layer – we feed inputs_embeds directly.
        fla_cfg = FLAMambaConfig(
            hidden_size=cfg.hidden_size,
            state_size=cfg.state_size,
            num_hidden_layers=cfg.num_hidden_layers,
            expand=cfg.expand,
            conv_kernel=cfg.conv_kernel,
            norm_eps=cfg.norm_eps,
            vocab_size=2,                  # placeholder; unused
            use_cache=False,               # not needed for training
            fuse_norm=True,                # use fla's fused Triton RMSNorm
            fuse_cross_entropy=False,      # irrelevant for regression
        )

        # The fla MambaModel backbone: embeddings + blocks + final norm.
        # We'll bypass the embedding via inputs_embeds.
        self.backbone = FLAMambaModel(fla_cfg)

        # Project raw features → hidden_size so we can pass as inputs_embeds
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_size)

        # Regression head – a small MLP
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.num_targets),
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, input_dim)
        mask: Optional[torch.Tensor] = None,    # (B, T) bool – True = keep
    ) -> torch.Tensor:                          # (B, num_targets)
        """
        Parameters
        ----------
        x    : continuous features per timestep.
        mask : optional boolean mask for variable-length sequences.
               True means *valid* timestep, False means padding.
        """
        # Project to hidden dimension
        embeds = self.input_proj(x)             # (B, T, hidden_size)

        # Run through fla's Mamba backbone.  Passing `inputs_embeds`
        # skips the internal nn.Embedding and feeds our projection instead.
        out = self.backbone(
            inputs_embeds=embeds,
            use_cache=False,
        )
        hidden = out.last_hidden_state          # (B, T, hidden_size)

        # Mean-pool over the time axis (respecting the mask if given)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)          # (B, hidden_size)

        return self.head(pooled)                 # (B, num_targets)


# ──────────────────────────────────────────────────────────────────────
# 3. Synthetic data generation
# ──────────────────────────────────────────────────────────────────────

def make_synthetic_data(
    n_samples: int,
    seq_len: int,
    input_dim: int,
    num_targets: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Create a synthetic regression dataset.

    The targets depend on both the *mean* and *variance* of the input
    sequence along the time axis plus a non-linear cross-feature term,
    so the model must actually learn to aggregate temporal information.
    """
    torch.manual_seed(42)
    X = torch.randn(n_samples, seq_len, input_dim, device=device, dtype=dtype)

    # Ground-truth: a non-trivial function of temporal statistics
    means = X.mean(dim=1)                         # (N, input_dim)
    stds  = X.std(dim=1)                          # (N, input_dim)
    # Combine means and stds, pick the first `num_targets` outputs
    W_mean = torch.randn(input_dim, num_targets, device=device, dtype=dtype) * 0.5
    W_std  = torch.randn(input_dim, num_targets, device=device, dtype=dtype) * 0.3
    Y = means @ W_mean + stds @ W_std
    # Add a mild non-linearity + noise
    Y = Y + 0.1 * torch.sin(Y * 2)
    Y = Y + 0.05 * torch.randn_like(Y)

    return X, Y


# ──────────────────────────────────────────────────────────────────────
# 4. Training utilities
# ──────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Simple cosine-decay LR scheduler with linear warm-up."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, device, dtype):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=dtype)
        yb = yb.to(device=device, dtype=dtype)

        preds = model(xb)
        loss = nn.functional.mse_loss(preds, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=dtype)
        yb = yb.to(device=device, dtype=dtype)
        preds = model(xb)
        total_loss += nn.functional.mse_loss(preds, yb).item() * xb.size(0)
    return total_loss / len(loader.dataset)


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    _check_triton_backend()

    cfg = MambaRegressionConfig(
        # Model
        hidden_size=128,
        state_size=16,
        num_hidden_layers=4,
        expand=2,
        conv_kernel=4,
        # I/O
        input_dim=10,
        num_targets=3,
        mlp_hidden=128,
        # Training
        seq_len=128,
        batch_size=64,
        lr=3e-4,
        weight_decay=1e-2,
        num_epochs=20,
        warmup_steps=100,
        dtype="float32",       # use "bfloat16" on Ampere+ for speed
        device="cuda",
        # Data
        num_train=4096,
        num_val=512,
    )

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.dtype]
    device = cfg.device

    # --- Data -----------------------------------------------------------
    X_train, Y_train = make_synthetic_data(
        cfg.num_train, cfg.seq_len, cfg.input_dim, cfg.num_targets,
        device="cpu", dtype=dtype,
    )
    X_val, Y_val = make_synthetic_data(
        cfg.num_val, cfg.seq_len, cfg.input_dim, cfg.num_targets,
        device="cpu", dtype=dtype,
    )
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=cfg.batch_size, pin_memory=True,
    )

    # --- Model ----------------------------------------------------------
    model = MambaForRegression(cfg).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"  backbone layers : {cfg.num_hidden_layers}")
    print(f"  hidden_size     : {cfg.hidden_size}")
    print(f"  state_size      : {cfg.state_size}")
    print(f"  expand           : {cfg.expand}")
    print(f"  num_targets     : {cfg.num_targets}")
    print()

    # --- Optimizer & Scheduler ------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    # --- Training loop --------------------------------------------------
    print(f"{'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>10}  {'Time':>7}")
    print("-" * 52)

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, dtype)
        val_loss = evaluate(model, val_loader, device, dtype)
        elapsed = time.perf_counter() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  {lr_now:10.2e}  {elapsed:6.2f}s")

    # --- Quick sanity check ---------------------------------------------
    model.eval()
    with torch.no_grad():
        sample_x = X_val[:4].to(device=device, dtype=dtype)
        sample_y = Y_val[:4].to(device=device, dtype=dtype)
        preds = model(sample_x)
        print("\nSample predictions vs targets:")
        for i in range(4):
            p = preds[i].cpu().tolist()
            t = sample_y[i].cpu().tolist()
            p_str = ", ".join(f"{v:+.3f}" for v in p)
            t_str = ", ".join(f"{v:+.3f}" for v in t)
            print(f"  pred=[{p_str}]  target=[{t_str}]")


if __name__ == "__main__":
    main()