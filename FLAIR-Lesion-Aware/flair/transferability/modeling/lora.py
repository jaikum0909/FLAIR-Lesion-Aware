"""
lora.py — Low-Rank Adaptation for FLAIR Vision Encoder
-------------------------------------------------------
Applies LoRA to Conv2d layers inside ResNet-50's layer3 and layer4.
Only LoRA parameters are trained; all pretrained FLAIR weights stay frozen.

Integration point in FLAIR:
    flair_model.vision_model.model  →  ResNet-50
        .layer3  →  apply LoRA here
        .layer4  →  apply LoRA here
        .fc      →  Identity() (not touched)

Usage:
    from flair.transferability.modeling.lora import apply_lora, get_lora_params
    apply_lora(flair_model.vision_model.model, rank=4, alpha=8,
               target_layers=["layer3", "layer4"])
"""

import math
import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# LoRA Conv2d wrapper
# ---------------------------------------------------------------------------

class LoRAConv2d(nn.Module):
    """
    Wraps a Conv2d layer with a LoRA low-rank update path.
    Output = conv(x) + scale * up(down(x))
    Original conv weights are frozen; only down/up are trained.
    """
    def __init__(self, conv: nn.Conv2d, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.rank   = rank
        self.scale  = alpha / rank

        # Freeze original conv
        self.weight  = conv.weight
        self.bias    = conv.bias
        self.stride  = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups  = conv.groups
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        out_ch = conv.weight.shape[0]
        in_ch  = conv.weight.shape[1] * conv.groups

        # Low-rank path: down-project then up-project
        self.lora_down = nn.Conv2d(
            in_ch, rank,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False
        )
        self.lora_up = nn.Conv2d(rank, out_ch, kernel_size=1, bias=False)

        # Init: down ~ kaiming, up = zero (so initial output = base conv)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )
        lora = self.lora_up(self.lora_down(x)) * self.scale
        return base + lora


# ---------------------------------------------------------------------------
# LoRA Linear wrapper (for projection heads if needed)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Wraps a Linear layer with a LoRA low-rank update.
    """
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.scale = alpha / rank

        self.weight = linear.weight
        self.bias   = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = nn.functional.linear(
            nn.functional.linear(x, self.lora_A), self.lora_B
        ) * self.scale
        return base + lora


# ---------------------------------------------------------------------------
# Apply LoRA to a module
# ---------------------------------------------------------------------------

def apply_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    target_layers: List[str] = None
) -> nn.Module:
    """
    Walk the module tree and replace Conv2d layers with LoRA equivalents.
    Only replaces layers whose parent name appears in target_layers.

    Args:
        module        : The model to patch (e.g. flair_model.vision_model.model)
        rank          : LoRA rank r
        alpha         : LoRA scaling alpha
        target_layers : Only patch layers inside these named sub-modules.
                        e.g. ['layer3', 'layer4']
                        If None, patches everything.
    Returns:
        The patched module (in-place).
    """
    for name, child in list(module.named_children()):
        should_patch = (
            target_layers is None or
            any(t in name for t in target_layers)
        )
        if should_patch:
            # Recursively patch this sub-module
            _patch_module(child, rank=rank, alpha=alpha)
        else:
            # Still recurse in case target is deeper
            apply_lora(child, rank=rank, alpha=alpha, target_layers=target_layers)
    return module


def _patch_module(module: nn.Module, rank: int, alpha: float) -> None:
    """Recursively replace all Conv2d inside a module with LoRAConv2d."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha))
        else:
            _patch_module(child, rank=rank, alpha=alpha)


def freeze_non_lora(module: nn.Module) -> None:
    """Freeze every parameter that is NOT a LoRA weight."""
    for name, param in module.named_parameters():
        if "lora_down" not in name and "lora_up" not in name \
           and "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)


def get_lora_params(module: nn.Module) -> List[nn.Parameter]:
    """Return only trainable LoRA parameters (pass to optimizer)."""
    return [p for p in module.parameters() if p.requires_grad]


def lora_param_count(module: nn.Module) -> dict:
    """Count trainable vs total parameters."""
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total"     : total,
        "trainable" : trainable,
        "ratio_%"   : round(100 * trainable / max(total, 1), 3)
    }
