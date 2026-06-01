"""
lesion_attention.py — Lesion-Aware Cross-Attention Module
----------------------------------------------------------
PRIMARY NOVEL CONTRIBUTION of the thesis.

Problem (Modality Gap):
    FLAIR uses global average pooling on the vision encoder output,
    collapsing all spatial information into a single 2048-d vector.
    Text embeddings describe fine-grained lesions ("small red dots",
    "venous beading") but the image side has no mechanism to spatially
    locate those lesions. This is the "Modality Gap."

Solution:
    Before global average pooling, we capture the spatial feature map
    from ResNet-50's layer4 (B, 2048, H, W). We then run cross-attention
    where lesion text embeddings are QUERIES and spatial image patches
    are KEYS and VALUES. This forces the model to look at the exact
    spatial regions described by the clinical text — bridging the gap.

Architecture:
    ResNet layer4 → spatial map (B, 2048, H, W)
                         ↓
    Flatten → image tokens (B, HW, 2048)
                         ↓
    Cross-Attention ← lesion text queries (N_lesion, 512)
                         ↓
    Pool over lesions → attended feature (B, 256)
                         ↓
    Project → shared FLAIR space (B, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LesionCrossAttention(nn.Module):
    """
    Multi-head cross-attention: lesion text embeddings attend over
    spatial image feature map, producing a lesion-grounded representation.

    Args:
        img_feat_dim  : Channels in spatial feature map (2048 for ResNet-50 layer4)
        text_feat_dim : Dimension of text embeddings from FLAIR (512)
        hidden_dim    : Internal projection size (256)
        num_heads     : Attention heads (8)
        dropout       : Attention dropout (0.1)
    """

    def __init__(
        self,
        img_feat_dim : int   = 2048,
        text_feat_dim: int   = 512,
        hidden_dim   : int   = 256,
        num_heads    : int   = 8,
        dropout      : float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads  = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim   = hidden_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        # Project image spatial tokens → Keys, Values
        self.proj_k = nn.Linear(img_feat_dim,  hidden_dim, bias=False)
        self.proj_v = nn.Linear(img_feat_dim,  hidden_dim, bias=False)

        # Project lesion text embeddings → Queries
        self.proj_q = nn.Linear(text_feat_dim, hidden_dim, bias=False)

        # Output projection
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        # Residual: project global image feature to hidden_dim
        self.residual_proj = nn.Linear(img_feat_dim, hidden_dim)

        # Norms, dropout, FFN
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        img_spatial : torch.Tensor,                   # (B, C, H, W)
        lesion_text : torch.Tensor,                   # (N_lesion, text_dim)
        return_attn : bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            img_spatial : Spatial feature map from ResNet-50 layer4.
                          Shape: (B, 2048, H, W)
            lesion_text : Stack of lesion text embeddings (pre-computed, frozen).
                          Shape: (N_lesion, 512)
            return_attn : If True, return attention weights for Grad-CAM.

        Returns:
            attended : Lesion-attended image feature. Shape: (B, hidden_dim)
            attn_map : (B, N_lesion, H, W) if return_attn else None
        """
        B, C, H, W = img_spatial.shape
        N = lesion_text.shape[0]

        # ── 1. Flatten spatial map → image tokens ──────────────────────────
        img_tokens = img_spatial.flatten(2).permute(0, 2, 1)    # (B, HW, C)

        # ── 2. Global average pool → residual ──────────────────────────────
        img_global = img_tokens.mean(dim=1)                      # (B, C)

        # ── 3. Project to Q, K, V ──────────────────────────────────────────
        # Expand lesion text across batch: (N, d) → (B, N, d)
        text_exp = lesion_text.unsqueeze(0).expand(B, -1, -1)

        Q = self.proj_q(text_exp)    # (B, N,  hidden)
        K = self.proj_k(img_tokens)  # (B, HW, hidden)
        V = self.proj_v(img_tokens)  # (B, HW, hidden)

        # ── 4. Multi-head split ────────────────────────────────────────────
        def split_heads(x, seq_len):
            return x.view(B, seq_len, self.num_heads, self.head_dim) \
                    .permute(0, 2, 1, 3)

        HW  = H * W
        Q_h = split_heads(Q, N)     # (B, heads, N,  head_dim)
        K_h = split_heads(K, HW)    # (B, heads, HW, head_dim)
        V_h = split_heads(V, HW)    # (B, heads, HW, head_dim)

        # ── 5. Scaled dot-product attention ────────────────────────────────
        attn_logits  = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)            # (B, heads, N, HW)
        attn_weights = self.dropout(attn_weights)

        # ── 6. Weighted sum over spatial positions ─────────────────────────
        attended = torch.matmul(attn_weights, V_h)               # (B, heads, N, head_dim)
        attended = attended.permute(0, 2, 1, 3).contiguous() \
                           .view(B, N, self.hidden_dim)           # (B, N, hidden)

        # ── 7. Output projection + mean-pool over lesion queries ───────────
        attended = self.proj_out(attended).mean(dim=1)           # (B, hidden)

        # ── 8. Residual + Layer Norm ───────────────────────────────────────
        residual = self.residual_proj(img_global)                # (B, hidden)
        attended = self.norm1(attended + residual)

        # ── 9. Feed-forward ────────────────────────────────────────────────
        attended = self.norm2(attended + self.ffn(attended))     # (B, hidden)

        if return_attn:
            # Average heads, reshape to spatial: (B, N, H, W)
            attn_map = attn_weights.mean(dim=1).view(B, N, H, W)
            return attended, attn_map

        return attended, None


class SpatialHook:
    """
    Forward hook to capture the spatial feature map from ResNet-50 layer4
    (before global average pooling collapses it).

    Usage:
        hook = SpatialHook(model.vision_model.model.layer4)
        _ = model.vision_model(images)
        spatial = hook.features    # (B, 2048, H, W)
        hook.remove()
    """
    def __init__(self, layer: nn.Module):
        self.features = None
        self._handle  = layer.register_forward_hook(self._fn)

    def _fn(self, module, input, output):
        self.features = output

    def remove(self):
        self._handle.remove()


class LesionAwareFLAIR(nn.Module):
    """
    Wraps a pretrained FLAIRModel and adds:
      1. LoRA on vision encoder  (from lora.py — already applied externally)
      2. Lesion-Aware Cross-Attention between spatial features & lesion text
      3. Projection to shared FLAIR 512-d embedding space

    This module only contains the NEW components. The base FLAIR model
    (with LoRA already applied) is passed in as `flair_model`.

    Args:
        flair_model   : Pretrained FLAIRModel with LoRA already applied.
        lesion_embeds : Pre-computed lesion text embeddings. (N_lesion, 512)
        hidden_dim    : Cross-attention hidden size (256)
        num_heads     : Attention heads (8)
        proj_dim      : Output projection dim — must match FLAIR's (512)
    """

    def __init__(
        self,
        flair_model,
        lesion_embeds : torch.Tensor,
        hidden_dim    : int = 256,
        num_heads     : int = 8,
        proj_dim      : int = 512,
        dropout       : float = 0.1,
    ):
        super().__init__()
        self.flair = flair_model

        # Register lesion embeddings as a buffer (not trained, moves with device)
        self.register_buffer("lesion_embeds", lesion_embeds)

        # Cross-attention module
        self.lesion_attn = LesionCrossAttention(
            img_feat_dim  = 2048,
            text_feat_dim = proj_dim,
            hidden_dim    = hidden_dim,
            num_heads     = num_heads,
            dropout       = dropout,
        )

        # Project attended features → FLAIR's 512-d shared space
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_image_lesion(
        self,
        images     : torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through LoRA vision encoder + Lesion Cross-Attention.

        Returns:
            image_embeds : (B, 512) — L2-normalised, in FLAIR shared space
            attn_map     : (B, N_lesion, H, W) if return_attn else None
        """
        # Hook into layer4 to grab spatial features
        hook = SpatialHook(self.flair.vision_model.model.layer4)

        # Forward through vision encoder (LoRA-adapted)
        _ = self.flair.vision_model(images)
        spatial = hook.features    # (B, 2048, H, W)
        hook.remove()

        # Cross-attention with lesion text queries
        attended, attn_map = self.lesion_attn(
            img_spatial = spatial,
            lesion_text = self.lesion_embeds,
            return_attn = return_attn,
        )

        # Project to 512-d and L2-normalise
        image_embeds = self.projector(attended)
        image_embeds = F.normalize(image_embeds, dim=-1)

        return image_embeds, attn_map

    def forward(
        self,
        images      : torch.Tensor,
        text_embeds : torch.Tensor,
        return_attn : bool = False,
    ) -> dict:
        """
        Full forward: image → lesion attention → logits against text prototypes.

        Args:
            images      : (B, 3, 512, 512)
            text_embeds : (C, 512) class prototype embeddings (frozen)
            return_attn : Whether to return attention maps

        Returns dict with:
            logits       : (B, C)
            image_embeds : (B, 512)
            attn_maps    : (B, N_lesion, H, W) or None
        """
        image_embeds, attn_maps = self.encode_image_lesion(images, return_attn)

        logit_scale = self.flair.logit_scale.exp()
        logits = image_embeds @ text_embeds.t() * logit_scale

        return {
            "logits"      : logits,
            "image_embeds": image_embeds,
            "attn_maps"   : attn_maps,
        }
