"""
gradcam_viz.py — Grad-CAM + Lesion Attention Visualisation
-----------------------------------------------------------
Produces side-by-side visualisations for sample images from each dataset:
  Column 1 : Original fundus image
  Column 2 : Grad-CAM on ResNet-50 layer4  (what the model focuses on)
  Column 3 : Lesion Attention map          (which spatial patches the lesion
                                            queries attend to)
  Column 4 : Grad-CAM × Lesion Attn blend (combined)

Usage:
    python gradcam_viz.py \
        --experiment 13_FIVES \
        --fold 0 \
        --n_samples 4 \
        --out_dir local_data/results/gradcam

The script loads the best saved weights for the experiment/fold and picks
n_samples images from the test split (one per class where possible).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from flair import FLAIRModel
from flair.transferability.modeling.lora import apply_lora
from flair.transferability.modeling.lesion_attention import (
    LesionAwareFLAIR, LesionCrossAttention, SpatialHook
)
from local_data.constants import (
    PATH_DATASETS, PATH_RESULTS_TRANSFERABILITY
)
from local_data.experiments import get_experiment_setting
from flair.transferability.data.dataloader import get_dataloader_splits

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"[gradcam_viz] device: {device}")


# ── helpers ───────────────────────────────────────────────────────────────────

def build_lesion_embeddings(flair_model, device):
    from flair.modeling.dictionary import definitions
    lesion_keys = [
        "microaneurysms", "haemorrhages", "hard exudates",
        "soft exudates", "cotton wool spots", "drusens",
        "optic disc cupping", "optic disc edema",
        "mild diabetic retinopathy", "moderate diabetic retinopathy",
        "severe diabetic retinopathy", "proliferative diabetic retinopathy",
        "age related macular degeneration", "macular hole",
        "preretinal haemorrhage", "exudates",
    ]
    caption = flair_model.caption
    embeds  = []
    flair_model.text_model.eval()
    with torch.no_grad():
        for key in lesion_keys:
            descs   = definitions.get(key, [key])
            prompts = [caption.replace("[CLS]", d) for d in descs]
            tokens  = flair_model.text_model.tokenizer(
                prompts, truncation=True, padding=True, return_tensors='pt'
            )
            emb = flair_model.text_model(
                tokens["input_ids"].to('cpu'),
                tokens["attention_mask"].to('cpu')
            )
            embeds.append(emb.mean(0))
    return torch.stack(embeds).to(device)


def load_model(experiment, fold, args):
    """Load FLAIRModel + LoRA + LesionAttn weights."""
    setting = get_experiment_setting(experiment)

    base = FLAIRModel.from_pretrained("jusiro2/FLAIR")
    for p in base.parameters():
        p.requires_grad_(False)

    apply_lora(
        base.vision_model.model,
        rank=args.lora_rank, alpha=args.lora_alpha,
        target_layers=args.lora_layers.split(",")
    )
    for p in base.vision_model.projection_head_vision.parameters():
        p.requires_grad_(True)

    base.vision_model.to(device)
    base.logit_scale.data = base.logit_scale.data.to(device)
    base.text_model.to('cpu')

    lesion_embeds = build_lesion_embeddings(base, device)

    model = LesionAwareFLAIR(
        flair_model   = base,
        lesion_embeds = lesion_embeds,
        hidden_dim    = args.hidden_dim,
        num_heads     = args.num_heads,
    )
    model.lesion_attn.to(device)
    model.projector.to(device)
    model.lesion_embeds = model.lesion_embeds.to(device)

    # Load saved weights
    ckpt_path = os.path.join(
        PATH_RESULTS_TRANSFERABILITY, "lesion", experiment,
        f"lesion_fold{fold}.pth"
    )
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded weights from {ckpt_path}")
        if missing:
            print(f"  Missing keys (expected for frozen base): {len(missing)}")
    else:
        print(f"  [WARNING] No checkpoint at {ckpt_path}. Using random weights.")

    model.eval()

    # Compute text prototypes
    with torch.no_grad():
        _, text_embeds = base.compute_text_embeddings(
            list(setting["targets"].keys()),
            domain_knowledge=True
        )
    text_embeds = text_embeds.to(device)

    return model, text_embeds, setting


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM on ResNet-50 layer4 for LesionAwareFLAIR.

    Because we need gradients, we temporarily re-enable grad computation
    even during visualisation.
    """

    def __init__(self, model: LesionAwareFLAIR):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []

        # Hook only the forward pass — gradients are captured via tensor hook
        layer = model.flair.vision_model.model.layer4
        self._hooks.append(
            layer.register_forward_hook(self._save_activation)
        )

    def _save_activation(self, module, input, output):
        # retain_grad() keeps the gradient on this non-leaf tensor after backward
        output.retain_grad()
        self.activations = output

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, images: torch.Tensor, text_embeds: torch.Tensor,
                 target_class: int = None):
        """
        Returns:
            cam_maps     : (B, H, W) numpy, values in [0,1]
            attn_maps    : (B, H, W) numpy, values in [0,1]  (mean over lesion queries)
            pred_classes : (B,) int numpy
        """
        self.model.eval()
        images = images.to(device)

        with torch.enable_grad():
            out = self.model(images, text_embeds, return_attn=True)
            logits    = out["logits"]        # (B, C)
            attn_maps = out["attn_maps"]     # (B, N_lesion, H, W)

            pred_classes = logits.argmax(dim=-1)  # (B,)
            targets = pred_classes if target_class is None \
                      else torch.full_like(pred_classes, target_class)

            # One-hot score for backward
            score = (logits * F.one_hot(targets, logits.shape[-1]).float()).sum()
            self.model.zero_grad()
            score.backward()

        # Grad-CAM: weight activations by mean gradient over spatial dims
        # activations.grad is populated because we called retain_grad()
        activations = self.activations.detach()
        gradients   = self.activations.grad  # (B, C, H, W)
        if gradients is None:
            # Fallback: use uniform weights (still shows activation pattern)
            print("  [warn] gradients not available — using activation-only CAM")
            gradients = torch.ones_like(activations)
        weights = gradients.mean(dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
        cam     = (weights * activations).sum(dim=1)              # (B, H, W)
        cam     = F.relu(cam)

        # Normalise each map to [0,1]
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1).values.view(B, 1, 1)
        cam_max  = cam_flat.max(dim=1).values.view(B, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam_np   = cam.cpu().numpy()

        # Lesion attention: average over lesion queries → (B, H, W)
        attn_np = attn_maps.detach().mean(dim=1).cpu().numpy()  # (B, H, W)
        attn_np = (attn_np - attn_np.min(axis=(1,2), keepdims=True)) / \
                  (attn_np.max(axis=(1,2), keepdims=True) - attn_np.min(axis=(1,2), keepdims=True) + 1e-8)

        return cam_np, attn_np, pred_classes.cpu().numpy()


# ── visualisation ─────────────────────────────────────────────────────────────

def overlay_heatmap(img_np, heatmap, alpha=0.45, colormap='jet'):
    """
    img_np  : (H, W, 3) float [0,1]
    heatmap : (h, w) float [0,1]  — will be resized to img size
    Returns : (H, W, 3) float [0,1]
    """
    H, W = img_np.shape[:2]
    # Resize heatmap
    hm_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    hm_pil = hm_pil.resize((W, H), Image.BILINEAR)
    hm     = np.array(hm_pil) / 255.0

    cmap   = matplotlib.colormaps[colormap]
    hm_rgb = cmap(hm)[:, :, :3]
    blend  = (1 - alpha) * img_np + alpha * hm_rgb
    return np.clip(blend, 0, 1)


def denorm_image(tensor):
    """Convert normalised tensor (3, H, W) → numpy (H, W, 3) in [0,1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    return np.clip(img, 0, 1)


def make_figure(images_t, cam_maps, attn_maps, labels, pred_labels,
                class_names, out_path):
    """
    images_t   : (B, 3, H, W) tensor
    cam_maps   : (B, h, w) numpy
    attn_maps  : (B, h, w) numpy
    labels     : (B,) int
    pred_labels: (B,) int
    class_names: list of str
    """
    B   = images_t.shape[0]
    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grad-CAM", "Lesion Attention", "Blend"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight='bold', pad=8)

    for i in range(B):
        img_np   = denorm_image(images_t[i])
        true_lbl = class_names[labels[i]]
        pred_lbl = class_names[pred_labels[i]]
        correct  = "✓" if labels[i] == pred_labels[i] else "✗"
        row_title = f"True: {true_lbl}  |  Pred: {pred_lbl} {correct}"

        # Col 0: original
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(row_title, fontsize=9, rotation=0,
                               labelpad=160, va='center')

        # Col 1: Grad-CAM
        axes[i, 1].imshow(overlay_heatmap(img_np, cam_maps[i], colormap='jet'))

        # Col 2: Lesion attention
        axes[i, 2].imshow(overlay_heatmap(img_np, attn_maps[i], colormap='hot'))

        # Col 3: blend of both
        blend = 0.5 * cam_maps[i] + 0.5 * attn_maps[i]
        blend = (blend - blend.min()) / (blend.max() - blend.min() + 1e-8)
        axes[i, 3].imshow(overlay_heatmap(img_np, blend, colormap='RdYlGn'))

        for c in range(4):
            axes[i, c].axis('off')

    plt.suptitle("FLAIR + LoRA + Lesion-Aware Cross-Attention — Interpretability",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment",   type=str, default="13_FIVES")
    p.add_argument("--fold",         type=int, default=0)
    p.add_argument("--n_samples",    type=int, default=4,
                   help="Number of images to visualise")
    p.add_argument("--lora_rank",    type=int,   default=4)
    p.add_argument("--lora_alpha",   type=float, default=8.0)
    p.add_argument("--lora_layers",  type=str,   default="layer3,layer4")
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--num_heads",    type=int,   default=8)
    p.add_argument("--shots_test",   type=str,   default="20%")
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--out_dir",      type=str,
                   default="local_data/results/gradcam")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Grad-CAM + Lesion Attention: {args.experiment} (fold {args.fold})")
    print(f"{'='*55}\n")

    # Load model
    model, text_embeds, setting = load_model(args.experiment, args.fold, args)
    class_names = list(setting["targets"].keys())

    # Data loader — grab test split
    loaders = get_dataloader_splits(
        dataframe_path = setting["dataframe"],
        data_root_path = PATH_DATASETS,
        targets_dict   = setting["targets"],
        shots_train    = "0%",
        shots_val      = "0%",
        shots_test     = args.shots_test,
        batch_size     = args.batch_size,
        num_workers    = 0,
        seed           = args.fold,
        task           = setting["task"],
    )
    loader = loaders["test"]

    # Collect one sample per class (up to n_samples total)
    collected   = {}
    images_list = []
    labels_list = []

    for batch in loader:
        imgs = batch["image"].to(torch.float32)
        lbls = batch["label"].numpy()
        for j in range(len(lbls)):
            c = int(lbls[j])
            if c not in collected:
                collected[c] = True
                images_list.append(imgs[j])
                labels_list.append(c)
            if len(images_list) >= args.n_samples:
                break
        if len(images_list) >= args.n_samples:
            break

    images_t = torch.stack(images_list)   # (N, 3, H, W)
    labels   = np.array(labels_list)

    # Run Grad-CAM
    gcam = GradCAM(model)
    print("  Running Grad-CAM forward+backward...")
    cam_maps, attn_maps, pred_labels = gcam(images_t, text_embeds)
    gcam.remove()

    # Save figure
    out_path = os.path.join(
        args.out_dir,
        f"{args.experiment}_fold{args.fold}_gradcam.png"
    )
    make_figure(images_t, cam_maps, attn_maps,
                labels, pred_labels, class_names, out_path)

    # Print per-sample summary
    print(f"\n  Sample results:")
    for i in range(len(labels)):
        status = "CORRECT" if labels[i] == pred_labels[i] else "WRONG"
        print(f"    [{i+1}] True={class_names[labels[i]]:<25} "
              f"Pred={class_names[pred_labels[i]]:<25} {status}")

    print(f"\n  Output saved to: {out_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
