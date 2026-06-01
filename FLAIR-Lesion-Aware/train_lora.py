"""
train_lora.py — LoRA Fine-tuning of FLAIR Vision Encoder
----------------------------------------------------------
Applies LoRA adapters to FLAIR's ResNet-50 vision encoder (layer3 + layer4)
and fine-tunes on a target dataset using the contrastive objective.

The text encoder (Bio_ClinicalBERT) stays completely frozen — preserving
the pre-trained medical vocabulary (key finding from our experiments).

Usage:
    python train_lora.py --experiment 13_FIVES --lora_rank 4 --epochs 10 --shots 80%

Results are saved to: local_data/results/transferability/lora/
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# FLAIR imports
from flair import FLAIRModel
from flair.transferability.modeling.lora import (
    apply_lora, freeze_non_lora, get_lora_params, lora_param_count
)
from local_data.constants import (
    PATH_DATASETS,
    PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION,
    PATH_RESULTS_TRANSFERABILITY
)
from local_data.experiments import get_experiment_setting

# Metrics
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

# Device — supports CUDA, Apple MPS (M1/M2/M3/M4), and CPU
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"[train_lora] Using device: {device}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for FLAIR")
    parser.add_argument("--experiment",  type=str, default="13_FIVES",
                        help="Experiment name (must match local_data/experiments.py)")
    parser.add_argument("--lora_rank",   type=int,   default=4,
                        help="LoRA rank r (default: 4)")
    parser.add_argument("--lora_alpha",  type=float, default=8.0,
                        help="LoRA scaling alpha (default: 8.0 = 2*rank)")
    parser.add_argument("--lora_layers", type=str,   default="layer3,layer4",
                        help="Comma-separated ResNet layers to apply LoRA")
    parser.add_argument("--epochs",      type=int,   default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate for LoRA params (default: 1e-4)")
    parser.add_argument("--batch_size",  type=int,   default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--shots",       type=str,   default="80%",
                        help="Training data fraction (default: 80%%)")
    parser.add_argument("--shots_test",  type=str,   default="20%%",
                        help="Test data fraction (default: 20%%)")
    parser.add_argument("--folds",       type=int,   default=3,
                        help="Number of cross-validation folds (default: 3)")
    parser.add_argument("--domain_knowledge", action="store_true", default=True,
                        help="Use expert domain knowledge text prompts")
    parser.add_argument("--save_model",  action="store_true", default=True,
                        help="Save LoRA weights after training")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (reusing FLAIR's existing data pipeline)
# ---------------------------------------------------------------------------

def get_dataloaders(experiment, shots_train, shots_test, batch_size, fold=0):
    """Use FLAIR's existing dataloader for the target dataset."""
    from flair.transferability.data.dataloader import get_dataloader_splits

    setting = get_experiment_setting(experiment)

    loaders = get_dataloader_splits(
        dataframe_path = setting["dataframe"],
        data_root_path = PATH_DATASETS,
        targets_dict   = setting["targets"],
        shots_train    = shots_train,
        shots_val      = "0%",
        shots_test     = shots_test,
        batch_size     = batch_size,
        num_workers    = 0,
        seed           = fold,   # different seed per fold = different split
        task           = setting["task"],
    )
    return loaders["train"], loaders["test"], setting["targets"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, epoch, text_embeds):
    """
    Fine-tune LoRA params using cross-entropy loss against frozen text prototypes.
    - text_embeds: (C, 512) pre-computed class prototype embeddings (frozen)
    - Only LoRA + projection head params are updated
    """
    model.train()
    model.text_model.eval()   # text encoder stays frozen throughout

    loss_ave = 0.0
    iterator = tqdm(loader, desc=f"Epoch {epoch} Training", dynamic_ncols=True)

    for step, batch in enumerate(iterator):
        images = batch["image"].to(torch.float32).to(device)
        labels = batch["label"].to(device).to(torch.long)

        # Forward through LoRA-adapted vision encoder
        img_embeds = model.vision_model(images)          # (B, 512)

        # Similarity logits against text prototypes
        logit_scale = model.logit_scale.exp()
        logits = img_embeds @ text_embeds.t() * logit_scale  # (B, C)

        # Cross-entropy loss with ground-truth labels
        loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(get_lora_params(model), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        loss_ave += loss.item()
        iterator.set_description(f"Epoch {epoch} | loss: {loss.item():.4f}")

    return loss_ave / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, targets, domain_knowledge=True):
    """Zero-shot style evaluation using text prototypes + LoRA vision encoder."""
    model.eval()

    # Build text prototype embeddings on CPU (BERT is kept on CPU for MPS compat)
    _, text_embeds = model.compute_text_embeddings(
        list(targets.keys()), domain_knowledge=domain_knowledge
    )
    text_embeds = text_embeds.to(device)   # move result to MPS/CUDA after encoding

    refs, preds = [], []
    iterator = tqdm(loader, desc="Evaluating", dynamic_ncols=True)

    for batch in iterator:
        images = batch["image"].to(torch.float32).to(device)
        labels = batch["label"].numpy()

        img_embeds = model.vision_model(images)
        logit_scale = model.logit_scale.exp()
        logits = img_embeds @ text_embeds.t() * logit_scale
        probs  = torch.softmax(logits, dim=-1)

        refs.extend(labels)
        preds.extend(probs.cpu().numpy())

    refs  = np.array(refs)
    preds = np.array(preds)
    pred_labels = preds.argmax(axis=1)

    # Metrics
    aca   = float(np.mean([
        accuracy_score(refs[refs == c], pred_labels[refs == c])
        for c in np.unique(refs)
    ]))
    try:
        kappa = float(cohen_kappa_score(refs, pred_labels, weights="quadratic"))
    except Exception:
        kappa = float(cohen_kappa_score(refs, pred_labels))
    f1    = float(f1_score(refs, pred_labels, average="macro", zero_division=0))

    return {"aca": aca, "kappa": kappa, "f1": f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    lora_target_layers = args.lora_layers.split(",")

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-tuning: {args.experiment}")
    print(f"  rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  target layers: {lora_target_layers}")
    print(f"  epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    print(f"{'='*60}\n")

    fold_results = []

    for fold in range(args.folds):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")

        # ── 1. Load pretrained FLAIR (stays on CPU initially) ────────────────
        model = FLAIRModel.from_pretrained("jusiro2/FLAIR")

        # ── 2. Freeze everything ─────────────────────────────────────────────
        for param in model.parameters():
            param.requires_grad_(False)

        # ── 3. Apply LoRA ONLY to vision encoder (layer3 + layer4) ───────────
        # IMPORTANT: Apply LoRA BEFORE moving to device so new LoRA layers
        # are created on CPU first, then moved to MPS together with the model.
        apply_lora(
            model.vision_model.model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_layers=lora_target_layers
        )
        # Also make projection head trainable
        for param in model.vision_model.projection_head_vision.parameters():
            param.requires_grad_(True)

        # ── 4. NOW move to device (vision → MPS, text → stays CPU) ───────────
        # ClinicalBERT is incompatible with MPS, so text model stays on CPU.
        # LoRA layers are now included in vision_model, so they move to MPS too.
        model.vision_model.to(device)
        model.logit_scale.data = model.logit_scale.data.to(device)

        stats = lora_param_count(model)
        print(f"  Trainable: {stats['trainable']:,} params ({stats['ratio_%']}% of total)")

        # ── 4. Data loaders ───────────────────────────────────────────────────
        loader_train, loader_test, targets = get_dataloaders(
            args.experiment, args.shots, args.shots_test, args.batch_size, fold=fold
        )

        # ── 5. Pre-compute text prototype embeddings (done ONCE, frozen) ─────
        print("  Computing text prototypes...")
        with torch.no_grad():
            _, text_embeds = model.compute_text_embeddings(
                list(targets.keys()), domain_knowledge=args.domain_knowledge
            )
        text_embeds = text_embeds.to(device)
        print(f"  Text prototypes: {text_embeds.shape}")

        # ── 6. Optimizer (only LoRA + projection head params) ─────────────────
        optimizer = torch.optim.AdamW(
            get_lora_params(model),
            lr=args.lr,
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        # ── 7. Training loop ──────────────────────────────────────────────────
        best_aca, best_metrics = 0.0, {}
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, loader_train, optimizer, epoch, text_embeds)
            metrics = evaluate(model, loader_test, targets, args.domain_knowledge)
            scheduler.step()

            print(
                f"  Epoch {epoch:02d} | loss={loss:.4f} | "
                f"ACA={metrics['aca']:.4f} | Kappa={metrics['kappa']:.4f} | F1={metrics['f1']:.4f}"
            )

            if metrics["aca"] > best_aca:
                best_aca     = metrics["aca"]
                best_metrics = metrics.copy()

                # Save best LoRA weights for this fold
                if args.save_model:
                    save_dir = os.path.join(
                        PATH_RESULTS_TRANSFERABILITY, "lora", args.experiment
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"lora_fold{fold}.pth")
                    # Save only LoRA parameters (lightweight)
                    lora_state = {
                        k: v for k, v in model.state_dict().items()
                        if "lora_down" in k or "lora_up" in k
                    }
                    torch.save(lora_state, save_path)
                    print(f"  Saved LoRA weights → {save_path}")

        print(f"\n  Best results (fold {fold+1}): "
              f"ACA={best_metrics.get('aca',0):.4f} | "
              f"Kappa={best_metrics.get('kappa',0):.4f} | "
              f"F1={best_metrics.get('f1',0):.4f}")
        fold_results.append(best_metrics)

    # ── 7. Cross-validation summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RESULTS (cross-validation)")
    print(f"{'='*60}")
    for metric in ["aca", "kappa", "f1"]:
        vals = [r[metric] for r in fold_results]
        print(f"  {metric.upper():<8}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
