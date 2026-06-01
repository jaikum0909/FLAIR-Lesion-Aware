"""
train_lesion.py — FLAIR + LoRA + Lesion-Aware Cross-Attention
--------------------------------------------------------------
Builds on top of the trained LoRA weights and adds the Lesion-Aware
Cross-Attention module on top.

Training is staged:
  Stage 1 (warm-up, epochs 1-3):  Train ONLY the LesionCrossAttention
                                   + projector. LoRA weights frozen.
  Stage 2 (joint, epochs 4-10):  Train LesionAttn + projector + LoRA
                                   together with a lower LR.

This staged approach prevents the randomly initialised attention module
from corrupting the already-trained LoRA weights in early epochs.

Usage:
    python train_lesion.py --experiment 13_FIVES --lora_rank 4 --epochs 10
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from flair import FLAIRModel
from flair.transferability.modeling.lora import (
    apply_lora, get_lora_params, lora_param_count
)
from flair.transferability.modeling.lesion_attention import LesionAwareFLAIR
from flair.modeling.dictionary import definitions

from local_data.constants import (
    PATH_DATASETS,
    PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION,
    PATH_RESULTS_TRANSFERABILITY
)
from local_data.experiments import get_experiment_setting
from flair.transferability.data.dataloader import get_dataloader_splits

from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

# Device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"[train_lesion] Using device: {device}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",    type=str,   default="13_FIVES")
    parser.add_argument("--lora_rank",     type=int,   default=4)
    parser.add_argument("--lora_alpha",    type=float, default=8.0)
    parser.add_argument("--lora_layers",   type=str,   default="layer3,layer4")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--warmup_epochs", type=int,   default=3,
                        help="Epochs to train only attention (LoRA frozen)")
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--lr_lora",       type=float, default=2e-5,
                        help="Lower LR for LoRA in joint stage")
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--shots",         type=str,   default="80%%")
    parser.add_argument("--shots_test",    type=str,   default="20%%")
    parser.add_argument("--folds",         type=int,   default=3)
    parser.add_argument("--hidden_dim",    type=int,   default=256)
    parser.add_argument("--num_heads",     type=int,   default=8)
    parser.add_argument("--domain_knowledge", action="store_true", default=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Build lesion text embeddings from FLAIR's dictionary
# ---------------------------------------------------------------------------

def build_lesion_embeddings(flair_model, n_lesions: int = 20) -> torch.Tensor:
    """
    Pull the most clinically relevant lesion descriptions from FLAIR's
    built-in expert knowledge dictionary and encode them with ClinicalBERT.

    Uses the same lesion terms FLAIR was pre-trained with — no new knowledge
    needed, just reusing what FLAIR already knows.

    Returns: (N_lesion, 512) on CPU
    """
    # Key lesion types relevant to DR, AMD, Glaucoma
    lesion_keys = [
        "microaneurysms", "haemorrhages", "hard exudates",
        "soft exudates", "cotton wool spots", "drusens",
        "optic disc cupping", "optic disc edema",
        "mild diabetic retinopathy", "moderate diabetic retinopathy",
        "severe diabetic retinopathy", "proliferative diabetic retinopathy",
        "age related macular degeneration", "macular hole",
        "preretinal haemorrhage", "exudates",
    ]

    caption = flair_model.caption   # "A fundus photograph of [CLS]"
    embeds  = []

    flair_model.text_model.eval()
    with torch.no_grad():
        for key in lesion_keys:
            if key in definitions:
                descs = definitions[key]
            else:
                descs = [key]

            prompts = [caption.replace("[CLS]", d) for d in descs]
            tokens  = flair_model.text_model.tokenizer(
                prompts, truncation=True, padding=True, return_tensors='pt'
            )
            ids   = tokens["input_ids"].to('cpu')
            masks = tokens["attention_mask"].to('cpu')
            emb   = flair_model.text_model(ids, masks)   # (n_desc, 512)
            embeds.append(emb.mean(0))                    # average descriptions

    lesion_embeds = torch.stack(embeds, dim=0)           # (N_lesion, 512)
    print(f"  Built {lesion_embeds.shape[0]} lesion embeddings from FLAIR dictionary")
    return lesion_embeds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, text_embeds):
    model.eval()
    refs, preds = [], []

    for batch in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        images = batch["image"].to(torch.float32).to(device)
        labels = batch["label"].numpy()

        out    = model(images, text_embeds)
        probs  = F.softmax(out["logits"], dim=-1)

        refs.extend(labels)
        preds.extend(probs.cpu().numpy())

    refs        = np.array(refs)
    pred_labels = np.array(preds).argmax(axis=1)

    aca = float(np.mean([
        accuracy_score(refs[refs == c], pred_labels[refs == c])
        for c in np.unique(refs)
    ]))
    try:
        kappa = float(cohen_kappa_score(refs, pred_labels, weights="quadratic"))
    except Exception:
        kappa = float(cohen_kappa_score(refs, pred_labels))
    f1 = float(f1_score(refs, pred_labels, average="macro", zero_division=0))

    return {"aca": aca, "kappa": kappa, "f1": f1}


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, epoch, text_embeds):
    model.train()
    model.flair.text_model.eval()   # text always frozen

    loss_ave = 0.0
    iterator = tqdm(loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

    for batch in iterator:
        images = batch["image"].to(torch.float32).to(device)
        labels = batch["label"].to(device).to(torch.long)

        out    = model(images, text_embeds)
        loss   = torch.nn.functional.cross_entropy(out["logits"], labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()

        loss_ave += loss.item()
        iterator.set_description(f"Epoch {epoch} | loss: {loss.item():.4f}")

    return loss_ave / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = get_args()
    lora_target_layers = args.lora_layers.split(",")

    print(f"\n{'='*60}")
    print(f"  FLAIR + LoRA + Lesion Attention: {args.experiment}")
    print(f"  rank={args.lora_rank}, hidden={args.hidden_dim}, heads={args.num_heads}")
    print(f"  warmup={args.warmup_epochs} epochs, total={args.epochs} epochs")
    print(f"{'='*60}\n")

    setting = get_experiment_setting(args.experiment)
    fold_results = []

    for fold in range(args.folds):
        print(f"\n--- Fold {fold + 1}/{args.folds} ---")

        # ── 1. Load FLAIR, freeze everything ─────────────────────────────────
        base = FLAIRModel.from_pretrained("jusiro2/FLAIR")
        for param in base.parameters():
            param.requires_grad_(False)

        # ── 2. Apply LoRA to vision encoder ──────────────────────────────────
        apply_lora(
            base.vision_model.model,
            rank          = args.lora_rank,
            alpha         = args.lora_alpha,
            target_layers = lora_target_layers
        )
        for param in base.vision_model.projection_head_vision.parameters():
            param.requires_grad_(True)

        # ── 3. Load previously saved LoRA weights (from train_lora.py) ───────
        lora_path = os.path.join(
            PATH_RESULTS_TRANSFERABILITY, "lora",
            args.experiment, f"lora_fold{fold}.pth"
        )
        if os.path.exists(lora_path):
            lora_state = torch.load(lora_path, map_location="cpu")
            missing, _ = base.load_state_dict(lora_state, strict=False)
            print(f"  Loaded LoRA weights from {lora_path}")
        else:
            print(f"  [WARNING] No LoRA weights found at {lora_path}. Starting fresh.")

        # ── 4. Move to device (vision → MPS/CUDA, text → CPU) ────────────────
        base.vision_model.to(device)
        base.logit_scale.data = base.logit_scale.data.to(device)

        # ── 5. Build lesion embeddings (CPU, frozen) ──────────────────────────
        print("  Building lesion embeddings...")
        lesion_embeds = build_lesion_embeddings(base)   # (N, 512) on CPU
        lesion_embeds = lesion_embeds.to(device)

        # ── 6. Build LesionAwareFLAIR model ──────────────────────────────────
        # NOTE: Do NOT call .to(device) on the full model — that would move
        # ClinicalBERT to MPS which is incompatible. Instead move only the
        # new modules (lesion_attn, projector) to device individually.
        model = LesionAwareFLAIR(
            flair_model   = base,
            lesion_embeds = lesion_embeds,
            hidden_dim    = args.hidden_dim,
            num_heads     = args.num_heads,
        )
        model.lesion_attn.to(device)
        model.projector.to(device)
        # lesion_embeds buffer also needs to be on device
        model.lesion_embeds = model.lesion_embeds.to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params (attn+proj+lora): {total_params:,}")

        # ── 7. Data loaders ───────────────────────────────────────────────────
        loaders = get_dataloader_splits(
            dataframe_path = setting["dataframe"],
            data_root_path = PATH_DATASETS,
            targets_dict   = setting["targets"],
            shots_train    = args.shots,
            shots_val      = "0%",
            shots_test     = args.shots_test,
            batch_size     = args.batch_size,
            num_workers    = 0,
            seed           = fold,
            task           = setting["task"],
        )
        loader_train, loader_test = loaders["train"], loaders["test"]

        # ── 8. Pre-compute class text prototypes on CPU then move to device ────
        # text_model stays on CPU (MPS incompatible), result moved after
        base.text_model.to('cpu')
        with torch.no_grad():
            _, text_embeds = base.compute_text_embeddings(
                list(setting["targets"].keys()),
                domain_knowledge=args.domain_knowledge
            )
        text_embeds = text_embeds.to(device)

        # ── 9. Staged training ────────────────────────────────────────────────
        best_aca, best_metrics = 0.0, {}

        for epoch in range(1, args.epochs + 1):

            # Stage 1 (warm-up): train ONLY attention + projector
            if epoch <= args.warmup_epochs:
                if epoch == 1:
                    print(f"\n  [Stage 1 — Warm-up] Training attention only (LoRA frozen)")
                    # Freeze LoRA params
                    for n, p in model.named_parameters():
                        if "lora_down" in n or "lora_up" in n:
                            p.requires_grad_(False)
                    attn_params = [
                        p for n, p in model.named_parameters()
                        if p.requires_grad and
                        ("lesion_attn" in n or "projector" in n)
                    ]
                    optimizer = torch.optim.AdamW(
                        attn_params, lr=args.lr, weight_decay=1e-5
                    )

            # Stage 2 (joint): train attention + projector + LoRA together
            elif epoch == args.warmup_epochs + 1:
                print(f"\n  [Stage 2 — Joint] Training attention + LoRA together")
                for n, p in model.named_parameters():
                    if "lora_down" in n or "lora_up" in n:
                        p.requires_grad_(True)
                optimizer = torch.optim.AdamW([
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and
                                ("lesion_attn" in n or "projector" in n)],
                     "lr": args.lr},
                    {"params": [p for n, p in model.named_parameters()
                                if p.requires_grad and
                                ("lora_down" in n or "lora_up" in n)],
                     "lr": args.lr_lora},
                ], weight_decay=1e-5)

            # Train one epoch
            loss    = train_one_epoch(model, loader_train, optimizer, epoch, text_embeds)
            metrics = evaluate(model, loader_test, text_embeds)

            print(
                f"  Epoch {epoch:02d} | loss={loss:.4f} | "
                f"ACA={metrics['aca']:.4f} | Kappa={metrics['kappa']:.4f} | "
                f"F1={metrics['f1']:.4f}"
            )

            if metrics["aca"] > best_aca:
                best_aca     = metrics["aca"]
                best_metrics = metrics.copy()

                # Save best weights
                save_dir = os.path.join(
                    PATH_RESULTS_TRANSFERABILITY, "lesion",
                    args.experiment
                )
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {k: v for k, v in model.state_dict().items()
                     if any(x in k for x in ["lora_down", "lora_up",
                                              "lesion_attn", "projector"])},
                    os.path.join(save_dir, f"lesion_fold{fold}.pth")
                )

        print(f"\n  Best (fold {fold+1}): "
              f"ACA={best_metrics.get('aca',0):.4f} | "
              f"Kappa={best_metrics.get('kappa',0):.4f} | "
              f"F1={best_metrics.get('f1',0):.4f}")
        fold_results.append(best_metrics)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {args.experiment}")
    print(f"{'='*60}")
    for metric in ["aca", "kappa", "f1"]:
        vals = [r[metric] for r in fold_results]
        print(f"  {metric.upper():<8}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
