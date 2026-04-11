#!/usr/bin/env python3
"""
Train CLIP-frozen visual-only CSN pipeline.

- Superclass similarity on full projection embeddings.
- Subclass similarity on shared-mask CSN embeddings.
- Visual branch only: CLIP image encoder -> projection head -> shared CSN mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# local CLIP checkout
CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore

from csn_pipeline.data import create_or_load_split, load_csn_records
from csn_pipeline.losses import SupConLoss, two_view_supcon_loss
from csn_pipeline.model import ProjectionHead, SharedCSNMask


# ============================================================
# Data structures
# ============================================================

@dataclass
class LossBundle:
    total: float
    super_simclr: float
    subclass_simclr: float


@dataclass
class TrainState:
    epoch: int
    best_subclass_metric: float
    best_subclass_epoch: int
    best_subclass_metrics: dict[str, Any]
    train_history: list[dict[str, Any]]
    val_history: list[dict[str, Any]]


# ============================================================
# Dataset
# ============================================================

class CSNIndexMultiViewDataset(Dataset):
    """Multi-view dataset returning global record indices for (anchor, super+, cat+)."""

    def __init__(self, records, indices: np.ndarray, seed: int = 0):
        self.records = records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = random.Random(seed)

        self.super_to_local: dict[int, list[int]] = {}
        self.cat_to_local: dict[int, list[int]] = {}
        for local_pos, global_idx in enumerate(self.indices.tolist()):
            rec = self.records[int(global_idx)]
            self.super_to_local.setdefault(int(rec.superclass_id), []).append(local_pos)
            self.cat_to_local.setdefault(int(rec.category_id), []).append(local_pos)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _sample_positive_local(self, pool: list[int], anchor_local: int) -> int:
        candidates = [i for i in pool if i != anchor_local]
        if not candidates:
            return anchor_local
        return self.rng.choice(candidates)

    def __getitem__(self, local_idx: int) -> dict[str, int]:
        anchor_global = int(self.indices[local_idx])
        anchor = self.records[anchor_global]

        super_pool = self.super_to_local[int(anchor.superclass_id)]
        cat_pool = self.cat_to_local[int(anchor.category_id)]

        pos_super_local = self._sample_positive_local(super_pool, int(local_idx))
        pos_cat_local = self._sample_positive_local(cat_pool, int(local_idx))

        super_global = int(self.indices[pos_super_local])
        cat_global = int(self.indices[pos_cat_local])

        return {
            "idx_view1": anchor_global,
            "idx_view2": super_global,
            "idx_view3": cat_global,
            "label_view1_2": int(anchor.superclass_id),
            "label_view1_3": int(anchor.category_id),
        }


def collate_csn_index_batch(batch: list[dict[str, int]]) -> dict[str, torch.Tensor]:
    """Stack a list of per-sample dicts into a batched dict of tensors."""
    return {
        "idx_view1": torch.tensor([b["idx_view1"] for b in batch], dtype=torch.long),
        "idx_view2": torch.tensor([b["idx_view2"] for b in batch], dtype=torch.long),
        "idx_view3": torch.tensor([b["idx_view3"] for b in batch], dtype=torch.long),
        "label_view1_2": torch.tensor([b["label_view1_2"] for b in batch], dtype=torch.long),
        "label_view1_3": torch.tensor([b["label_view1_3"] for b in batch], dtype=torch.long),
    }


# ============================================================
# Device / seed utilities
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (including all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, model_name: str) -> torch.device:
    """Select the best available device, applying safety fallbacks for MPS/ResNet combinations."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            d = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            d = "mps"
        else:
            d = "cpu"
    else:
        d = device_arg

    if d == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
        print("Warning: MPS requested but unavailable. Falling back to CPU.")
        d = "cpu"

    if d == "mps" and model_name.strip().upper().startswith("RN"):
        print("Warning: ResNet models have MPS issues. Falling back to CPU.")
        d = "cpu"

    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    return torch.device(d)


def freeze_clip_model(model: torch.nn.Module) -> None:
    """Disable gradients on all CLIP parameters and set the model to eval mode."""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


# ============================================================
# Data split + CLIP feature cache utilities
# ============================================================

def default_split_dir_for_csv(csv_file: str | Path, seed: int, train_ratio: float) -> Path:
    """Return the canonical split directory path derived from the CSV filename, seed, and ratio."""
    csv_path = Path(csv_file).resolve()
    ratio_tag = int(round(float(train_ratio) * 100))
    return csv_path.parent / f"{csv_path.stem}_splits_seed{int(seed)}_tr{ratio_tag}"


def create_or_load_balanced_validation_split(
    records,
    test_indices: np.ndarray,
    split_dir: str | Path,
    seed: int,
    force_resplit: bool = False,
    samples_per_subclass: int | None = None,
    min_samples_per_subclass: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Carve a balanced validation set from test_indices (equal samples per subclass).

    Loads existing split files if they exist and force_resplit is False.
    Returns (val_indices, holdout_test_indices, metadata_dict).
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    val_path = split_dir / "val_indices_balanced.npy"
    holdout_path = split_dir / "holdout_test_indices.npy"
    meta_path = split_dir / "val_split_metadata.json"

    if val_path.exists() and holdout_path.exists() and meta_path.exists() and not force_resplit:
        val_idx = np.load(val_path).astype(np.int64)
        holdout_idx = np.load(holdout_path).astype(np.int64)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        max_idx = len(records) - 1
        if val_idx.size == 0:
            raise ValueError("Existing validation split is empty.")
        if int(val_idx.max()) > max_idx or (holdout_idx.size > 0 and int(holdout_idx.max()) > max_idx):
            raise ValueError("Existing validation split indices exceed current record count. Use --force-resplit.")
        return val_idx, holdout_idx, meta

    rng = random.Random(seed + 17)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    by_subclass: dict[int, list[int]] = {}
    for idx in test_indices.tolist():
        by_subclass.setdefault(int(records[int(idx)].category_id), []).append(int(idx))

    eligible = {sid: idxs for sid, idxs in by_subclass.items() if len(idxs) >= int(min_samples_per_subclass)}
    if not eligible:
        raise ValueError(
            f"No test subclasses have at least {int(min_samples_per_subclass)} samples for balanced validation."
        )

    min_count = min(len(idxs) for idxs in eligible.values())
    if samples_per_subclass is None:
        samples_per_subclass = int(min_count)
    else:
        samples_per_subclass = int(samples_per_subclass)
        if samples_per_subclass < int(min_samples_per_subclass):
            raise ValueError("--val-samples-per-subclass must be >= --val-min-samples-per-subclass")
        samples_per_subclass = min(samples_per_subclass, int(min_count))

    val_idx: list[int] = []
    holdout_idx: list[int] = []
    excluded_subclasses = 0
    excluded_samples = 0

    for subclass_id, idxs in sorted(by_subclass.items(), key=lambda x: x[0]):
        shuffled = list(int(i) for i in idxs)
        rng.shuffle(shuffled)
        if len(shuffled) < int(min_samples_per_subclass):
            excluded_subclasses += 1
            excluded_samples += len(shuffled)
            holdout_idx.extend(shuffled)
            continue

        chosen = shuffled[:samples_per_subclass]
        remainder = shuffled[samples_per_subclass:]
        val_idx.extend(chosen)
        holdout_idx.extend(remainder)

    val_idx_np = np.asarray(sorted(val_idx), dtype=np.int64)
    holdout_idx_np = np.asarray(sorted(holdout_idx), dtype=np.int64)
    if val_idx_np.size == 0:
        raise ValueError("Balanced validation split produced zero samples.")

    meta = {
        "seed": int(seed),
        "source_test_count": int(test_indices.shape[0]),
        "val_count": int(val_idx_np.shape[0]),
        "holdout_test_count": int(holdout_idx_np.shape[0]),
        "num_subclasses_total": int(len(by_subclass)),
        "num_subclasses_in_validation": int(len(eligible)),
        "excluded_singleton_subclasses": int(excluded_subclasses),
        "excluded_singleton_samples": int(excluded_samples),
        "samples_per_subclass": int(samples_per_subclass),
        "min_samples_per_subclass": int(min_samples_per_subclass),
        "val_indices_path": str(val_path),
        "holdout_test_indices_path": str(holdout_path),
    }

    np.save(val_path, val_idx_np)
    np.save(holdout_path, holdout_idx_np)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return val_idx_np, holdout_idx_np, meta


def _records_path_hash(records) -> str:
    """Return a SHA-1 hash of all image paths, used to detect cache invalidation."""
    h = hashlib.sha1()
    for rec in records:
        h.update(str(rec.image_path).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_or_build_clip_image_cache(
    model: torch.nn.Module,
    preprocess,
    records,
    model_name: str,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return pre-computed CLIP image features for every record as a CPU float32 tensor.

    Loads from cache_path if the file exists and metadata matches; otherwise
    encodes all images with the given CLIP model and writes cache_path to disk.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".meta.json")

    expected = {
        "model": str(model_name),
        "num_records": int(len(records)),
        "records_path_hash": _records_path_hash(records),
    }

    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            arr = np.load(cache_path)
            if (
                arr.ndim == 2
                and int(arr.shape[0]) == expected["num_records"]
                and str(meta.get("model")) == expected["model"]
                and str(meta.get("records_path_hash")) == expected["records_path_hash"]
            ):
                print(f"Loaded CLIP image cache: {cache_path}  shape={arr.shape}")
                return torch.from_numpy(np.asarray(arr, dtype=np.float32))
            print("CLIP image cache metadata mismatch; rebuilding cache.")
        except Exception as e:
            print(f"Failed to load CLIP image cache ({e}); rebuilding cache.")

    print(f"Building CLIP image cache for {len(records)} records...")
    feats_cpu: list[torch.Tensor] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size), desc="CLIP image cache"):
            batch_records = records[i : i + batch_size]
            imgs = []
            for rec in batch_records:
                try:
                    img = Image.open(rec.image_path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), color="black")
                imgs.append(preprocess(img))

            x = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)
            f = model.encode_image(x).float().cpu()
            feats_cpu.append(f)

    features = torch.cat(feats_cpu, dim=0).contiguous()
    np.save(cache_path, features.numpy())

    meta = {
        **expected,
        "feature_dim": int(features.shape[1]),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved CLIP image cache: {cache_path}  shape={tuple(features.shape)}")
    return features


def collect_subclass_embeddings(
    clip_feature_cache: torch.Tensor,
    records,
    indices: np.ndarray,
    image_head: torch.nn.Module,
    subclass_head: torch.nn.Module | None,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Project cached CLIP features through image_head (and optionally subclass_head).

    Returns (embeddings [N, D], subclass_ids [N]) for all requested indices.
    """
    device = next(image_head.parameters()).device
    idx_np = np.asarray(indices, dtype=np.int64)
    embeddings: list[np.ndarray] = []
    subclass_ids: list[int] = []

    with torch.no_grad():
        for start in range(0, idx_np.shape[0], batch_size):
            batch_idx = idx_np[start : start + batch_size]
            idx_tensor = torch.from_numpy(batch_idx).long()
            feat = clip_feature_cache.index_select(0, idx_tensor).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            proj = image_head(feat)
            emb = subclass_head(proj) if subclass_head is not None else proj
            embeddings.append(emb.float().cpu().numpy())
            subclass_ids.extend(int(records[int(i)].category_id) for i in batch_idx.tolist())

    return np.vstack(embeddings).astype(np.float32), np.asarray(subclass_ids, dtype=np.int64)


def compute_retrieval_metrics_at_k(
    embeddings: np.ndarray,
    labels_eval: np.ndarray,
    k_list: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], list[int]]:
    """Compute precision, recall, and F1 at each k via cosine-similarity retrieval.

    Returns (precision, recall, f1, clipped_k) where clipped_k lists any k
    values that exceeded N-1 and were silently reduced.
    """
    emb = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
    emb = F.normalize(emb, dim=1)
    labels_t = torch.from_numpy(np.asarray(labels_eval, dtype=np.int64)).to(device)
    n = emb.shape[0]

    if n < 2:
        raise ValueError("Need at least 2 validation samples for retrieval metrics.")

    requested_k = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not requested_k:
        raise ValueError("k_list must contain at least one positive integer.")

    max_allowed_k = n - 1
    max_k = max(min(k, max_allowed_k) for k in requested_k)

    recall_hits = {k: 0 for k in requested_k}
    precision_sum = {k: 0.0 for k in requested_k}

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        query = emb[i:end]
        query_labels = labels_t[i:end]
        sim = query @ emb.T
        for j in range(sim.shape[0]):
            sim[j, i + j] = float("-inf")
        topk_idx = torch.topk(sim, max_k, dim=1).indices
        retrieved_labels = labels_t[topk_idx]
        same = retrieved_labels == query_labels.unsqueeze(1)

        for k_req in requested_k:
            kk = min(k_req, max_allowed_k)
            top_same = same[:, :kk]
            recall_hits[k_req] += int(top_same.any(dim=1).sum().item())
            precision_sum[k_req] += float(top_same.float().mean(dim=1).sum().item())

    recall = {k: recall_hits[k] / n for k in requested_k}
    precision = {k: precision_sum[k] / n for k in requested_k}
    f1 = {
        k: (0.0 if (precision[k] + recall[k]) <= 0 else 2.0 * precision[k] * recall[k] / (precision[k] + recall[k]))
        for k in requested_k
    }
    clipped_k = sorted(k for k in requested_k if k > max_allowed_k)
    return precision, recall, f1, clipped_k


def summarize_subclass_retrieval(
    clip_feature_cache: torch.Tensor,
    records,
    val_indices: np.ndarray,
    image_head: torch.nn.Module,
    subclass_head: torch.nn.Module | None,
    device: torch.device,
    k_values: list[int],
    batch_size: int = 512,
) -> dict[str, Any]:
    """Embed validation samples and return a summary dict of precision/recall/F1 at each k."""
    embeddings, subclass_ids = collect_subclass_embeddings(
        clip_feature_cache=clip_feature_cache,
        records=records,
        indices=val_indices,
        image_head=image_head,
        subclass_head=subclass_head,
        batch_size=batch_size,
    )
    precision, recall, f1, clipped_k = compute_retrieval_metrics_at_k(
        embeddings=embeddings,
        labels_eval=subclass_ids,
        k_list=k_values,
        device=device,
        batch_size=batch_size,
    )
    f1_mean = float(np.mean([f1[k] for k in sorted(f1)]))
    return {
        "subclass_precision_at_k": {str(k): float(v) for k, v in precision.items()},
        "subclass_recall_at_k": {str(k): float(v) for k, v in recall.items()},
        "subclass_f1_at_k": {str(k): float(v) for k, v in f1.items()},
        "subclass_f1_mean": f1_mean,
        "clipped_k": [int(k) for k in clipped_k],
        "num_validation_samples": int(val_indices.shape[0]),
    }


# ============================================================
# Validation metrics
# ============================================================

def rename_user_facing_terms(obj: Any) -> Any:
    """Recursively replace internal 'category'/'cat_' keys with user-facing 'subclass' equivalents."""
    if isinstance(obj, dict):
        return {rename_user_facing_terms(k): rename_user_facing_terms(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename_user_facing_terms(v) for v in obj]
    if isinstance(obj, str):
        out = obj.replace("category", "subclass").replace("Category", "Subclass")
        out = out.replace("w_cat_", "w_subclass_").replace("_cat_", "_subclass_")
        out = out.replace("cat_", "subclass_")
        return out
    return obj


# ============================================================
# Training loop
# ============================================================

def compute_losses(
    clip_feature_cache: torch.Tensor,
    batch: dict[str, torch.Tensor],
    image_head: torch.nn.Module,
    csn_mask: torch.nn.Module,
    supcon_loss: SupConLoss,
    weights: dict[str, float],
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted superclass + subclass SimCLR losses for one batch.

    Returns (total_loss, {component_name: loss_tensor}).
    """
    device = next(image_head.parameters()).device

    idx_views = torch.stack(
        [batch["idx_view1"], batch["idx_view2"], batch["idx_view3"]],
        dim=1,
    )
    labels_super = batch["label_view1_2"].to(device=device, non_blocking=True)
    labels_cat = batch["label_view1_3"].to(device=device, non_blocking=True)

    bsz, n_views = idx_views.shape
    idx_flat = idx_views.view(-1).cpu()

    with torch.cuda.amp.autocast(enabled=use_amp):
        img_feat_flat = clip_feature_cache.index_select(0, idx_flat).to(device=device, dtype=torch.float32, non_blocking=True)

        img_proj = image_head(img_feat_flat).view(bsz, n_views, -1)
        img_masked = csn_mask(img_proj.view(bsz * n_views, -1)).view(bsz, n_views, -1)

        l_super_simclr = two_view_supcon_loss(supcon_loss, img_proj[:, 0], img_proj[:, 1], labels_super)
        l_cat_simclr = two_view_supcon_loss(supcon_loss, img_masked[:, 0], img_masked[:, 2], labels_cat)

        total = (
            weights["w_super_simclr"] * l_super_simclr
            + weights["w_cat_simclr"] * l_cat_simclr
        )

    parts = {
        "super_simclr": l_super_simclr,
        "subclass_simclr": l_cat_simclr,
    }
    return total, parts


def run_epoch(
    loader: DataLoader,
    clip_feature_cache: torch.Tensor,
    image_head: torch.nn.Module,
    csn_mask: torch.nn.Module,
    supcon_loss: SupConLoss,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    train: bool,
    epoch: int,
    total_epochs: int,
) -> LossBundle:
    """Run one full pass over loader in train or eval mode and return averaged losses."""
    if train:
        image_head.train()
        csn_mask.train()
        mode = f"Train {epoch}/{total_epochs}"
    else:
        image_head.eval()
        csn_mask.eval()
        mode = f"Validation {epoch}/{total_epochs}"

    running = {
        "total": 0.0,
        "super_simclr": 0.0,
        "subclass_simclr": 0.0,
    }
    n_batches = 0

    pbar = tqdm(loader, desc=mode, total=len(loader))
    for batch in pbar:
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            total, parts = compute_losses(
                clip_feature_cache=clip_feature_cache,
                batch=batch,
                image_head=image_head,
                csn_mask=csn_mask,
                supcon_loss=supcon_loss,
                weights=weights,
                use_amp=use_amp,
            )

            if train:
                if use_amp:
                    scaler.scale(total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    optimizer.step()

        running["total"] += float(total.item())
        for k in ("super_simclr", "subclass_simclr"):
            running[k] += float(parts[k].item())
        n_batches += 1

        pbar.set_postfix(total=f"{running['total']/n_batches:.4f}")

    if n_batches == 0:
        raise RuntimeError("No batches processed in epoch.")

    return LossBundle(
        total=running["total"] / n_batches,
        super_simclr=running["super_simclr"] / n_batches,
        subclass_simclr=running["subclass_simclr"] / n_batches,
    )


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    image_head: ProjectionHead,
    csn_mask: SharedCSNMask,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    train_state: TrainState,
    is_best: bool,
) -> None:
    """Write per-epoch, latest, and (optionally) best checkpoint files under out_dir/checkpoints/."""
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "model_name": args.model,
        "visual_only": True,
        "image_head_state_dict": image_head.state_dict(),
        "csn_mask_state_dict": csn_mask.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_state": {
            "epoch": train_state.epoch,
            "best_subclass_metric": train_state.best_subclass_metric,
            "best_subclass_epoch": train_state.best_subclass_epoch,
            "best_subclass_metrics": train_state.best_subclass_metrics,
            "train_history": train_state.train_history,
            "val_history": train_state.val_history,
        },
        "split_dir": str(args.split_dir) if args.split_dir else None,
        "seed": int(args.seed),
        "args": vars(args),
    }

    epoch_path = ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    latest_path = ckpt_dir / "latest_checkpoint.pt"
    torch.save(ckpt, epoch_path)
    torch.save(ckpt, latest_path)
    if is_best:
        best_path = ckpt_dir / "best_checkpoint.pt"
        torch.save(ckpt, best_path)


def save_loss_log(out_dir: Path, train_state: TrainState) -> None:
    """Serialise the full training history and best-model summary to loss_log.json."""
    log_path = out_dir / "loss_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "best_subclass_metric_name": "subclass_f1_mean",
                "best_subclass_metric": train_state.best_subclass_metric,
                "best_subclass_epoch": train_state.best_subclass_epoch,
                "best_subclass_metrics": train_state.best_subclass_metrics,
                "train_history": train_state.train_history,
                "val_history": train_state.val_history,
            },
            f,
            indent=2,
        )


# ============================================================
# Plotting
# ============================================================

def plot_loss_curves(out_dir: Path, train_history: list[dict[str, Any]], val_history: list[dict[str, Any]]) -> None:
    """Plot and save train/validation total-loss curves to loss_curve.png."""
    if not train_history:
        return

    plt.figure(figsize=(10, 6))
    train_epochs = [x["epoch"] for x in train_history]
    train_total = [x["total"] for x in train_history]
    plt.plot(train_epochs, train_total, label="train_total", linewidth=2)

    if val_history:
        val_epochs = [x["epoch"] for x in val_history]
        val_total = [x["total"] for x in val_history]
        plt.plot(val_epochs, val_total, label="validation_total", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CSN training/validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_subclass_retrieval_curves(
    out_dir: Path,
    val_history: list[dict[str, Any]],
    k_values: list[int],
) -> None:
    """Plot precision/recall/F1 at each k over epochs and save to subclass_retrieval_metrics.png."""
    if not val_history:
        return

    epochs = [int(row["epoch"]) for row in val_history]
    metric_specs = [
        ("subclass_precision_at_k", "Validation subclass precision@k"),
        ("subclass_recall_at_k", "Validation subclass recall@k"),
        ("subclass_f1_at_k", "Validation subclass F1@k"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, (metric_key, title) in zip(axes, metric_specs):
        for k in k_values:
            vals = [float(row.get(metric_key, {}).get(str(k), 0.0)) for row in val_history]
            ax.plot(epochs, vals, marker="o", linewidth=1.8, label=f"k={k}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "subclass_retrieval_metrics.png", dpi=150)
    plt.close(fig)


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Train frozen-CLIP visual-only CSN pipeline")

    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)
    parser.add_argument("--force-resplit", action="store_true")

    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--mask-init", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--linear-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for projection linear layers (image_head). Mask uses --weight-decay.",
    )
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--w-super-simclr", type=float, default=1.0)
    parser.add_argument("--w-super-it", type=float, default=1.0)
    parser.add_argument("--w-cat-simclr", type=float, default=1.0)
    parser.add_argument("--w-cat-it", type=float, default=1.0)

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-samples-per-subclass", type=int, default=None)
    parser.add_argument("--val-min-samples-per-subclass", type=int, default=2)
    parser.add_argument("--val-metric-batch-size", type=int, default=512)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default="./training_output")
    parser.add_argument("--experiment-name", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    """Full training loop: data loading, CLIP caching, model init, train/val epochs, checkpointing."""
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device, args.model)
    use_amp = bool(args.amp and device.type == "cuda")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.experiment_name:
        exp_dir = out_root / args.experiment_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace("@", "_")
        exp_dir = out_root / f"{model_safe}_csn_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = 0.5
    if args.split_dir is None:
        args.split_dir = str(default_split_dir_for_csv(args.csv_file, args.seed, train_ratio))

    print(f"Output dir: {exp_dir}")
    print(f"Device: {device}  AMP(cuda-only): {use_amp}")
    print(f"Split dir: {args.split_dir}")

    # data
    records, data_stats = load_csn_records(args.csv_file, args.base_image_dir)
    train_idx, test_idx, split_meta = create_or_load_split(
        records=records,
        split_dir=args.split_dir,
        seed=args.seed,
        force_resplit=args.force_resplit,
        train_ratio=train_ratio,
    )
    val_idx, holdout_test_idx, val_split_meta = create_or_load_balanced_validation_split(
        records=records,
        test_indices=test_idx,
        split_dir=args.split_dir,
        seed=args.seed,
        force_resplit=args.force_resplit,
        samples_per_subclass=args.val_samples_per_subclass,
        min_samples_per_subclass=args.val_min_samples_per_subclass,
    )
    print(
        f"Loaded records: {len(records)}  train={len(train_idx)} test={len(test_idx)} "
        f"validation={len(val_idx)} holdout_test={len(holdout_test_idx)}"
    )
    print(
        "Balanced validation split: "
        f"subclasses={val_split_meta['num_subclasses_in_validation']}  "
        f"samples_per_subclass={val_split_meta['samples_per_subclass']}"
    )

    # model and CLIP cache
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model = model.float()
    freeze_clip_model(model)

    model_safe = args.model.replace("/", "_").replace("@", "_")
    split_dir_path = Path(args.split_dir)
    cache_path = split_dir_path / f"{model_safe}_clip_image_features.npy"

    clip_feature_cache = load_or_build_clip_image_cache(
        model=model,
        preprocess=preprocess,
        records=records,
        model_name=args.model,
        cache_path=cache_path,
        device=device,
        batch_size=max(int(args.batch_size), 64),
    )

    img_dim = int(clip_feature_cache.shape[1])
    if device.type == "cuda":
        clip_feature_cache = clip_feature_cache.pin_memory()

    # CLIP model no longer needed after feature cache is ready
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_head = ProjectionHead(img_dim, args.hidden_dim, args.proj_dim).to(device).float()
    csn_mask = SharedCSNMask(args.proj_dim, mask_init=args.mask_init).to(device).float()

    if args.w_super_it != 0.0 or args.w_cat_it != 0.0:
        print("Note: --w-super-it and --w-cat-it are ignored in visual-only mode.")

    train_ds = CSNIndexMultiViewDataset(records=records, indices=train_idx, seed=args.seed)
    val_ds = CSNIndexMultiViewDataset(records=records, indices=val_idx, seed=args.seed + 1)

    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_csn_index_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_csn_index_batch,
    )

    supcon_loss = SupConLoss(temperature=args.temperature).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(image_head.parameters()), "weight_decay": float(args.linear_weight_decay)},
            {"params": list(csn_mask.parameters()), "weight_decay": float(args.weight_decay)},
        ],
        lr=args.lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    weights = {
        "w_super_simclr": float(args.w_super_simclr),
        "w_cat_simclr": float(args.w_cat_simclr),
    }

    k_values = [1, 10, 100, 1000]
    state = TrainState(
        epoch=0,
        best_subclass_metric=float("-inf"),
        best_subclass_epoch=0,
        best_subclass_metrics={},
        train_history=[],
        val_history=[],
    )
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        image_head.load_state_dict(ckpt["image_head_state_dict"], strict=True)
        csn_mask.load_state_dict(ckpt["csn_mask_state_dict"], strict=True)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: could not load optimizer state from resume checkpoint ({e}); continuing with fresh optimizer.")
        if use_amp and ckpt.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                print(f"Warning: could not load AMP scaler state ({e}); continuing with fresh scaler.")

        ts_data = ckpt.get("train_state", {})
        state = TrainState(
            epoch=int(ts_data.get("epoch", ckpt.get("epoch", 0))),
            best_subclass_metric=float(ts_data.get("best_subclass_metric", float("-inf"))),
            best_subclass_epoch=int(ts_data.get("best_subclass_epoch", 0)),
            best_subclass_metrics=dict(ts_data.get("best_subclass_metrics", {})),
            train_history=list(ts_data.get("train_history", [])),
            val_history=list(ts_data.get("val_history", ts_data.get("test_history", []))),
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    # save static config now
    config = {
        "args": rename_user_facing_terms(vars(args)),
        "device": str(device),
        "amp_enabled": use_amp,
        "data_stats": data_stats,
        "split_meta": split_meta,
        "val_split_meta": val_split_meta,
        "img_embed_dim": img_dim,
        "visual_only": True,
        "clip_image_cache_path": str(cache_path),
        "validation_k_values": k_values,
        "best_subclass_selection_metric": "subclass_f1_mean",
        "ignored_weights": rename_user_facing_terms({
            "w_super_it": float(args.w_super_it),
            "w_cat_it": float(args.w_cat_it),
        }),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(
            loader=train_loader,
            clip_feature_cache=clip_feature_cache,
            image_head=image_head,
            csn_mask=csn_mask,
            supcon_loss=supcon_loss,
            weights=weights,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            train=True,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        train_row = {"epoch": epoch, **asdict(train_loss)}
        state.train_history.append(train_row)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        metric_for_best = float("-inf")
        if do_eval:
            val_loss = run_epoch(
                loader=val_loader,
                clip_feature_cache=clip_feature_cache,
                image_head=image_head,
                csn_mask=csn_mask,
                supcon_loss=supcon_loss,
                weights=weights,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                train=False,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            subclass_metrics = summarize_subclass_retrieval(
                clip_feature_cache=clip_feature_cache,
                records=records,
                val_indices=val_idx,
                image_head=image_head,
                subclass_head=csn_mask,
                device=device,
                k_values=k_values,
                batch_size=args.val_metric_batch_size,
            )
            val_row = {"epoch": epoch, **asdict(val_loss), **subclass_metrics}
            state.val_history.append(val_row)
            metric_for_best = float(subclass_metrics["subclass_f1_mean"])

        is_best = metric_for_best > state.best_subclass_metric
        if is_best:
            state.best_subclass_metric = metric_for_best
            state.best_subclass_epoch = epoch
            state.best_subclass_metrics = dict(state.val_history[-1]) if state.val_history else {}

        state.epoch = epoch

        if (epoch % args.save_every == 0) or (epoch == args.epochs) or is_best:
            save_checkpoint(
                out_dir=exp_dir,
                epoch=epoch,
                args=args,
                image_head=image_head,
                csn_mask=csn_mask,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                train_state=state,
                is_best=is_best,
            )

        save_loss_log(exp_dir, state)
        plot_loss_curves(exp_dir, state.train_history, state.val_history)
        plot_subclass_retrieval_curves(exp_dir, state.val_history, k_values)

        dt = time.perf_counter() - t0
        latest_val = state.val_history[-1] if state.val_history else None
        val_summary = ""
        if latest_val is not None:
            val_summary = (
                f" val_subclass_f1_mean={float(latest_val['subclass_f1_mean']):.4f}"
                f" best_subclass_epoch={state.best_subclass_epoch}"
                f" best_subclass_f1_mean={state.best_subclass_metric:.4f}"
            )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_total={train_loss.total:.4f} "
            f"train_subclass_simclr={train_loss.subclass_simclr:.4f}"
            f"{val_summary} "
            f"time={dt:.1f}s"
        )

    print("Training complete")
    print(f"Best subclass epoch: {state.best_subclass_epoch}  best_subclass_f1_mean={state.best_subclass_metric:.4f}")
    if state.best_subclass_metrics:
        print(f"Best subclass metrics by k: {json.dumps(state.best_subclass_metrics['subclass_f1_at_k'], sort_keys=True)}")
    print(f"Artifacts: {exp_dir}")


if __name__ == "__main__":
    main()
