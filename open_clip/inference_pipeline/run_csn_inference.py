#!/usr/bin/env python3
"""CSN inference: recall/precision, optional distributions, t-SNE, and tail-sample analysis.

Also includes:
  A) Neighbourhood purity  — top-k breakdown: same-cat / same-sup-diff-cat / diff-sup
  B) Cluster shape          — covariance traces + per-sample distance to class center
  C) Border mass density    — intra/inter-superclass margin distributions (approximated)
  D) Embedding anisotropy   — SVD spectrum, participation ratio, per-dim activity

Supports:
- Single-space mode via --embeddings + --labels
- Multi-space CSN mode via --prefix-metadata from generate_csn_embeddings.py
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import umap as _umap_module
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

_CAT_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "h", "p"]


# ============================================================
# Utilities
# ============================================================

def resolve_device(device_arg: str) -> torch.device:
    """Return the best available torch.device, or the explicitly requested one."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_array(path: Path) -> np.ndarray:
    """Load a .npy or single-array .npz file and return a plain ndarray."""
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if len(arr.files) != 1:
            raise ValueError(f"Expected one array in npz: {path}")
        arr = arr[arr.files[0]]
    return np.asarray(arr)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row; clamps norm to 1e-12 to avoid division by zero."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


# ============================================================
# Embedding diagnostics
# ============================================================

def compute_bits_left_stats(
    embeddings: np.ndarray,
    eps_list: tuple[float, ...] = (1e-2, 1e-3, 1e-4),
) -> dict[str, float | int | dict[str, float | int]]:
    """Compute per-sample active-dimension counts and global L2-norm statistics."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [N, D]")

    abs_emb = np.abs(embeddings)
    l2_norms = np.linalg.norm(embeddings, axis=1)

    per_sample_active: dict[str, dict[str, float | int]] = {}
    for eps in eps_list:
        counts = np.sum(abs_emb > eps, axis=1).astype(np.int64)
        key = f"eps_{eps:g}"
        per_sample_active[key] = {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
        }

    dim_std = np.std(embeddings, axis=0)
    globally_active_dims = int(np.sum(dim_std > 1e-6))

    return {
        "dim_total": int(embeddings.shape[1]),
        "embedding_l2_norm": {
            "mean": float(np.mean(l2_norms)),
            "std": float(np.std(l2_norms)),
            "min": float(np.min(l2_norms)),
            "max": float(np.max(l2_norms)),
        },
        "per_sample_active_dims": per_sample_active,
        "globally_active_dims_std_gt_1e-6": globally_active_dims,
        "globally_active_frac_std_gt_1e-6": float(globally_active_dims / max(embeddings.shape[1], 1)),
    }


# ============================================================
# Retrieval metrics
# ============================================================

def compute_retrieval_metrics_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[int, float], dict[int, float], list[int], dict, dict]:
    """Compute recall@k and precision@k with per-class breakdowns.

    Returns (recall, precision, clipped_k, per_class_recall, per_class_precision).
    """
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    labels_t = torch.from_numpy(labels).to(device)
    n = emb.shape[0]

    if n < 2:
        raise ValueError("Need at least 2 samples for retrieval metrics")

    requested_k = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not requested_k:
        raise ValueError("k_list must include at least one positive integer")

    max_allowed_k = n - 1
    max_k = max(min(k, max_allowed_k) for k in requested_k)

    recall_hits = {k: 0 for k in requested_k}
    precision_sum = {k: 0.0 for k in requested_k}

    per_class_recall_hits: dict[int, dict[int, float]] = defaultdict(lambda: {k: 0.0 for k in k_list})
    per_class_prec_sum: dict[int, dict[int, float]] = defaultdict(lambda: {k: 0.0 for k in k_list})
    per_class_count: dict[int, int] = defaultdict(int)

    for i in tqdm(range(0, n, batch_size), desc="Recall/Precision@k"):
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

        # per-class accumulation
        hits_np = {k_req: same[:, :min(k_req, max_allowed_k)].any(dim=1).cpu().numpy()
                   for k_req in k_list}
        prec_np = {k_req: same[:, :min(k_req, max_allowed_k)].float().mean(dim=1).cpu().numpy()
                   for k_req in k_list}
        batch_labels_cpu = labels[i:end]
        for bi, lbl in enumerate(batch_labels_cpu):
            lbl = int(lbl)
            per_class_count[lbl] += 1
            for k_req in k_list:
                per_class_recall_hits[lbl][k_req] += float(hits_np[k_req][bi])
                per_class_prec_sum[lbl][k_req] += float(prec_np[k_req][bi])

    recall = {k: recall_hits[k] / n for k in requested_k}
    precision = {k: precision_sum[k] / n for k in requested_k}
    clipped_k = sorted(k for k in requested_k if k > max_allowed_k)
    per_class_recall = {
        int(c): {str(k): per_class_recall_hits[c][k] / max(per_class_count[c], 1) for k in k_list}
        for c in per_class_count
    }
    per_class_precision = {
        int(c): {str(k): per_class_prec_sum[c][k] / max(per_class_count[c], 1) for k in k_list}
        for c in per_class_count
    }
    return recall, precision, clipped_k, per_class_recall, per_class_precision


def compute_match_nonmatch_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    block_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (match_sims, nonmatch_sims) from the upper-triangle pairwise cosine matrix."""
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    n = emb.shape[0]
    cosine_vals = []
    match_flags = []

    for i in tqdm(range(0, n, block_size), desc="Match/non-match similarity"):
        end_i = min(i + block_size, n)
        block_i = emb[i:end_i]
        for j in range(i, n, block_size):
            end_j = min(j + block_size, n)
            block_j = emb[j:end_j]
            sim = block_i @ block_j.T
            if i == j:
                mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
            else:
                mask = torch.ones_like(sim).bool()
            sim_vals = sim[mask].cpu().numpy()
            cosine_vals.append(sim_vals)
            ids_i = labels[i:end_i]
            ids_j = labels[j:end_j]
            label_block = ids_i[:, None] == ids_j[None, :]
            match_flags.append(label_block[mask.cpu().numpy()])

    cosine_vals = np.concatenate(cosine_vals)
    match_flags = np.concatenate(match_flags)
    return cosine_vals[match_flags], cosine_vals[~match_flags]


def plot_match_nonmatch_distribution(match_sims: np.ndarray, nonmatch_sims: np.ndarray, out_path: Path, bins: int = 50) -> None:
    """Plot overlapping histograms of match vs non-match cosine similarities and save to out_path."""
    plt.figure(figsize=(8, 4))
    plt.hist(match_sims, bins=bins, alpha=0.5, density=True, label="Match")
    plt.hist(nonmatch_sims, bins=bins, alpha=0.5, density=True, label="Non-match")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Match vs Non-match Similarity Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# Annotation / visualisation helpers
# ============================================================

def load_annotation(csv_path: Path) -> dict:
    """Parse labels_annotation.csv into color/marker/name mappings."""
    sup_id_to_name: dict[int, str] = {}
    cat_id_to_sup_id: dict[int, int] = {}
    cat_id_to_name: dict[int, str] = {}
    sup_to_cats: dict[int, list[int]] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cur_sup_id: int = -1
        for row in reader:
            sid_raw = row["Superclass ID"].strip()
            if sid_raw:
                cur_sup_id = int(sid_raw)
                sup_id_to_name[cur_sup_id] = row["Superclass"].strip()
                sup_to_cats.setdefault(cur_sup_id, [])
            cat_id = int(row["Category ID"].strip())
            cat_id_to_sup_id[cat_id] = cur_sup_id
            cat_id_to_name[cat_id] = row["Category"].strip()
            sup_to_cats[cur_sup_id].append(cat_id)

    cmap10 = plt.cm.get_cmap("tab10")
    sup_id_to_color = {s: cmap10(i / 10) for i, s in enumerate(sorted(sup_id_to_name))}

    cat_id_to_marker: dict[int, str] = {}
    for cats in sup_to_cats.values():
        for j, cat_id in enumerate(cats):
            cat_id_to_marker[cat_id] = _CAT_MARKERS[j % len(_CAT_MARKERS)]

    return {
        "sup_id_to_name":   sup_id_to_name,
        "sup_id_to_color":  sup_id_to_color,
        "cat_id_to_sup_id": cat_id_to_sup_id,
        "cat_id_to_name":   cat_id_to_name,
        "cat_id_to_marker": cat_id_to_marker,
    }


def run_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    subsample: int | None = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run UMAP, optionally subsampled. Returns (umap_2d, labels_subset, indices_used)."""
    if not _UMAP_AVAILABLE:
        raise RuntimeError("umap-learn is not installed. Run: pip install umap-learn")
    N = embeddings.shape[0]
    if subsample is not None and N > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=subsample, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        idx = np.arange(N)
        X = embeddings
        y = labels

    valid_mask = np.isfinite(X).all(axis=1)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        print(f"Warning: dropping {n_invalid}/{X.shape[0]} UMAP points with NaN/Inf values.")
        X = X[valid_mask]
        y = y[valid_mask]
        idx = idx[valid_mask]

    reducer = _umap_module.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=random_state, verbose=False,
    )
    X_2d = reducer.fit_transform(X.astype(np.float32))
    return X_2d, y, idx


def _scatter_annotated(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    annotation: dict,
    tail_mask: np.ndarray,
) -> None:
    """Per-group scatter: superclass color + category marker shape + two-part legend."""
    unique_labels = {int(l) for l in np.unique(labels)}
    is_cat = unique_labels.issubset(annotation["cat_id_to_sup_id"])

    seen_sups: set[int] = set()
    seen_cats: set[int] = set()

    for uid in sorted(unique_labels):
        mask_all = labels == uid
        normal_mask = mask_all & ~tail_mask
        tail_sub   = mask_all & tail_mask

        if is_cat:
            sup_id = annotation["cat_id_to_sup_id"].get(uid, 0)
            color  = annotation["sup_id_to_color"].get(sup_id, "gray")
            marker = annotation["cat_id_to_marker"].get(uid, "o")
            seen_sups.add(sup_id)
            seen_cats.add(uid)
        else:
            color  = annotation["sup_id_to_color"].get(uid, "gray")
            marker = "o"
            seen_sups.add(uid)

        if normal_mask.any():
            ax.scatter(xy[normal_mask, 0], xy[normal_mask, 1],
                       color=color, s=10, alpha=0.55, marker=marker)
        if tail_sub.any():
            ax.scatter(xy[tail_sub, 0], xy[tail_sub, 1],
                       color=color, s=80, alpha=0.95,
                       marker="*", edgecolors="black", linewidths=0.5)

    # Legend 1: superclass color patches
    sup_handles = [
        mpatches.Patch(color=annotation["sup_id_to_color"][s],
                       label=annotation["sup_id_to_name"].get(s, str(s)))
        for s in sorted(seen_sups) if s in annotation["sup_id_to_color"]
    ]
    leg1 = ax.legend(handles=sup_handles, title="Superclass",
                     loc="upper left", fontsize=7, title_fontsize=8,
                     framealpha=0.8, borderpad=0.6)
    ax.add_artist(leg1)

    # Legend 2: category marker shapes (only in category-label mode)
    if is_cat and seen_cats:
        cat_handles = [
            mlines.Line2D([], [],
                          color=annotation["sup_id_to_color"].get(
                              annotation["cat_id_to_sup_id"].get(c, 0), "gray"),
                          marker=annotation["cat_id_to_marker"].get(c, "o"),
                          linestyle="None", markersize=6,
                          label=annotation["cat_id_to_name"].get(c, str(c)))
            for c in sorted(seen_cats)
        ]
        n_cols = max(1, (len(cat_handles) + 5) // 6)
        ax.legend(handles=cat_handles, title="Sub-class",
                  loc="lower right", fontsize=6, title_fontsize=7,
                  framealpha=0.8, borderpad=0.6, ncol=n_cols)
    elif tail_mask.any():
        ax.legend(
            handles=[mlines.Line2D([], [], color="gray", marker="*",
                                    linestyle="None", markersize=8, label="Tail samples")],
            loc="lower right", fontsize=7, framealpha=0.8,
        )


def run_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    subsample: int | None = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally subsample and run t-SNE. Returns (coords_2d, labels_subset, indices_used)."""
    n = embeddings.shape[0]
    if subsample is not None and n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        x = embeddings[idx]
        y = labels[idx]
    else:
        idx = np.arange(n)
        x = embeddings
        y = labels

    # Cast to float64: sklearn's randomized SVD (used for t-SNE PCA init) is numerically
    # unstable with float32 when embeddings are L2-normalized (ill-conditioned covariance).
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    x_2d = tsne.fit_transform(x.astype(np.float64))
    return x_2d, y, idx


def plot_tsne(
    x_2d: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    tail_mask: np.ndarray | None = None,
    center_labels: np.ndarray | None = None,
    center_label_name: str = "Center IDs",
    contour_labels: np.ndarray | None = None,
    contour_label_name: str = "Superclass 90% contours",
    title: str = "t-SNE of Embeddings",
    annotation: dict | None = None,
) -> None:
    if tail_mask is None:
        tail_mask = np.zeros(labels.shape[0], dtype=bool)
    else:
        tail_mask = np.asarray(tail_mask, dtype=bool).flatten()
        if tail_mask.shape[0] != labels.shape[0]:
            raise ValueError("tail_mask length must match labels length")

    fig, ax = plt.subplots(figsize=(10, 8))

    if annotation is not None:
        _scatter_annotated(ax, x_2d, labels, annotation, tail_mask)
    else:
        cmap = "tab20" if np.unique(labels).size <= 20 else "viridis"
        normal_mask = ~tail_mask
        sc = None
        if normal_mask.any():
            sc = ax.scatter(x_2d[normal_mask, 0], x_2d[normal_mask, 1],
                            c=labels[normal_mask], s=8, alpha=0.55, marker="o", cmap=cmap)
        if tail_mask.any():
            sc_t = ax.scatter(x_2d[tail_mask, 0], x_2d[tail_mask, 1],
                              c=labels[tail_mask], s=70, alpha=0.95, marker="*",
                              edgecolors="black", linewidths=0.5, cmap=cmap, label="Tail samples")
            if sc is None:
                sc = sc_t
            ax.legend(loc="best")
        if sc is not None:
            fig.colorbar(sc, ax=ax, label="Class ID")

    if center_labels is not None:
        center_labels = np.asarray(center_labels).flatten()
        if center_labels.shape[0] != labels.shape[0]:
            raise ValueError("center_labels length must match labels length")
        center_classes = np.unique(center_labels)
        centers = np.zeros((center_classes.shape[0], 2), dtype=np.float32)
        for i, c in enumerate(center_classes):
            idx = np.where(center_labels == c)[0]
            centers[i] = x_2d[idx].mean(axis=0).astype(np.float32)
        ax.scatter(centers[:, 0], centers[:, 1], s=130, marker="X", facecolors="none",
                   edgecolors="black", linewidths=1.0, label=center_label_name, zorder=5)
        for i, c in enumerate(center_classes):
            ax.annotate(str(int(c)), (centers[i, 0], centers[i, 1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="black", zorder=6)
        handles, lgnd_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    if contour_labels is not None:
        contour_labels = np.asarray(contour_labels).flatten()
        if contour_labels.shape[0] != labels.shape[0]:
            raise ValueError("contour_labels length must match labels length")
        chi2_q_90_df2 = 4.605170186
        contour_classes = np.unique(contour_labels)
        cmap_contour = plt.cm.get_cmap("tab20", max(int(contour_classes.shape[0]), 1))
        contour_count = 0
        first_label = True
        for i, c in enumerate(contour_classes):
            idx = np.where(contour_labels == c)[0]
            if idx.size < 3:
                continue
            pts = x_2d[idx]
            mu = pts.mean(axis=0)
            cov = np.cov(pts.T)
            if not np.all(np.isfinite(cov)):
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, a_min=1e-9, a_max=None)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            width  = 2.0 * np.sqrt(chi2_q_90_df2 * eigvals[0])
            height = 2.0 * np.sqrt(chi2_q_90_df2 * eigvals[1])
            angle  = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            ell = Ellipse(xy=(float(mu[0]), float(mu[1])), width=float(width),
                          height=float(height), angle=angle, fill=False,
                          edgecolor=cmap_contour(i), linewidth=1.1, alpha=0.75,
                          label=contour_label_name if first_label else None, zorder=4)
            ax.add_patch(ell)
            first_label = False
            contour_count += 1
        if contour_count > 0:
            print(f"Plotted {contour_count} superclass 90% contours.")
            handles, lgnd_labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_umap(
    x_2d: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    tail_mask: np.ndarray | None = None,
    center_labels: np.ndarray | None = None,
    center_label_name: str = "Center IDs",
    contour_labels: np.ndarray | None = None,
    contour_label_name: str = "Superclass 90% contours",
    title: str = "UMAP of Embeddings",
    annotation: dict | None = None,
) -> None:
    """UMAP visualisation with same overlays and annotation format as plot_tsne."""
    if tail_mask is None:
        tail_mask = np.zeros(labels.shape[0], dtype=bool)
    else:
        tail_mask = np.asarray(tail_mask, dtype=bool).flatten()

    fig, ax = plt.subplots(figsize=(10, 8))

    if annotation is not None:
        _scatter_annotated(ax, x_2d, labels, annotation, tail_mask)
    else:
        cmap = "tab20" if np.unique(labels).size <= 20 else "viridis"
        normal_mask = ~tail_mask
        sc = None
        if normal_mask.any():
            sc = ax.scatter(x_2d[normal_mask, 0], x_2d[normal_mask, 1],
                            c=labels[normal_mask], s=8, alpha=0.55, marker="o", cmap=cmap)
        if tail_mask.any():
            sc_t = ax.scatter(x_2d[tail_mask, 0], x_2d[tail_mask, 1],
                              c=labels[tail_mask], s=70, alpha=0.95, marker="*",
                              edgecolors="black", linewidths=0.5, cmap=cmap, label="Tail samples")
            if sc is None:
                sc = sc_t
            ax.legend(loc="best")
        if sc is not None:
            fig.colorbar(sc, ax=ax, label="Class ID")

    if center_labels is not None:
        center_labels = np.asarray(center_labels).flatten()
        center_classes = np.unique(center_labels)
        centers = np.zeros((center_classes.shape[0], 2), dtype=np.float32)
        for i, c in enumerate(center_classes):
            idx = np.where(center_labels == c)[0]
            centers[i] = x_2d[idx].mean(axis=0).astype(np.float32)
        ax.scatter(centers[:, 0], centers[:, 1], s=130, marker="X", facecolors="none",
                   edgecolors="black", linewidths=1.0, label=center_label_name, zorder=5)
        for i, c in enumerate(center_classes):
            ax.annotate(str(int(c)), (centers[i, 0], centers[i, 1]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="black", zorder=6)
        handles, lgnd_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    if contour_labels is not None:
        contour_labels = np.asarray(contour_labels).flatten()
        chi2_q_90_df2 = 4.605170186
        contour_classes = np.unique(contour_labels)
        cmap_contour = plt.cm.get_cmap("tab20", max(int(contour_classes.shape[0]), 1))
        contour_count = 0
        first_label = True
        for i, c in enumerate(contour_classes):
            idx = np.where(contour_labels == c)[0]
            if idx.size < 3:
                continue
            pts = x_2d[idx]
            mu = pts.mean(axis=0)
            cov = np.cov(pts.T)
            if not np.all(np.isfinite(cov)):
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, a_min=1e-9, a_max=None)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            width  = 2.0 * np.sqrt(chi2_q_90_df2 * eigvals[0])
            height = 2.0 * np.sqrt(chi2_q_90_df2 * eigvals[1])
            angle  = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            ell = Ellipse(xy=(float(mu[0]), float(mu[1])), width=float(width),
                          height=float(height), angle=angle, fill=False,
                          edgecolor=cmap_contour(i), linewidth=1.1, alpha=0.75,
                          label=contour_label_name if first_label else None, zorder=4)
            ax.add_patch(ell)
            first_label = False
            contour_count += 1
        if contour_count > 0:
            handles, lgnd_labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Class-center and tail-sample analysis
# ============================================================

def compute_class_centers(emb_norm: np.ndarray, labels: np.ndarray):
    """Compute L2-normalised class centroids and each sample's cosine similarity to its own center.

    Returns (classes, centers [C, D], own_class_sim [N]).
    """
    classes = np.unique(labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    d = emb_norm.shape[1]
    centers = np.zeros((classes.shape[0], d), dtype=np.float32)
    own_class_sim = np.zeros((emb_norm.shape[0],), dtype=np.float32)

    for c in classes:
        idx = np.where(labels == c)[0]
        class_emb = emb_norm[idx]
        center = class_emb.mean(axis=0)
        center = center / max(np.linalg.norm(center), 1e-12)
        ci = class_to_idx[c]
        centers[ci] = center.astype(np.float32)
        own_class_sim[idx] = class_emb @ centers[ci]

    return classes, centers, own_class_sim


def get_tail_sample_indices(labels: np.ndarray, own_class_sim: np.ndarray, classes: np.ndarray, tail_samples_per_class: int) -> np.ndarray:
    """Return indices of the tail_samples_per_class lowest-similarity samples in each class."""
    picked = []
    for c in classes:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        scores = own_class_sim[idx]
        n_pick = min(int(tail_samples_per_class), idx.size)
        order = np.argsort(scores)[:n_pick]
        picked.extend(idx[order].tolist())
    if not picked:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(picked, dtype=np.int64))


def save_intra_class_stats(labels: np.ndarray, own_class_sim: np.ndarray, classes: np.ndarray, out_path: Path) -> None:
    """Write per-class centre-similarity statistics (mean, std, min, max) to a CSV file."""
    rows = []
    for c in classes:
        idx = np.where(labels == c)[0]
        sims = own_class_sim[idx]
        rows.append(
            {
                "class_id": int(c),
                "count": int(idx.shape[0]),
                "mean_own_center_sim": float(np.mean(sims)),
                "std_own_center_sim": float(np.std(sims)),
                "min_own_center_sim": float(np.min(sims)),
                "max_own_center_sim": float(np.max(sims)),
            }
        )

    rows.sort(key=lambda x: x["class_id"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "count",
                "mean_own_center_sim",
                "std_own_center_sim",
                "min_own_center_sim",
                "max_own_center_sim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_path}")


def _safe_open_image(path: str) -> Image.Image:
    """Open an image as RGB; return a blank 224×224 black image on any error."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color="black")


def _format_path_for_overlay(path: str, width: int = 55) -> str:
    """Wrap a file path string to width characters for use in matplotlib text overlays."""
    return "\n".join(textwrap.wrap(path, width=width)) if path else "N/A"


def generate_tail_sample_analysis(
    emb_norm: np.ndarray,
    labels: np.ndarray,
    image_paths: list[str],
    classes: np.ndarray,
    class_centers: np.ndarray,
    own_class_sim: np.ndarray,
    out_dir: Path,
    tail_samples_per_class: int = 20,
) -> None:
    """For each class, save a figure per tail sample showing the image, its nearest neighbor, and center similarities."""
    if len(image_paths) != emb_norm.shape[0]:
        raise ValueError(
            f"image_paths length mismatch: paths={len(image_paths)}, embeddings={emb_norm.shape[0]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for c in tqdm(classes, desc="Tail analysis (per class)"):
        c_int = int(c)
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue

        class_scores = own_class_sim[idx]
        order = np.argsort(class_scores)
        n_pick = min(tail_samples_per_class, idx.size)
        tail_indices = idx[order[:n_pick]]

        class_dir = out_dir / f"class_{c_int}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for rank_in_tail, q_idx in enumerate(tail_indices, start=1):
            q_emb = emb_norm[q_idx]
            sim_to_all = q_emb @ emb_norm.T
            sim_to_all[q_idx] = -np.inf
            nn_idx = int(np.argmax(sim_to_all))
            nn_sim = float(sim_to_all[nn_idx])

            center_sims = q_emb @ class_centers.T
            pred_center_class = int(classes[int(np.argmax(center_sims))])
            own_center = float(own_class_sim[q_idx])

            q_label = int(labels[q_idx])
            nn_label = int(labels[nn_idx])

            q_img = _safe_open_image(image_paths[q_idx])
            nn_img = _safe_open_image(image_paths[nn_idx])

            fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.0, 1.0, 1.6]})

            axes[0].imshow(q_img)
            axes[0].axis("off")
            axes[0].set_title("Tail Sample")
            axes[0].text(
                0.02,
                0.02,
                (
                    f"class={q_label}\n"
                    f"nearest-sim={nn_sim:.4f}\n"
                    f"own-center-sim={own_center:.4f}\n"
                    f"path:\n{_format_path_for_overlay(image_paths[q_idx])}"
                ),
                transform=axes[0].transAxes,
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, pad=6),
            )

            axes[1].imshow(nn_img)
            axes[1].axis("off")
            axes[1].set_title("Nearest Neighbor")
            axes[1].text(
                0.02,
                0.02,
                (
                    f"class={nn_label}\n"
                    f"sim-to-tail={nn_sim:.4f}\n"
                    f"path:\n{_format_path_for_overlay(image_paths[nn_idx])}"
                ),
                transform=axes[1].transAxes,
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, pad=6),
            )

            order_desc = np.argsort(center_sims)[::-1]
            sorted_scores = center_sims[order_desc]
            sorted_classes = classes[order_desc]
            y_pos = np.arange(sorted_scores.shape[0])
            bar_colors = np.array(["#5f6c7b"] * sorted_scores.shape[0], dtype=object)
            bar_colors[np.where(sorted_classes == q_label)[0]] = "#2ca02c"
            bar_colors[np.where(sorted_classes == pred_center_class)[0]] = "#d62728"
            axes[2].barh(y_pos, sorted_scores, color=bar_colors)
            axes[2].invert_yaxis()
            axes[2].set_xlabel("Cosine similarity")
            axes[2].set_title("Similarity to Each Class Center")
            if sorted_scores.shape[0] <= 60:
                axes[2].set_yticks(y_pos)
                axes[2].set_yticklabels([str(int(x)) for x in sorted_classes], fontsize=8)
                axes[2].set_ylabel("Class ID")
            else:
                axes[2].set_yticks([])
            axes[2].axvline(x=0.0, color="black", linewidth=0.8, alpha=0.5)

            fig.suptitle(
                f"class={q_label} tail-rank={rank_in_tail} idx={q_idx} "
                f"(pred-center-class={pred_center_class})",
                fontsize=12,
            )
            fig.tight_layout()

            out_path = class_dir / f"tail_{rank_in_tail:02d}_idx_{q_idx}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            summary_rows.append(
                {
                    "class_id": q_label,
                    "tail_rank_within_class": rank_in_tail,
                    "sample_index": int(q_idx),
                    "sample_image_path": image_paths[q_idx],
                    "own_center_similarity": own_center,
                    "nearest_index": nn_idx,
                    "nearest_image_path": image_paths[nn_idx],
                    "nearest_class_id": nn_label,
                    "nearest_similarity": nn_sim,
                    "pred_center_class": pred_center_class,
                    "figure_path": str(out_path),
                }
            )

    summary_path = out_dir / "tail_samples_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "tail_rank_within_class",
                "sample_index",
                "sample_image_path",
                "own_center_similarity",
                "nearest_index",
                "nearest_image_path",
                "nearest_class_id",
                "nearest_similarity",
                "pred_center_class",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_path}")


# ============================================================
# A) Neighbourhood purity
# ============================================================

def compute_neighborhood_purity(
    embeddings: np.ndarray,
    category_labels: np.ndarray,
    superclass_labels: np.ndarray,
    k_list: list[int],
    device: torch.device,
    batch_size: int = 512,
) -> dict[int, dict[str, float]]:
    """For each k, compute mean fraction of top-k neighbours that are:
      same_cat_frac:           same sub-class (fine label)
      same_sup_diff_cat_frac:  same superclass but different sub-class
      diff_sup_frac:           different superclass entirely
    """
    emb = torch.from_numpy(embeddings).float().to(device)
    emb = torch.nn.functional.normalize(emb, dim=1)
    cat_t = torch.from_numpy(category_labels).to(device)
    sup_t = torch.from_numpy(superclass_labels).to(device)
    n = emb.shape[0]

    requested_k = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not requested_k:
        return {}
    max_allowed_k = n - 1
    max_k = max(min(k, max_allowed_k) for k in requested_k)

    same_cat_sum = {k: 0.0 for k in requested_k}
    same_sup_sum = {k: 0.0 for k in requested_k}
    diff_sup_sum = {k: 0.0 for k in requested_k}

    for i in tqdm(range(0, n, batch_size), desc="Neighbourhood purity"):
        end = min(i + batch_size, n)
        q = emb[i:end]
        q_cat = cat_t[i:end]
        q_sup = sup_t[i:end]
        sim = q @ emb.T
        for j in range(sim.shape[0]):
            sim[j, i + j] = float("-inf")
        topk_idx = torch.topk(sim, max_k, dim=1).indices  # [B, max_k]

        for k_req in requested_k:
            kk = min(k_req, max_allowed_k)
            nb_cat = cat_t[topk_idx[:, :kk]]   # [B, kk]
            nb_sup = sup_t[topk_idx[:, :kk]]   # [B, kk]
            same_cat = nb_cat == q_cat.unsqueeze(1)
            same_sup = nb_sup == q_sup.unsqueeze(1)
            same_sup_diff = same_sup & ~same_cat
            diff_sup_mask = ~same_sup
            same_cat_sum[k_req] += float(same_cat.float().mean(dim=1).sum().item())
            same_sup_sum[k_req] += float(same_sup_diff.float().mean(dim=1).sum().item())
            diff_sup_sum[k_req] += float(diff_sup_mask.float().mean(dim=1).sum().item())

    return {
        k: {
            "same_cat_frac": same_cat_sum[k] / n,
            "same_sup_diff_cat_frac": same_sup_sum[k] / n,
            "diff_sup_frac": diff_sup_sum[k] / n,
        }
        for k in requested_k
    }


def plot_neighborhood_purity(purity: dict[int, dict[str, float]], out_path: Path) -> None:
    k_vals = sorted(purity.keys())
    if not k_vals:
        return
    same_sup = np.array([purity[k]["same_sup_diff_cat_frac"] for k in k_vals])
    diff_sup = np.array([purity[k]["diff_sup_frac"] for k in k_vals])

    x = np.arange(len(k_vals))
    fig, ax = plt.subplots(figsize=(max(6, len(k_vals) * 1.4), 5))
    ax.bar(x, same_sup, label="Same superclass, diff sub-class", color="#ff7f0e")
    ax.bar(x, diff_sup, bottom=same_sup, label="Diff superclass", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("Fraction of top-k neighbours (excl. same sub-class)")
    ax.set_ylim(0, 0.2)
    ax.set_title("Neighbourhood Purity Breakdown")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# B) Cluster shape evaluation
# ============================================================

def compute_cluster_shape_stats(
    emb_norm: np.ndarray,
    category_labels: np.ndarray,
    superclass_labels: np.ndarray,
    classes: np.ndarray,
    class_centers: np.ndarray,
) -> dict:
    """Covariance-trace analysis: intra-sub-class, intra-superclass, between-sub-class, total.
    Also computes per-sample cosine distance to own sub-class class center.
    """
    # Per-category covariance traces + per-sample distances to center
    class_to_center = {int(c): class_centers[i] for i, c in enumerate(classes)}
    cat_traces: list[float] = []
    per_cat_dist = np.zeros(emb_norm.shape[0], dtype=np.float32)

    for c in np.unique(category_labels):
        idx = np.where(category_labels == c)[0]
        pts = emb_norm[idx]
        if pts.shape[0] >= 2:
            cat_traces.append(float(np.trace(np.cov(pts.T))))
        center = class_to_center.get(int(c))
        if center is not None:
            per_cat_dist[idx] = (1.0 - (pts @ center)).astype(np.float32)

    # Per-superclass covariance traces
    sup_traces: list[float] = []
    for s in np.unique(superclass_labels):
        idx = np.where(superclass_labels == s)[0]
        if idx.size >= 2:
            sup_traces.append(float(np.trace(np.cov(emb_norm[idx].T))))

    # Between-category scatter: covariance of class-center matrix
    between_trace = float(np.trace(np.cov(class_centers.T))) if class_centers.shape[0] > 1 else 0.0

    # Total
    total_trace = float(np.trace(np.cov(emb_norm.T)))

    return {
        "intra_cat_trace_mean": float(np.mean(cat_traces)) if cat_traces else 0.0,
        "intra_cat_trace_std":  float(np.std(cat_traces)) if cat_traces else 0.0,
        "intra_sup_trace_mean": float(np.mean(sup_traces)) if sup_traces else 0.0,
        "intra_sup_trace_std":  float(np.std(sup_traces)) if sup_traces else 0.0,
        "between_cat_trace":    between_trace,
        "total_trace":          total_trace,
        "intra_cat_to_total":   float(np.mean(cat_traces) / max(total_trace, 1e-9)) if cat_traces else 0.0,
        "intra_sup_to_total":   float(np.mean(sup_traces) / max(total_trace, 1e-9)) if sup_traces else 0.0,
        "per_cat_dist_to_center": per_cat_dist,  # ndarray, excluded from JSON
    }


def plot_cluster_shape(shape_stats: dict, out_path: Path, bins: int = 60) -> None:
    """1×2 figure: covariance-trace bar chart (left) + per-sample center-distance histogram (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names  = ["Intra-sub-class\n(mean)", "Intra-sup\n(mean)", "Between-sub-class", "Total"]
    values = [
        shape_stats["intra_cat_trace_mean"],
        shape_stats["intra_sup_trace_mean"],
        shape_stats["between_cat_trace"],
        shape_stats["total_trace"],
    ]
    errors = [shape_stats["intra_cat_trace_std"], shape_stats["intra_sup_trace_std"], 0.0, 0.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    bars = axes[0].bar(names, values, yerr=errors, capsize=5, color=colors)
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    axes[0].set_ylabel("Covariance trace")
    axes[0].set_title("Cluster Scatter (Covariance Trace)")

    dists = shape_stats["per_cat_dist_to_center"]
    mean_d = float(np.mean(dists))
    axes[1].hist(dists, bins=bins, density=True, color="#1f77b4", edgecolor="white", linewidth=0.3)
    axes[1].axvline(mean_d, color="red", linestyle="--", linewidth=1.2, label=f"mean={mean_d:.3f}")
    axes[1].set_xlabel("Cosine distance to class centre (1 − cos sim)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Per-Sample Cosine Distance to Sub-class Centre")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# C) Border mass density
# ============================================================

def compute_border_mass_density(
    emb_norm: np.ndarray,
    category_labels: np.ndarray,
    superclass_labels: np.ndarray,
    n_probes: int = 2000,
    neg_sample_size: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Approximate intra- and inter-superclass margin for a random subset of probes.

    For each probe x with category C, superclass S:
      intra_margin = min(sim(x, same_sup_diff_cat)) - min(sim(x, same_cat))
      inter_margin = min(sim(x, diff_sup))          - min(sim(x, same_cat))

    Positive margin => impostors are less similar than the hardest positive (clean boundary).
    Negative margin => some impostors are closer than some positives (boundary violation).
    """
    rng = np.random.default_rng(random_state)
    n = emb_norm.shape[0]

    # Precompute per-category and per-superclass index pools
    cat_to_idx: dict[int, np.ndarray] = {
        int(c): np.where(category_labels == c)[0] for c in np.unique(category_labels)
    }
    sup_to_idx: dict[int, np.ndarray] = {
        int(s): np.where(superclass_labels == s)[0] for s in np.unique(superclass_labels)
    }

    probe_indices = rng.choice(n, size=min(n_probes, n), replace=False)

    intra_margins: list[float] = []
    inter_margins: list[float] = []
    per_sup_intra: dict[int, list[float]] = {int(s): [] for s in np.unique(superclass_labels)}
    per_sup_inter: dict[int, list[float]] = {int(s): [] for s in np.unique(superclass_labels)}
    n_skipped = 0

    for pi in tqdm(probe_indices, desc="Border density"):
        p_cat = int(category_labels[pi])
        p_sup = int(superclass_labels[pi])
        probe = emb_norm[pi]

        # Same-category set (exclude self)
        same_cat_pool = cat_to_idx[p_cat]
        same_cat_pool = same_cat_pool[same_cat_pool != pi]
        if same_cat_pool.size == 0:
            n_skipped += 1
            continue

        # Same-superclass, different-category set
        same_sup_pool = sup_to_idx[p_sup]
        same_sup_diff_cat = same_sup_pool[category_labels[same_sup_pool] != p_cat]

        if same_sup_diff_cat.size == 0:
            n_skipped += 1
            continue

        # Subsample negatives
        sc_sample = same_cat_pool if same_cat_pool.size <= neg_sample_size else rng.choice(same_cat_pool, size=neg_sample_size, replace=False)
        ss_sample = same_sup_diff_cat if same_sup_diff_cat.size <= neg_sample_size else rng.choice(same_sup_diff_cat, size=neg_sample_size, replace=False)

        min_sim_same_cat = float(np.dot(emb_norm[sc_sample], probe).min())
        min_sim_same_sup = float(np.dot(emb_norm[ss_sample], probe).min())
        intra_val = min_sim_same_sup - min_sim_same_cat
        intra_margins.append(intra_val)
        per_sup_intra[p_sup].append(intra_val)

        # Inter-superclass: oversample globally, filter to different superclass
        candidates = rng.choice(n, size=min(neg_sample_size * 6, n), replace=False)
        diff_sup_pool = candidates[superclass_labels[candidates] != p_sup]
        if diff_sup_pool.size == 0:
            continue
        ds_sample = diff_sup_pool[:neg_sample_size]
        min_sim_diff_sup = float(np.dot(emb_norm[ds_sample], probe).min())
        inter_val = min_sim_diff_sup - min_sim_same_cat
        inter_margins.append(inter_val)
        per_sup_inter[p_sup].append(inter_val)

    intra_arr = np.array(intra_margins, dtype=np.float32)
    inter_arr  = np.array(inter_margins,  dtype=np.float32)
    per_sup_data = {
        s: (np.array(per_sup_intra[s], dtype=np.float32),
            np.array(per_sup_inter[s], dtype=np.float32))
        for s in per_sup_intra
    }

    def _margin_summary(arr: np.ndarray) -> dict:
        if arr.size == 0:
            return {"mean": None, "std": None, "frac_positive": None}
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "frac_positive": float(np.mean(arr > 0)),
        }

    stats = {
        "n_probes_requested": int(len(probe_indices)),
        "n_probes_skipped": int(n_skipped),
        "n_intra_computed": int(len(intra_margins)),
        "n_inter_computed":  int(len(inter_margins)),
        "intra_margin": _margin_summary(intra_arr),
        "inter_margin":  _margin_summary(inter_arr),
    }
    return intra_arr, inter_arr, stats, per_sup_data


def plot_border_density(
    intra_margins: np.ndarray,
    inter_margins: np.ndarray,
    per_sup_data: dict[int, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    bins: int = 50,
) -> None:
    """Global subplot + one subplot per superclass, each showing intra (blue) / inter (red) margins."""
    sup_ids = sorted(per_sup_data.keys())
    n_total = len(sup_ids) + 1          # +1 for global
    n_cols = min(4, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), squeeze=False)
    axes_flat = axes.flatten()

    def _plot_one(ax: plt.Axes, intra: np.ndarray, inter: np.ndarray, title: str) -> None:
        if intra.size > 0:
            ax.hist(intra, bins=bins, alpha=0.65, density=True, color="blue",
                    label="Intra (same-sup, diff sub-class)")
        if inter.size > 0:
            ax.hist(inter, bins=bins, alpha=0.55, density=True, color="red",
                    label="Inter (diff-sup)")
        ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Margin", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if intra.size > 0 or inter.size > 0:
            ax.legend(fontsize=7)

    _plot_one(axes_flat[0], intra_margins, inter_margins, "Global")
    for i, sup_id in enumerate(sup_ids):
        intra_s, inter_s = per_sup_data[sup_id]
        _plot_one(axes_flat[i + 1], intra_s, inter_s, f"Superclass {sup_id}")

    for ax in axes_flat[n_total:]:
        ax.set_visible(False)

    fig.suptitle("Border Mass Density — Global + Per Superclass", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# D) Embedding anisotropy
# ============================================================

def compute_embedding_anisotropy(embeddings: np.ndarray) -> dict:
    """SVD spectrum via eigendecomposition of the empirical covariance matrix.

    Uses D×D covariance (efficient when N >> D, the common case).
    Returns eigenvalues, participation ratio, 90%/99% variance cutoff ranks,
    and per-dimension mean absolute value (mask activity proxy).
    """
    N, D = embeddings.shape
    emb_f64 = embeddings.astype(np.float64)
    mu = emb_f64.mean(axis=0, keepdims=True)
    centered = emb_f64 - mu

    if N >= D:
        # D×D covariance — standard path when N >> D
        cov = (centered.T @ centered) / max(N - 1, 1)
        raw_eigvals = np.linalg.eigvalsh(cov)  # ascending
    else:
        # N×N Gram matrix — fallback when D > N (rare for this use case)
        gram = (centered @ centered.T) / max(N - 1, 1)
        raw_eigvals_gram = np.linalg.eigvalsh(gram)
        # Gram eigenvalues == cov eigenvalues (up to N zeros); keep only N
        raw_eigvals = raw_eigvals_gram

    eigvals = np.clip(raw_eigvals[::-1].copy(), a_min=0.0, a_max=None)  # descending

    total_var = float(eigvals.sum())
    if total_var > 1e-12:
        cumvar = np.cumsum(eigvals) / total_var
        rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    else:
        cumvar = np.ones_like(eigvals)
        rank_90, rank_99 = D, D

    # Participation ratio: (Σλ)² / Σλ² — effective rank
    sum_sq = float(np.sum(eigvals ** 2))
    pr = float(total_var ** 2 / sum_sq) if sum_sq > 1e-12 else 0.0

    # Per-dimension activity: mean |x_i| over all samples (mask proxy for masked embeddings)
    dim_mean_abs = np.mean(np.abs(embeddings), axis=0).astype(np.float32)
    dim_std = embeddings.std(axis=0).astype(np.float32)

    return {
        "eigenvalues":        eigvals.astype(np.float32),
        "cumvar":             cumvar.astype(np.float32),
        "participation_ratio": pr,
        "rank_90pct_var":     int(rank_90),
        "rank_99pct_var":     int(rank_99),
        "total_variance":     float(total_var),
        "total_dim":          int(D),
        "dim_mean_abs":       dim_mean_abs,
        "dim_std":            dim_std,
    }


def plot_anisotropy_spectrum(aniso: dict, out_path: Path) -> None:
    """1×3 figure: eigenvalue spectrum (log-y) | cumulative variance | per-dim activity bar."""
    eigvals  = aniso["eigenvalues"]
    cumvar   = aniso["cumvar"]
    dim_abs  = aniso["dim_mean_abs"]
    rank_90  = aniso["rank_90pct_var"]
    rank_99  = aniso["rank_99pct_var"]
    pr       = aniso["participation_ratio"]
    D        = aniso["total_dim"]
    x_ranks  = np.arange(1, len(eigvals) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: eigenvalue spectrum
    axes[0].semilogy(x_ranks, eigvals + 1e-12, linewidth=1.2, color="#1f77b4")
    axes[0].axvline(rank_90, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"90% var @ {rank_90}")
    axes[0].axvline(rank_99, color="#d62728", linestyle="--", linewidth=1.0, label=f"99% var @ {rank_99}")
    axes[0].set_xlabel("Eigenvalue rank")
    axes[0].set_ylabel("Eigenvalue (log scale)")
    axes[0].set_title(f"SVD Spectrum  (PR = {pr:.1f} / {D})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.35)

    # Centre: cumulative variance
    axes[1].plot(x_ranks, cumvar, linewidth=1.5, color="#2ca02c")
    axes[1].axhline(0.90, color="#ff7f0e", linestyle="--", linewidth=0.8, label=f"90% @ {rank_90}")
    axes[1].axhline(0.99, color="#d62728", linestyle="--", linewidth=0.8, label=f"99% @ {rank_99}")
    axes[1].axvline(rank_90, color="#ff7f0e", linestyle="--", linewidth=0.8)
    axes[1].axvline(rank_99, color="#d62728", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.35)

    # Right: per-dimension activity (sorted descending — mask proxy)
    dim_abs_sorted = np.sort(dim_abs)[::-1]
    axes[2].bar(np.arange(1, len(dim_abs_sorted) + 1), dim_abs_sorted, color="#9467bd", width=1.0)
    axes[2].set_xlabel("Dimension rank (by activity)")
    axes[2].set_ylabel("Mean |x_i|")
    axes[2].set_title("Per-Dimension Activity (mask value proxy)")
    axes[2].grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Core evaluation orchestrator
# ============================================================

def evaluate_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    tag: str,
    image_paths: list[str] | None,
    center_overlay_labels: np.ndarray | None = None,
    center_overlay_name: str = "Centers",
    contour_overlay_labels: np.ndarray | None = None,
    contour_overlay_name: str = "Superclass 90% contours",
    category_labels: np.ndarray | None = None,
    superclass_labels: np.ndarray | None = None,
    embedding_mode: str = "unknown",
    annotation: dict | None = None,
) -> None:
    """Run the full evaluation suite (retrieval, purity, cluster shape, border density, anisotropy, t-SNE/UMAP, tail analysis) for one embedding space."""
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"Shape mismatch for {tag}: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}")
    if center_overlay_labels is not None and center_overlay_labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"center_overlay_labels length mismatch for {tag}")
    if contour_overlay_labels is not None and contour_overlay_labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"contour_overlay_labels length mismatch for {tag}")
    if category_labels is not None and category_labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"category_labels length mismatch for {tag}")
    if superclass_labels is not None and superclass_labels.shape[0] != embeddings.shape[0]:
        raise ValueError(f"superclass_labels length mismatch for {tag}")

    dual_labels_available = (category_labels is not None and superclass_labels is not None)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{tag}] Computing recall@k and precision@k...")
    recall, precision, clipped_k, per_class_recall, per_class_precision = compute_retrieval_metrics_at_k(
        embeddings=embeddings,
        labels=labels,
        k_list=args.rank_k,
        device=device,
        batch_size=args.batch_size,
    )

    if clipped_k:
        print(f"[{tag}] Warning: requested k {clipped_k} exceed N-1={embeddings.shape[0]-1}; clipped.")

    recall_path    = output_dir / "recall_at_k.txt"
    precision_path = output_dir / "precision_at_k.txt"
    metrics_path   = output_dir / "metrics.json"

    with open(recall_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Recall@{k}: {recall[k]:.4f}\n")

    with open(precision_path, "w") as f:
        for k in args.rank_k:
            f.write(f"Precision@{k}: {precision[k]:.4f}\n")

    # payload is written to disk at end-of-function after all modules append to it
    payload: dict = {
        "tag": tag,
        "embedding_mode": embedding_mode,
        "num_samples": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "rank_k": args.rank_k,
        "recall":              {str(k): float(v) for k, v in recall.items()},
        "precision":           {str(k): float(v) for k, v in precision.items()},
        "clipped_k":           clipped_k,
        "per_class_recall":    per_class_recall,
        "per_class_precision": per_class_precision,
        "bits_left":           compute_bits_left_stats(embeddings),
    }

    print(f"[{tag}] Saved: {recall_path}")
    print(f"[{tag}] Saved: {precision_path}")

    # Pre-compute emb_norm / class centers when needed by multiple analyses
    precompute_tail = (
        (not args.skip_tsne)
        or (not args.skip_tail_analysis)
        or (not args.skip_cluster_shape and dual_labels_available)
        or (not args.skip_border_density and dual_labels_available)
    )
    emb_norm = None
    classes = None
    class_centers = None
    own_class_sim = None
    tail_indices_global = np.array([], dtype=np.int64)

    if precompute_tail:
        emb_norm = _normalize_embeddings(embeddings)
        classes, class_centers, own_class_sim = compute_class_centers(emb_norm, labels)
        tail_indices_global = get_tail_sample_indices(
            labels=labels,
            own_class_sim=own_class_sim,
            classes=classes,
            tail_samples_per_class=args.tail_samples_per_class,
        )

    if not args.skip_distribution:
        print(f"\n[{tag}] Computing match/non-match distribution...")
        match_sims, nonmatch_sims = compute_match_nonmatch_distribution(
            embeddings=embeddings,
            labels=labels,
            device=device,
            block_size=args.block_size,
        )
        plot_match_nonmatch_distribution(match_sims, nonmatch_sims, output_dir / "match_vs_nonmatch_distribution.png")
    else:
        print(f"\n[{tag}] Skipping match/non-match distribution (--skip-distribution).")

    if not args.skip_tsne:
        subsample = None if args.tsne_subsample <= 0 else args.tsne_subsample
        n_tsne = min(embeddings.shape[0], subsample) if subsample else embeddings.shape[0]
        print(f"\n[{tag}] Running t-SNE (n={n_tsne})...")
        x_2d, y_sub, idx_used = run_tsne(
            embeddings,
            labels,
            perplexity=args.tsne_perplexity,
            subsample=subsample,
        )
        tail_mask_sub = np.isin(idx_used, tail_indices_global)
        print(f"[{tag}] t-SNE tail markers: {int(tail_mask_sub.sum())}/{len(idx_used)} points")

        center_labels_sub = None
        if center_overlay_labels is not None:
            center_labels_sub = center_overlay_labels[idx_used]
            print(f"[{tag}] Overlaying {np.unique(center_labels_sub).size} {center_overlay_name.lower()} on t-SNE.")

        contour_labels_sub = None
        if contour_overlay_labels is not None:
            contour_labels_sub = contour_overlay_labels[idx_used]
            print(f"[{tag}] Overlaying {np.unique(contour_labels_sub).size} {contour_overlay_name.lower()} on t-SNE.")

        plot_tsne(
            x_2d,
            y_sub,
            output_dir / "tsne_embeddings.png",
            tail_mask=tail_mask_sub,
            center_labels=center_labels_sub,
            center_label_name=center_overlay_name,
            contour_labels=contour_labels_sub,
            contour_label_name=contour_overlay_name,
            title=f"t-SNE ({tag}) tail samples as stars",
            annotation=annotation,
        )
    else:
        print(f"\n[{tag}] Skipping t-SNE (--skip-tsne).")

    # ---- UMAP ----
    if not args.skip_umap:
        subsample = None if args.tsne_subsample <= 0 else args.tsne_subsample
        n_umap = min(embeddings.shape[0], subsample) if subsample else embeddings.shape[0]
        print(f"\n[{tag}] Running UMAP (n={n_umap})...")
        x_2d_u, y_sub_u, idx_u = run_umap(
            embeddings, labels,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            subsample=subsample,
        )
        tail_mask_u = np.isin(idx_u, tail_indices_global)
        print(f"[{tag}] UMAP tail markers: {int(tail_mask_u.sum())}/{len(idx_u)} points")

        center_labels_u = center_overlay_labels[idx_u] if center_overlay_labels is not None else None
        contour_labels_u = contour_overlay_labels[idx_u] if contour_overlay_labels is not None else None

        plot_umap(
            x_2d_u, y_sub_u,
            output_dir / "umap_embeddings.png",
            tail_mask=tail_mask_u,
            center_labels=center_labels_u,
            center_label_name=center_overlay_name,
            contour_labels=contour_labels_u,
            contour_label_name=contour_overlay_name,
            title=f"UMAP ({tag}) — tail samples as stars",
            annotation=annotation,
        )
    else:
        print(f"\n[{tag}] Skipping UMAP (--skip-umap).")

    if not args.skip_tail_analysis:
        if args.tail_samples_per_class <= 0:
            raise ValueError("--tail-samples-per-class must be > 0")

        if image_paths is None:
            print(f"\n[{tag}] Skipping tail analysis: no image paths available.")
        else:
            print(f"\n[{tag}] Computing per-class tail sample analysis...")
            if emb_norm is None or classes is None or class_centers is None or own_class_sim is None:
                emb_norm = _normalize_embeddings(embeddings)
                classes, class_centers, own_class_sim = compute_class_centers(emb_norm, labels)

            intra_path = output_dir / "intra_class_similarity_stats.csv"
            save_intra_class_stats(labels, own_class_sim, classes, intra_path)

            generate_tail_sample_analysis(
                emb_norm=emb_norm,
                labels=labels,
                image_paths=image_paths,
                classes=classes,
                class_centers=class_centers,
                own_class_sim=own_class_sim,
                out_dir=output_dir / "tail_analysis",
                tail_samples_per_class=args.tail_samples_per_class,
            )
    else:
        print(f"\n[{tag}] Skipping tail analysis (--skip-tail-analysis).")

    # ---- D) Embedding anisotropy (only requires embeddings — always available) ----
    if not args.skip_anisotropy:
        print(f"\n[{tag}] Computing embedding anisotropy...")
        aniso = compute_embedding_anisotropy(embeddings)
        plot_anisotropy_spectrum(aniso, output_dir / "anisotropy_spectrum.png")
        payload["anisotropy"] = {
            "participation_ratio": aniso["participation_ratio"],
            "rank_90pct_var":      aniso["rank_90pct_var"],
            "rank_99pct_var":      aniso["rank_99pct_var"],
            "total_variance":      aniso["total_variance"],
            "total_dim":           aniso["total_dim"],
        }
        print(
            f"[{tag}] Anisotropy — PR={aniso['participation_ratio']:.1f}/{aniso['total_dim']}  "
            f"90% var @ rank {aniso['rank_90pct_var']}  "
            f"99% var @ rank {aniso['rank_99pct_var']}"
        )
    else:
        print(f"\n[{tag}] Skipping anisotropy (--skip-anisotropy).")

    # ---- A) Neighbourhood purity (dual labels required) ----
    if not args.skip_neighborhood_purity:
        if not dual_labels_available:
            print(f"\n[{tag}] Skipping neighbourhood purity: requires both category and superclass labels.")
        else:
            print(f"\n[{tag}] Computing neighbourhood purity...")
            purity = compute_neighborhood_purity(
                embeddings=embeddings,
                category_labels=category_labels,
                superclass_labels=superclass_labels,
                k_list=args.rank_k,
                device=device,
                batch_size=args.batch_size,
            )
            plot_neighborhood_purity(purity, output_dir / "neighborhood_purity.png")
            payload["neighborhood_purity"] = {str(k): v for k, v in purity.items()}
            for k in sorted(purity):
                d = purity[k]
                print(
                    f"[{tag}] k={k}: same_cat={d['same_cat_frac']:.3f}  "
                    f"same_sup_diff_cat={d['same_sup_diff_cat_frac']:.3f}  "
                    f"diff_sup={d['diff_sup_frac']:.3f}"
                )
    else:
        print(f"\n[{tag}] Skipping neighbourhood purity (--skip-neighborhood-purity).")

    # ---- B) Cluster shape (dual labels + emb_norm required) ----
    if not args.skip_cluster_shape:
        if not dual_labels_available:
            print(f"\n[{tag}] Skipping cluster shape: requires both category and superclass labels.")
        else:
            print(f"\n[{tag}] Computing cluster shape statistics...")
            if emb_norm is None or classes is None or class_centers is None:
                emb_norm = _normalize_embeddings(embeddings)
                classes, class_centers, own_class_sim = compute_class_centers(emb_norm, labels)
            shape_stats = compute_cluster_shape_stats(
                emb_norm=emb_norm,
                category_labels=category_labels,
                superclass_labels=superclass_labels,
                classes=classes,
                class_centers=class_centers,
            )
            plot_cluster_shape(shape_stats, output_dir / "cluster_shape.png")
            payload["cluster_shape"] = {k: v for k, v in shape_stats.items() if not isinstance(v, np.ndarray)}
            print(
                f"[{tag}] Cluster shape — intra-sub-class trace={shape_stats['intra_cat_trace_mean']:.4f}  "
                f"intra-sup trace={shape_stats['intra_sup_trace_mean']:.4f}  "
                f"between-sub-class trace={shape_stats['between_cat_trace']:.4f}  "
                f"total={shape_stats['total_trace']:.4f}"
            )
    else:
        print(f"\n[{tag}] Skipping cluster shape (--skip-cluster-shape).")

    # ---- C) Border mass density (dual labels + emb_norm required) ----
    if not args.skip_border_density:
        if not dual_labels_available:
            print(f"\n[{tag}] Skipping border density: requires both category and superclass labels.")
        else:
            print(f"\n[{tag}] Computing border mass density (n_probes={args.border_density_probes})...")
            if emb_norm is None:
                emb_norm = _normalize_embeddings(embeddings)
            intra_m, inter_m, margin_stats, per_sup_data = compute_border_mass_density(
                emb_norm=emb_norm,
                category_labels=category_labels,
                superclass_labels=superclass_labels,
                n_probes=args.border_density_probes,
                neg_sample_size=args.border_density_neg_sample,
            )
            plot_border_density(intra_m, inter_m, per_sup_data, output_dir / "border_mass_density.png")
            payload["border_mass_density"] = margin_stats
            if intra_m.size > 0:
                print(
                    f"[{tag}] Intra margin: mean={margin_stats['intra_margin']['mean']:.4f}  "
                    f"frac_positive={margin_stats['intra_margin']['frac_positive']:.2%}"
                )
            if inter_m.size > 0:
                print(
                    f"[{tag}] Inter margin: mean={margin_stats['inter_margin']['mean']:.4f}  "
                    f"frac_positive={margin_stats['inter_margin']['frac_positive']:.2%}"
                )
    else:
        print(f"\n[{tag}] Skipping border density (--skip-border-density).")

    # Write consolidated metrics.json (deferred to end so all modules can contribute)
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{tag}] Saved: {metrics_path}")


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="CSN inference with retrieval metrics and advanced analysis")

    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument(
        "--prefix-metadata",
        type=Path,
        default=None,
        help="Metadata json from generate_csn_embeddings.py; evaluates 4 spaces by default.",
    )

    parser.add_argument("--image-paths", type=Path, default=None, help="Optional .npy/.npz/.txt aligned image paths")

    parser.add_argument("--rank-k", type=int, nargs="+", default=[1, 5, 10, 100, 1000])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=2000)

    parser.add_argument("--tsne-subsample", type=int, default=5000, help="0 => all points")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)

    parser.add_argument("--annotation-csv", type=Path, default=None,
                        help="labels_annotation.csv for superclass-color + category-shape encoding.")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP visualisation.")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15).")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1).")

    # --- existing skip flags ---
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--skip-tail-analysis", action="store_true")
    parser.add_argument("--tail-samples-per-class", type=int, default=20)

    # --- new skip flags ---
    parser.add_argument("--skip-neighborhood-purity", action="store_true",
                        help="Skip neighbourhood purity breakdown (A).")
    parser.add_argument("--skip-cluster-shape", action="store_true",
                        help="Skip covariance trace / cluster shape analysis (B).")
    parser.add_argument("--skip-border-density", action="store_true",
                        help="Skip border mass density margin analysis (C).")
    parser.add_argument("--skip-anisotropy", action="store_true",
                        help="Skip SVD anisotropy / per-dim activity analysis (D).")

    # --- border density parameters ---
    parser.add_argument("--border-density-probes", type=int, default=2000,
                        help="Number of probe samples for border density analysis (C).")
    parser.add_argument("--border-density-neg-sample", type=int, default=500,
                        help="Max negative samples per probe for border density (C).")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=Path("./csn_inference_output"))
    return parser.parse_args()


def main() -> None:
    """Load embeddings from disk and run the full CSN evaluation suite."""
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotation = None
    if args.annotation_csv is not None:
        annotation = load_annotation(args.annotation_csv.resolve())
        print(f"Loaded annotation from: {args.annotation_csv} "
              f"({len(annotation['sup_id_to_name'])} superclasses, "
              f"{len(annotation['cat_id_to_name'])} categories)")

    if args.prefix_metadata is None and (args.embeddings is None or args.labels is None):
        raise ValueError("Provide either --prefix-metadata OR both --embeddings and --labels")

    if args.prefix_metadata is not None:
        with open(args.prefix_metadata.resolve(), "r") as f:
            meta = json.load(f)

        outputs = meta.get("outputs", {})
        paths = {
            "image_super":    Path(outputs["image_super"]),
            "image_subclass": Path(outputs["image_subclass"]),
            "text_super":     Path(outputs["text_super"])    if "text_super"    in outputs else None,
            "text_subclass":  Path(outputs["text_subclass"]) if "text_subclass" in outputs else None,
            "superclass_ids": Path(outputs["superclass_ids"]),
            "subclass_ids":   Path(outputs["subclass_ids"]),
            "paths":          Path(outputs["paths"]) if "paths" in outputs else None,
        }

        image_paths = None
        if args.image_paths is not None:
            image_paths = (
                [str(x) for x in load_array(args.image_paths).flatten().tolist()]
                if args.image_paths.suffix.lower() in {".npy", ".npz"}
                else [ln.strip() for ln in args.image_paths.read_text().splitlines() if ln.strip()]
            )
        elif paths["paths"] is not None and paths["paths"].exists():
            image_paths = [str(x) for x in load_array(paths["paths"]).flatten().tolist()]

        superclass_ids = load_array(paths["superclass_ids"]).astype(np.int64).flatten()
        subclass_ids   = load_array(paths["subclass_ids"]).astype(np.int64).flatten()

        embedding_spaces = [
            ("image_super",    paths["image_super"]),
            ("image_subclass", paths["image_subclass"]),
        ]
        for tag_name in ("text_super", "text_subclass"):
            if paths[tag_name] is not None and paths[tag_name].exists():
                embedding_spaces.append((tag_name, paths[tag_name]))

        label_sets = [
            ("superclass_ids", superclass_ids),
            ("subclass_ids",   subclass_ids),
        ]

        for emb_tag, emb_path in embedding_spaces:
            embeddings = load_array(emb_path).astype(np.float32)
            embedding_mode = "masked" if emb_tag.endswith("subclass") else "unmasked"
            for label_tag, labels in label_sets:
                tag = f"{emb_tag}__eval_{label_tag}"
                center_overlay_labels = subclass_ids
                center_overlay_name   = "Sub-class Centers"
                contour_overlay_labels = None
                contour_overlay_name   = "Superclass 90% contours"

                if label_tag == "subclass_ids":
                    contour_overlay_labels = superclass_ids

                evaluate_space(
                    embeddings=embeddings,
                    labels=labels,
                    args=args,
                    device=device,
                    output_dir=args.output_dir / tag,
                    tag=tag,
                    image_paths=image_paths,
                    center_overlay_labels=center_overlay_labels,
                    center_overlay_name=center_overlay_name,
                    contour_overlay_labels=contour_overlay_labels,
                    contour_overlay_name=contour_overlay_name,
                    category_labels=subclass_ids,
                    superclass_labels=superclass_ids,
                    embedding_mode=embedding_mode,
                    annotation=annotation,
                )
    else:
        embeddings = load_array(args.embeddings.resolve()).astype(np.float32)
        labels     = load_array(args.labels.resolve()).astype(np.int64).flatten()

        image_paths = None
        if args.image_paths is not None:
            if args.image_paths.suffix.lower() in {".npy", ".npz"}:
                image_paths = [str(x) for x in load_array(args.image_paths).flatten().tolist()]
            else:
                image_paths = [ln.strip() for ln in args.image_paths.read_text().splitlines() if ln.strip()]

        evaluate_space(
            embeddings=embeddings,
            labels=labels,
            args=args,
            device=device,
            output_dir=args.output_dir,
            tag="single",
            image_paths=image_paths,
            embedding_mode="unknown",
            # category_labels / superclass_labels intentionally None — A/B/C gracefully skipped
            annotation=annotation,
        )

    print("Done")


if __name__ == "__main__":
    main()
