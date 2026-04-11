#!/usr/bin/env python3
"""Generate CSN embeddings from a trained checkpoint (image branch only).

This is a duplicate of generate_csn_embeddings.py updated for the current model
architecture where the text projection head has been removed. Only image embeddings
are produced; text_super / text_category outputs are omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore

from csn_pipeline.data import load_csn_records
from csn_pipeline.model import ProjectionHead, SharedCSNMask


def resolve_device(device_arg: str, model_name: str) -> torch.device:
    """Select the best available device, applying safety fallbacks for MPS/ResNet."""
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
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(d)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate image CSN embeddings from checkpoint (image branch only)")
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, default=None)
    parser.add_argument("--split-indices", type=str, required=True, help="Path to test_indices.npy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--output-dir", type=str, default="./embeddings")
    parser.add_argument("--output-prefix", type=str, default="csn_test")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--quality-csv", type=str, default=None, help="Optional quality CSV with image_path + score")
    parser.add_argument(
        "--quality-score-column",
        type=str,
        default="agreement_score",
        help="Score column in quality CSV (e.g. agreement_score, is_valid)",
    )
    parser.add_argument(
        "--quality-cutoff",
        type=float,
        default=None,
        help="Keep sample if score >= cutoff (required when --quality-csv is provided)",
    )
    return parser.parse_args()


def _path_key_variants(path_like: str) -> set[str]:
    """Return a set of normalised path strings for robust dict look-up (handles slashes, relative prefixes, etc.)."""
    p = Path(path_like)
    raw = str(path_like).strip()
    out = {raw, raw.replace("\\", "/"), p.as_posix(), p.as_posix().lstrip("./")}
    try:
        rp = p.resolve()
        out.add(str(rp))
        out.add(rp.as_posix())
    except Exception:
        pass
    return {x for x in out if x}


def _parse_quality_score(raw: str) -> float:
    """Convert a raw CSV cell (bool string or float string) to a float quality score."""
    s = str(raw).strip()
    low = s.lower()
    if low in {"true", "t", "yes", "y"}:
        return 1.0
    if low in {"false", "f", "no", "n"}:
        return 0.0
    return float(s)


def load_quality_scores(
    quality_csv: str | Path,
    score_column: str,
    base_image_dir: str | Path | None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Read a quality CSV and return a {path_key: score} map and a parse-statistics dict."""
    quality_csv = Path(quality_csv)
    if not quality_csv.exists():
        raise FileNotFoundError(f"Quality CSV not found: {quality_csv}")

    base_dir = Path(base_image_dir) if base_image_dir else None
    quality_map: dict[str, float] = {}
    stats = {"rows_total": 0, "rows_parsed": 0, "rows_bad_score": 0, "rows_missing_path": 0}

    with open(quality_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if "image_path" not in cols:
            raise ValueError("Quality CSV must contain 'image_path' column")
        if score_column not in cols:
            raise ValueError(f"Quality CSV missing score column '{score_column}'")

        for row in reader:
            stats["rows_total"] += 1
            rel_path = str(row.get("image_path", "")).strip()
            if not rel_path:
                stats["rows_missing_path"] += 1
                continue
            try:
                score = _parse_quality_score(str(row.get(score_column, "")))
            except Exception:
                stats["rows_bad_score"] += 1
                continue

            for k in _path_key_variants(rel_path):
                quality_map[k] = score
            if base_dir is not None:
                abs_path = (base_dir / rel_path).resolve()
                for k in _path_key_variants(str(abs_path)):
                    quality_map[k] = score
            stats["rows_parsed"] += 1

    if not quality_map:
        raise ValueError(f"No usable quality scores found in {quality_csv}")
    return quality_map, stats


def filter_split_indices_by_quality(
    records,
    split_indices: np.ndarray,
    quality_map: dict[str, float],
    cutoff: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return a filtered index array keeping only samples whose quality score >= cutoff.

    Samples with no entry in quality_map are kept unconditionally.
    Returns the filtered indices and a statistics dict.
    """
    kept: list[int] = []
    matched = 0
    dropped_below = 0
    missing_score_kept = 0

    for idx in split_indices.astype(np.int64).tolist():
        rec = records[int(idx)]
        score = None
        for k in _path_key_variants(rec.image_path):
            if k in quality_map:
                score = quality_map[k]
                break
        if score is None:
            missing_score_kept += 1
            kept.append(int(idx))
            continue
        matched += 1
        if float(score) >= float(cutoff):
            kept.append(int(idx))
        else:
            dropped_below += 1

    stats = {
        "input_count": int(split_indices.shape[0]),
        "kept_count": int(len(kept)),
        "matched_scores": int(matched),
        "dropped_below_cutoff": int(dropped_below),
        "missing_score_kept": int(missing_score_kept),
    }
    return np.asarray(kept, dtype=np.int64), stats


def main() -> None:
    """Load checkpoint, apply image-head + CSN mask, and save superclass/subclass embeddings."""
    args = parse_args()
    device = resolve_device(args.device, args.model)

    records, _ = load_csn_records(args.csv_file, args.base_image_dir)
    split_indices = np.load(args.split_indices).astype(np.int64)
    if split_indices.size == 0:
        raise ValueError("split_indices is empty")

    max_idx = len(records) - 1
    if int(split_indices.max()) > max_idx:
        raise ValueError("split_indices contains index beyond record count")

    quality_filter = None
    if args.quality_csv is not None:
        if args.quality_cutoff is None:
            raise ValueError("--quality-cutoff is required when --quality-csv is provided")
        quality_map, quality_csv_stats = load_quality_scores(
            quality_csv=args.quality_csv,
            score_column=args.quality_score_column,
            base_image_dir=args.base_image_dir,
        )
        split_indices, filter_stats = filter_split_indices_by_quality(
            records=records,
            split_indices=split_indices,
            quality_map=quality_map,
            cutoff=float(args.quality_cutoff),
        )
        quality_filter = {
            "quality_csv": str(Path(args.quality_csv).resolve()),
            "score_column": str(args.quality_score_column),
            "cutoff": float(args.quality_cutoff),
            "quality_csv_stats": quality_csv_stats,
            "filter_stats": filter_stats,
        }
        print(
            "Quality filtering: "
            f"kept={filter_stats['kept_count']} / {filter_stats['input_count']}  "
            f"dropped_below_cutoff={filter_stats['dropped_below_cutoff']}  "
            f"missing_score_kept={filter_stats['missing_score_kept']}"
        )
        if split_indices.size == 0:
            raise ValueError("No test samples left after quality filtering")

    model, preprocess = clip.load(args.model, device=device, jit=False)
    model = model.float().eval()

    with torch.no_grad():
        dummy_img = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device=device, dtype=torch.float32)
        img_dim = int(model.encode_image(dummy_img).shape[-1])

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)

    ckpt_args = ckpt.get("args", {})
    hidden_dim = int(ckpt_args.get("hidden_dim", 512))
    proj_dim = int(ckpt_args.get("proj_dim", 128))
    mask_init = float(ckpt_args.get("mask_init", 0.0))

    image_head = ProjectionHead(img_dim, hidden_dim, proj_dim).to(device).float().eval()
    csn_mask = SharedCSNMask(proj_dim, mask_init=mask_init).to(device).float().eval()

    image_head.load_state_dict(ckpt["image_head_state_dict"], strict=True)
    csn_mask.load_state_dict(ckpt["csn_mask_state_dict"], strict=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_super_all = []
    image_subclass_all = []
    superclass_ids = []
    subclass_ids = []
    paths = []

    for i in tqdm(range(0, split_indices.size, args.batch_size), desc="Embedding batches"):
        batch_idxs = split_indices[i : i + args.batch_size]
        imgs = []
        sup = []
        cat = []
        pths = []

        for idx in batch_idxs.tolist():
            r = records[int(idx)]
            try:
                img = Image.open(r.image_path).convert("RGB")
                img_t = preprocess(img)
            except Exception:
                img_t = preprocess(Image.new("RGB", (224, 224), color="black"))
            imgs.append(img_t)
            sup.append(int(r.superclass_id))
            cat.append(int(r.category_id))
            pths.append(r.image_path)

        x_img = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            f_img = model.encode_image(x_img).float()

            z_img = image_head(f_img)
            z_img_subclass = csn_mask(z_img)

            if args.normalize:
                z_img = F.normalize(z_img, dim=-1)
                z_img_subclass = F.normalize(z_img_subclass, dim=-1)

        image_super_all.append(z_img.cpu().numpy())
        image_subclass_all.append(z_img_subclass.cpu().numpy())
        superclass_ids.extend(sup)
        subclass_ids.extend(cat)
        paths.extend(pths)

    image_super = np.vstack(image_super_all)
    image_subclass = np.vstack(image_subclass_all)

    superclass_ids_np = np.asarray(superclass_ids, dtype=np.int64)
    subclass_ids_np = np.asarray(subclass_ids, dtype=np.int64)
    paths_np = np.asarray(paths)

    prefix = args.output_prefix
    out_files = {
        "image_super": out_dir / f"{prefix}_image_super_embeddings.npy",
        "image_subclass": out_dir / f"{prefix}_image_subclass_embeddings.npy",
        "superclass_ids": out_dir / f"{prefix}_superclass_ids.npy",
        "subclass_ids": out_dir / f"{prefix}_subclass_ids.npy",
        "paths": out_dir / f"{prefix}_paths.npy",
        "metadata": out_dir / f"{prefix}_metadata.json",
    }

    np.save(out_files["image_super"], image_super)
    np.save(out_files["image_subclass"], image_subclass)
    np.save(out_files["superclass_ids"], superclass_ids_np)
    np.save(out_files["subclass_ids"], subclass_ids_np)
    np.save(out_files["paths"], paths_np)

    metadata = {
        "csv_file": str(Path(args.csv_file).resolve()),
        "split_indices": str(Path(args.split_indices).resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "model": args.model,
        "device": str(device),
        "normalize": bool(args.normalize),
        "num_samples": int(image_super.shape[0]),
        "proj_dim": int(image_super.shape[1]),
        "quality_filter": quality_filter,
        "outputs": {k: str(v) for k, v in out_files.items()},
    }
    with open(out_files["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    for name, p in out_files.items():
        print(f"Saved {name}: {p}")


if __name__ == "__main__":
    main()
