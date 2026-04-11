#!/usr/bin/env python3
"""Generate image-only no-mask CSN ablation embeddings from a trained checkpoint."""

from __future__ import annotations

import argparse
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
from csn_pipeline.model import ProjectionHead
from generate_csn_embeddings_image import (
    filter_split_indices_by_quality,
    load_quality_scores,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate image-only no-mask CSN embeddings from checkpoint")
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


def main() -> None:
    """Load checkpoint, encode images through the CSN projection head, and save embeddings to disk."""
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

    image_head = ProjectionHead(img_dim, hidden_dim, proj_dim).to(device).float().eval()
    image_head.load_state_dict(ckpt["image_head_state_dict"], strict=True)

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
            rec = records[int(idx)]
            try:
                img = Image.open(rec.image_path).convert("RGB")
                img_t = preprocess(img)
            except Exception:
                img_t = preprocess(Image.new("RGB", (224, 224), color="black"))
            imgs.append(img_t)
            sup.append(int(rec.superclass_id))
            cat.append(int(rec.category_id))
            pths.append(rec.image_path)

        x_img = torch.stack(imgs, dim=0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            f_img = model.encode_image(x_img).float()
            z_img = image_head(f_img)
            if args.normalize:
                z_img = F.normalize(z_img, dim=-1)

        z_img_np = z_img.cpu().numpy()
        image_super_all.append(z_img_np)
        image_subclass_all.append(z_img_np.copy())
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
        "image_only": True,
        "uses_mask_head": False,
        "image_subclass_equals_image_super": True,
        "quality_filter": quality_filter,
        "outputs": {key: str(value) for key, value in out_files.items()},
    }
    with open(out_files["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    for name, path in out_files.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
