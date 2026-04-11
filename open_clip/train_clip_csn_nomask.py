#!/usr/bin/env python3
"""
Train CLIP-frozen visual-only CSN ablation without the masking head.

- Keeps the same CSV / split / caching pipeline as train_clip_csn.py.
- Uses the same projection head and SupCon loss implementation.
- Optimizes only subclass-level SimCLR on the projection output directly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

import torch

CLIP_DIR = Path(__file__).parent.parent / "CLIP"
if str(CLIP_DIR) not in sys.path:
    sys.path.insert(0, str(CLIP_DIR))

import clip  # type: ignore

from csn_pipeline.losses import SupConLoss, two_view_supcon_loss
from csn_pipeline.model import ProjectionHead
from train_clip_csn import (
    CSNIndexMultiViewDataset,
    LossBundle,
    TrainState,
    collate_csn_index_batch,
    create_or_load_balanced_validation_split,
    create_or_load_split,
    default_split_dir_for_csv,
    freeze_clip_model,
    load_csn_records,
    load_or_build_clip_image_cache,
    plot_loss_curves,
    plot_subclass_retrieval_curves,
    rename_user_facing_terms,
    resolve_device,
    save_loss_log,
    set_seed,
    summarize_subclass_retrieval,
)


def compute_losses(
    clip_feature_cache: torch.Tensor,
    batch: dict[str, torch.Tensor],
    image_head: torch.nn.Module,
    supcon_loss: SupConLoss,
    weights: dict[str, float],
    use_amp: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the subclass SimCLR loss (no mask, no superclass term) for one batch.

    Returns (total_loss, {component_name: loss_tensor}).
    """
    device = next(image_head.parameters()).device

    idx_views = torch.stack([batch["idx_view1"], batch["idx_view3"]], dim=1)
    labels_cat = batch["label_view1_3"].to(device=device, non_blocking=True)

    bsz, n_views = idx_views.shape
    idx_flat = idx_views.view(-1).cpu()

    with torch.cuda.amp.autocast(enabled=use_amp):
        img_feat_flat = clip_feature_cache.index_select(0, idx_flat).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        img_proj = image_head(img_feat_flat).view(bsz, n_views, -1)
        l_cat_simclr = two_view_supcon_loss(supcon_loss, img_proj[:, 0], img_proj[:, 1], labels_cat)
        total = weights["w_cat_simclr"] * l_cat_simclr

    zero = torch.zeros_like(l_cat_simclr)
    return total, {"super_simclr": zero, "subclass_simclr": l_cat_simclr}


def run_epoch(
    loader,
    clip_feature_cache: torch.Tensor,
    image_head: torch.nn.Module,
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
        mode = f"Train {epoch}/{total_epochs}"
    else:
        image_head.eval()
        mode = f"Validation {epoch}/{total_epochs}"

    running = {"total": 0.0, "super_simclr": 0.0, "subclass_simclr": 0.0}
    n_batches = 0

    from tqdm import tqdm

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
        for key in ("super_simclr", "subclass_simclr"):
            running[key] += float(parts[key].item())
        n_batches += 1
        pbar.set_postfix(total=f"{running['total'] / n_batches:.4f}")

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
        "uses_mask_head": False,
        "image_head_state_dict": image_head.state_dict(),
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
        torch.save(ckpt, ckpt_dir / "best_checkpoint.pt")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Train frozen-CLIP visual-only CSN no-mask ablation")

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
        help="Weight decay for projection linear layers (image_head). --weight-decay is ignored in no-mask mode.",
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
    """Full training loop (no-mask ablation): data loading, CLIP caching, train/val epochs, checkpointing."""
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
        exp_dir = out_root / f"{model_safe}_csn_nomask_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_ratio = 0.5
    if args.split_dir is None:
        args.split_dir = str(default_split_dir_for_csv(args.csv_file, args.seed, train_ratio))

    print(f"Output dir: {exp_dir}")
    print(f"Device: {device}  AMP(cuda-only): {use_amp}")
    print(f"Split dir: {args.split_dir}")

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

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_head = ProjectionHead(img_dim, args.hidden_dim, args.proj_dim).to(device).float()

    ignored_args = {
        "w_super_simclr": float(args.w_super_simclr),
        "w_super_it": float(args.w_super_it),
        "w_cat_it": float(args.w_cat_it),
        "mask_init": float(args.mask_init),
        "weight_decay": float(args.weight_decay),
    }
    if any(value != 0.0 for value in ignored_args.values()):
        print("Note: no-mask ablation ignores --w-super-simclr, --w-super-it, --w-cat-it, --mask-init, and --weight-decay.")

    train_ds = CSNIndexMultiViewDataset(records=records, indices=train_idx, seed=args.seed)
    val_ds = CSNIndexMultiViewDataset(records=records, indices=val_idx, seed=args.seed + 1)

    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    from torch.utils.data import DataLoader

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
        [{"params": list(image_head.parameters()), "weight_decay": float(args.linear_weight_decay)}],
        lr=args.lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    weights = {"w_cat_simclr": float(args.w_cat_simclr)}

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
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            print(f"Warning: could not load optimizer state from resume checkpoint ({exc}); continuing with fresh optimizer.")
        if use_amp and ckpt.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as exc:
                print(f"Warning: could not load AMP scaler state ({exc}); continuing with fresh scaler.")

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

    config = {
        "args": rename_user_facing_terms(vars(args)),
        "device": str(device),
        "amp_enabled": use_amp,
        "data_stats": data_stats,
        "split_meta": split_meta,
        "val_split_meta": val_split_meta,
        "img_embed_dim": img_dim,
        "visual_only": True,
        "uses_mask_head": False,
        "clip_image_cache_path": str(cache_path),
        "validation_k_values": k_values,
        "best_subclass_selection_metric": "subclass_f1_mean",
        "ignored_args": rename_user_facing_terms(ignored_args),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(
            loader=train_loader,
            clip_feature_cache=clip_feature_cache,
            image_head=image_head,
            supcon_loss=supcon_loss,
            weights=weights,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            train=True,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        state.train_history.append({"epoch": epoch, **asdict(train_loss)})

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        metric_for_best = float("-inf")
        if do_eval:
            val_loss = run_epoch(
                loader=val_loader,
                clip_feature_cache=clip_feature_cache,
                image_head=image_head,
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
                subclass_head=None,
                device=device,
                k_values=k_values,
                batch_size=args.val_metric_batch_size,
            )
            state.val_history.append({"epoch": epoch, **asdict(val_loss), **subclass_metrics})
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
