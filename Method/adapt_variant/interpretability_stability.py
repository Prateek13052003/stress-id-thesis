"""
Repeats the interpretability pipeline (exact Shapley utilization + head
ablation) across multiple TRAINING seeds, on the exact same held-out test
subjects every time.

Why: the first two interpretability runs in this project gave two different
answers — "EDA matters most for Math" in the first (overfit) model, and
"head 2 is the important head" in that run, neither of which reappeared after
the overfitting fix (second run: EDA negative on Math, head 1 was important
instead of head 2). That is a direct demonstration that a SINGLE run's
interpretability numbers are not reliable evidence by themselves. This script
answers the actual question properly: which findings survive across many
independently-trained models on the same test subjects, and which don't.

Only the outer train/test subject split is held fixed (via --split-seed).
Everything about how each model is trained — weight init, the inner
validation carve-out used for early stopping, modality-dropout draws — is
allowed to vary with --seed, because that is exactly the kind of variation a
real "retrain with a different seed" would produce.

Usage:
    python3 -m Method.adapt_variant.interpretability_stability --n-seeds 10
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from Method.adapt_variant.data import MODALITIES, build_full_table, subject_groups
from Method.adapt_variant.train import run_one_fold, set_seed, collate
from Method.adapt_variant.interpretability import exact_shapley, head_ablation

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="binary-stress")
    ap.add_argument("--anchor", default="ecg", choices=MODALITIES)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--split-seed", type=int, default=1999,
                     help="Fixes which subjects are train vs test, held constant across all runs.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs-anchor", type=int, default=150)
    ap.add_argument("--epochs-fusion", type=int, default=150)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.08)
    ap.add_argument("--dropout-p", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    OUT_DIR.mkdir(exist_ok=True)

    X, mask, y, groups = build_full_table(args.target)
    grp = subject_groups(X.index)

    # Outer split fixed ONCE, independent of the training seed loop below.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.split_seed)
    train_pos, test_pos = next(splitter.split(X, y, grp))
    train_idx, test_idx = X.index[train_pos], X.index[test_pos]
    print(f"Fixed outer split: {len(set(grp[train_pos]))} train subjects, "
          f"{len(set(grp[test_pos]))} test subjects (split_seed={args.split_seed}). "
          f"This does NOT change across the {args.n_seeds} training seeds below.\n")

    per_seed_shapley_mean = []      # list of Series (modality -> mean shapley), one per seed
    per_seed_shapley_by_task = []   # list of DataFrame (task x modality), one per seed
    per_seed_head_important = []    # per seed: which head index had the most negative delta
    per_seed_head_deltas = []       # list of dict head_idx -> delta
    per_seed_test_acc = []

    for seed in range(args.n_seeds):
        set_seed(seed)
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = seed

        model, results, extra = run_one_fold(X, mask, y, groups, train_idx, test_idx, run_args, device,
                                              anchor=args.anchor, verbose=False)
        test_loader = extra["test_loader"]
        test_acc = results["full"]["balanced_acc"]
        per_seed_test_acc.append(test_acc)

        shap_df, v_full, v_empty = exact_shapley(model, test_loader, device, MODALITIES)
        shap_df.index = test_idx
        shap_df["task"] = [i.split("_", 1)[1] for i in test_idx]
        per_seed_shapley_mean.append(shap_df[MODALITIES].mean())
        per_seed_shapley_by_task.append(shap_df.groupby("task")[MODALITIES].mean())

        n_layers = len(model.fusion.blocks)
        n_heads = model.fusion.blocks[0].attn.n_heads
        head_df = head_ablation(model, test_loader, device, n_layers, n_heads)
        head_only = head_df[head_df["head"].notna()]
        most_important = int(head_only.loc[head_only["delta"].idxmin(), "head"])
        per_seed_head_important.append(most_important)
        per_seed_head_deltas.append(dict(zip(head_only["head"].astype(int), head_only["delta"])))

        print(f"seed {seed:2d}: test_acc={test_acc:.3f}  "
              f"top-utilization modality={shap_df[MODALITIES].mean().idxmax()}  "
              f"most-important head={most_important}")

    print(f"\n{'=' * 70}\nSTABILITY ACROSS {args.n_seeds} SEEDS (same {len(set(grp[test_pos]))} "
          f"test subjects every time)\n{'=' * 70}")

    print(f"\nTest balanced_acc across seeds: {np.mean(per_seed_test_acc):.3f} "
          f"+/- {np.std(per_seed_test_acc):.3f}  (min={min(per_seed_test_acc):.3f}, "
          f"max={max(per_seed_test_acc):.3f})")

    shap_mean_df = pd.DataFrame(per_seed_shapley_mean)  # rows=seed, cols=modality
    print("\nMean Shapley utilization per modality, aggregated over seeds "
          "(mean +/- std across seeds; how often it was the single top modality that seed):")
    top_counts = Counter(shap_mean_df.idxmax(axis=1))
    for m in MODALITIES:
        print(f"  {m:6s} mean={shap_mean_df[m].mean():+.4f}  std={shap_mean_df[m].std():.4f}  "
              f"top-modality in {top_counts.get(m, 0)}/{args.n_seeds} seeds")

    # Stack per-task-per-modality Shapley values across seeds
    all_tasks = sorted(set().union(*[df.index for df in per_seed_shapley_by_task]))
    task_stability = {}
    for task in all_tasks:
        row = {}
        for m in MODALITIES:
            vals = [df.loc[task, m] for df in per_seed_shapley_by_task if task in df.index]
            row[m] = (float(np.mean(vals)), float(np.std(vals)),
                       sum(v > 0 for v in vals), len(vals))
        task_stability[task] = row

    print("\nEDA utilization on Math/Counting tasks specifically (the claim from the "
          "first, overfit run) — mean +/- std across seeds, and how many seeds it was positive in:")
    for task in [t for t in all_tasks if t in ("Math", "Counting1", "Counting2", "Counting3")]:
        m, s, pos, n = task_stability[task]["eda"]
        print(f"  EDA on {task:10s}: mean={m:+.4f}  std={s:.4f}  positive in {pos}/{n} seeds")

    head_count = Counter(per_seed_head_important)
    print(f"\nWhich attention head was 'most important' (most negative delta when ablated), "
          f"tallied across {args.n_seeds} seeds:")
    for h in range(n_heads):
        print(f"  head {h}: chosen as most-important in {head_count.get(h, 0)}/{args.n_seeds} seeds")

    out = {
        "args": vars(args),
        "per_seed_test_acc": per_seed_test_acc,
        "per_seed_shapley_mean": [s.to_dict() for s in per_seed_shapley_mean],
        "per_seed_shapley_by_task": {str(i): df.to_dict() for i, df in enumerate(per_seed_shapley_by_task)},
        "per_seed_most_important_head": per_seed_head_important,
        "per_seed_head_deltas": per_seed_head_deltas,
        "modality_top_counts": {m: top_counts.get(m, 0) for m in MODALITIES},
        "head_importance_counts": {str(h): head_count.get(h, 0) for h in range(n_heads)},
    }
    with open(OUT_DIR / f"interpretability_stability_{args.target}_anchor-{args.anchor}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full per-seed data to outputs/interpretability_stability_{args.target}_anchor-{args.anchor}.json")


if __name__ == "__main__":
    main()
