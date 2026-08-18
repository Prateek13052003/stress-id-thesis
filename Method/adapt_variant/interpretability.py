"""
Mechanistic interpretability for the physio-anchored ADAPT variant — the piece
that is absent from both the missing-modality survey (Wu et al.) and ADAPT
itself (Mordacq et al.): neither paper does head-level or exact-attribution
analysis of *why* a missing-modality model relies on what it relies on.

Two analyses, both cheap because the model is tiny (5 modalities, 1-2
transformer layers, 4 heads):

1. Exact Shapley utilization score. With only 5 modalities there are 2^5=32
   coalitions total — small enough to enumerate exactly, no Monte Carlo
   approximation needed. The value function v(S) is the model's predicted
   probability of the TRUE class when only modalities in S are visible
   (others forced-masked via ablate_modalities). This gives a per-sample,
   per-modality attribution that sums exactly to v(full) - v(empty).

2. Attention-head ablation. Zero one head at a time in the fusion transformer
   and measure the balanced-accuracy drop on the test set, to see whether
   heads specialize (e.g. one head carrying most of the anchor<->EDA path).

Usage:
    python3 -m Method.adapt_variant.interpretability --n-folds 1
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Method.adapt_variant.data import MODALITIES, build_full_table, subject_groups
from Method.adapt_variant.train import (
    set_seed, run_one_fold, collate,
)
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

OUT_DIR = Path(__file__).resolve().parent / "outputs"


@torch.no_grad()
def true_class_prob_for_coalition(model, loader, device, present_modalities):
    """v(S): predicted probability of the true class, averaged as a per-sample
    vector, when only `present_modalities` are visible to the model."""
    missing = [m for m in model.modalities if m not in present_modalities]
    probs, ys, indices = [], [], []
    for feats, avail, y in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        avail = avail.to(device)
        logits, _ = model(feats, avail, ablate_modalities=missing if missing else None)
        p = F.softmax(logits, dim=-1)
        probs.append(p.gather(1, y.to(device).view(-1, 1)).squeeze(1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy()


def exact_shapley(model, loader, device, modalities):
    """Returns DataFrame: rows = test samples, columns = modalities, values =
    exact per-sample Shapley contribution to true-class probability."""
    n = len(modalities)
    v = {}
    for r in range(n + 1):
        for S in itertools.combinations(modalities, r):
            v[frozenset(S)] = true_class_prob_for_coalition(model, loader, device, set(S))

    n_samples = len(next(iter(v.values())))
    shapley = {m: np.zeros(n_samples) for m in modalities}
    fact = math.factorial
    for i in modalities:
        others = [m for m in modalities if m != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                weight = fact(len(S)) * fact(n - len(S) - 1) / fact(n)
                shapley[i] += weight * (v[S | {i}] - v[S])

    return pd.DataFrame(shapley), v[frozenset(modalities)], v[frozenset()]


@torch.no_grad()
def head_ablation(model, loader, device, n_layers, n_heads):
    baseline = evaluate_balanced_acc(model, loader, device)
    rows = [{"layer": None, "head": None, "balanced_acc": baseline, "delta": 0.0}]
    for layer in range(n_layers):
        for head in range(n_heads):
            acc = evaluate_balanced_acc(model, loader, device, ablate_heads=[head], ablate_layer=layer)
            rows.append({"layer": layer, "head": head, "balanced_acc": acc, "delta": acc - baseline})
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_balanced_acc(model, loader, device, ablate_heads=None, ablate_layer=None):
    preds, ys = [], []
    for feats, avail, y in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        avail = avail.to(device)
        logits, _ = model(feats, avail, ablate_heads=ablate_heads, ablate_layer=ablate_layer)
        preds.append(logits.argmax(-1).cpu())
        ys.append(y)
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    return balanced_accuracy_score(ys, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="binary-stress")
    ap.add_argument("--anchor", default="ecg", choices=MODALITIES)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs-anchor", type=int, default=150)
    ap.add_argument("--epochs-fusion", type=int, default=150)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.08)
    ap.add_argument("--dropout-p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1999)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    OUT_DIR.mkdir(exist_ok=True)

    X, mask, y, groups = build_full_table(args.target)
    grp = subject_groups(X.index)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_pos, test_pos = next(splitter.split(X, y, grp))
    train_idx, test_idx = X.index[train_pos], X.index[test_pos]

    print(f"Training on {len(set(grp[train_pos]))} subjects (with an inner "
          f"validation split for early stopping), interpreting on "
          f"{len(set(grp[test_pos]))} held-out test subjects...")
    model, results, extra = run_one_fold(X, mask, y, groups, train_idx, test_idx, args, device)
    print(f"Train balanced_acc: {results['train']['balanced_acc']:.3f}  "
          f"Val balanced_acc: {results['val']['balanced_acc']:.3f}  "
          f"Test balanced_acc: {results['full']['balanced_acc']:.3f}")

    # Reuse the EXACT test loader (and the standardization stats fit only on
    # the inner fit-subjects) that the model was trained and evaluated
    # against — rebuilding it independently here would silently use
    # different stats (fit on train_idx, which includes the validation
    # subjects) than what the model actually saw during training.
    test_loader = extra["test_loader"]

    print("\nComputing exact Shapley utilization scores (32 coalitions)...")
    shap_df, v_full, v_empty = exact_shapley(model, test_loader, device, MODALITIES)
    shap_df.index = test_idx
    shap_df["task"] = [i.split("_", 1)[1] for i in test_idx]
    shap_df.to_csv(OUT_DIR / f"shapley_utilization_{args.target}.csv")

    print("\nMean Shapley utilization per modality (share of true-class probability "
          "attributable to each modality, averaged over test samples):")
    print(shap_df[MODALITIES].mean().sort_values(ascending=False).round(4))

    print("\nMean Shapley utilization per modality, broken down by task:")
    task_means = shap_df.groupby("task")[MODALITIES].mean().round(4)
    print(task_means)
    task_means.to_csv(OUT_DIR / f"shapley_by_task_{args.target}.csv")

    print("\nRunning attention-head ablation on the fusion transformer...")
    n_layers = len(model.fusion.blocks)
    n_heads = model.fusion.blocks[0].attn.n_heads
    head_df = head_ablation(model, test_loader, device, n_layers, n_heads)
    head_df.to_csv(OUT_DIR / f"head_ablation_{args.target}.csv", index=False)
    print(head_df.to_string(index=False))


if __name__ == "__main__":
    main()
