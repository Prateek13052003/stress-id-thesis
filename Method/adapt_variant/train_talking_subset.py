"""
Runs the same fixed, regularized, early-stopped pipeline (train.py) restricted
to StressID's 7 "talking" tasks only (Counting1-3, Math, Reading, Speaking,
Stroop) — the subset where every task's majority class is "stressed", so the
task-identity confound documented in Classification/task_identity_baseline.py
mathematically collapses to exactly balanced_acc=0.5 (see
Classification/talking_subset_eval.py, which verified this: majority-vote and
RF-on-task-identity both score exactly 0.500 here, and even RF on real
physiological features is at chance: 0.497+/-0.012).

This is also the subset with audio available (StressID only records audio
for these 7 tasks), so it is the closest match to what ADAPT calls X*_test.

Usage:
    python3 -m Method.adapt_variant.train_talking_subset --anchor ecg --n-folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import KFold

from Method.adapt_variant.data import MODALITIES, build_full_table, subject_groups
from Method.adapt_variant.train import run_one_fold, set_seed

OUT_DIR = Path(__file__).resolve().parent / "outputs"
TALKING_TASKS = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="binary-stress")
    ap.add_argument("--anchor", default="ecg", choices=MODALITIES)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--leaky", action="store_true",
                     help="Use plain KFold (task-wise, matches StressID paper's own "
                          "protocol) for the OUTER test split instead of subject-wise "
                          "GroupKFold, to measure how much subject leakage alone "
                          "inflates results. The inner validation split used for "
                          "early stopping stays subject-wise either way.")
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
    task = X.index.to_series().apply(lambda i: i.split("_", 1)[1])
    keep = task.isin(TALKING_TASKS)
    X, mask, y = X[keep], mask[keep], y[keep]
    grp = subject_groups(X.index)

    print(f"Talking-tasks-only subset: n={len(X)} rows, {len(set(grp))} subjects, "
          f"anchor={args.anchor}, leaky={args.leaky}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    if args.leaky:
        splitter = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = splitter.split(X, y)
    else:
        splitter = GroupKFold(n_splits=args.n_folds)
        splits = splitter.split(X, y, grp)

    all_results = []
    for fold, (train_pos, test_pos) in enumerate(splits):
        train_idx, test_idx = X.index[train_pos], X.index[test_pos]
        overlap = len(set(grp[train_pos]) & set(grp[test_pos])) / len(set(grp[test_pos]))
        print(f"\n=== Fold {fold + 1} ({len(set(grp[train_pos]))} train subjects, "
              f"{len(set(grp[test_pos]))} test subjects, subject_overlap={overlap:.1%}) ===")
        _, results, _ = run_one_fold(X, mask, y, groups, train_idx, test_idx, args, device,
                                      anchor=args.anchor, verbose=True)
        for scenario, metrics in results.items():
            print(f"  {scenario:32s} balanced_acc={metrics['balanced_acc']:.4f}  f1={metrics['f1']:.4f}")
        all_results.append(results)

    chance = 1 / int(y.nunique())
    print(f"\n{'=' * 70}\nSummary across {args.n_folds} folds (talking-tasks-only, anchor={args.anchor}, "
          f"chance={chance:.3f})\n{'=' * 70}")
    summary = {}
    for scenario in all_results[0]:
        accs = [r[scenario]["balanced_acc"] for r in all_results]
        summary[scenario] = {"balanced_acc_mean": float(np.mean(accs)), "balanced_acc_std": float(np.std(accs))}
        print(f"  {scenario:32s} balanced_acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  "
              f"(delta vs chance {chance:.3f}: {np.mean(accs) - chance:+.4f})")

    leaky_tag = "-leaky" if args.leaky else ""
    with open(OUT_DIR / f"results_talking-subset_{args.target}_anchor-{args.anchor}{leaky_tag}.json", "w") as f:
        json.dump({"args": vars(args), "n_rows": len(X), "n_subjects": int(len(set(grp))),
                    "per_fold": all_results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
