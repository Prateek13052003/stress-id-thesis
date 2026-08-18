"""
Controlled comparison: which modality should be the anchor?

ADAPT (Mordacq et al.) only ever anchors StressID to video. This script trains
the SAME fixed, regularized, early-stopped pipeline (train.py) with every one
of the 5 modalities as anchor in turn, reusing the IDENTICAL GroupKFold folds
across all 5 conditions (same seed, same subject splits) so the comparison
isolates the effect of the anchor choice and nothing else.

Usage:
    python3 -m Method.adapt_variant.anchor_sweep --n-folds 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold

from Method.adapt_variant.data import MODALITIES, build_full_table, subject_groups
from Method.adapt_variant.train import run_one_fold, set_seed

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="binary-stress")
    ap.add_argument("--n-folds", type=int, default=5)
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
    import torch
    device = torch.device(args.device)

    X, mask, y, groups = build_full_table(args.target)
    grp = subject_groups(X.index)
    OUT_DIR.mkdir(exist_ok=True)

    # Compute the folds ONCE so every anchor condition trains/tests on the
    # exact same subject splits.
    splitter = GroupKFold(n_splits=args.n_folds)
    folds = list(splitter.split(X, y, grp))

    all_results = {}
    for anchor in MODALITIES:
        print(f"\n{'=' * 70}\nANCHOR = {anchor}\n{'=' * 70}")
        fold_results = []
        for fold, (train_pos, test_pos) in enumerate(folds):
            train_idx, test_idx = X.index[train_pos], X.index[test_pos]
            _, results, _ = run_one_fold(X, mask, y, groups, train_idx, test_idx, args, device,
                                          anchor=anchor, verbose=False)
            fold_results.append(results)
            print(f"  fold {fold + 1}: test balanced_acc={results['full']['balanced_acc']:.3f}  "
                  f"f1={results['full']['f1']:.3f}  "
                  f"(train={results['train']['balanced_acc']:.3f}, val={results['val']['balanced_acc']:.3f})")
        all_results[anchor] = fold_results

    print(f"\n{'=' * 70}\nSUMMARY: anchor comparison, {args.n_folds}-fold subject-wise CV, "
          f"identical folds across anchors\n{'=' * 70}")
    summary = {}
    header = f"{'anchor':8s} {'train_acc':>12s} {'val_acc':>12s} {'test_acc':>12s} {'test_f1':>12s} " \
             f"{'no_video':>12s} {'no_audio':>12s} {'real_life':>12s}"
    print(header)
    for anchor, fold_results in all_results.items():
        def agg(scenario, metric):
            vals = [r[scenario][metric] for r in fold_results]
            return np.mean(vals), np.std(vals)

        tr_m, tr_s = agg("train", "balanced_acc")
        va_m, va_s = agg("val", "balanced_acc")
        te_m, te_s = agg("full", "balanced_acc")
        f1_m, f1_s = agg("full", "f1")
        nv_m, nv_s = agg("no_video", "balanced_acc")
        na_m, na_s = agg("no_audio", "balanced_acc")
        rl_m, rl_s = agg("real_life_no_video_no_audio", "balanced_acc")

        summary[anchor] = {
            "train_acc": [tr_m, tr_s], "val_acc": [va_m, va_s],
            "test_acc": [te_m, te_s], "test_f1": [f1_m, f1_s],
            "no_video_acc": [nv_m, nv_s], "no_audio_acc": [na_m, na_s],
            "real_life_acc": [rl_m, rl_s],
        }
        print(f"{anchor:8s} {tr_m:.3f}+-{tr_s:.3f}  {va_m:.3f}+-{va_s:.3f}  "
              f"{te_m:.3f}+-{te_s:.3f}  {f1_m:.3f}+-{f1_s:.3f}  "
              f"{nv_m:.3f}+-{nv_s:.3f}  {na_m:.3f}+-{na_s:.3f}  {rl_m:.3f}+-{rl_s:.3f}")

    best_anchor = max(summary, key=lambda a: summary[a]["test_acc"][0])
    print(f"\nBest test balanced_acc: anchor={best_anchor} "
          f"({summary[best_anchor]['test_acc'][0]:.3f})")

    # Paired significance test: since every anchor was trained/tested on the
    # IDENTICAL folds, per-fold test scores are paired samples, so a paired
    # t-test is the right tool (not an unpaired test on the means).
    per_fold_test_acc = {
        a: [r["full"]["balanced_acc"] for r in fold_results]
        for a, fold_results in all_results.items()
    }
    print(f"\n{'=' * 70}\nPaired significance (per-fold test balanced_acc), physio anchors vs "
          f"video/audio\n{'=' * 70}")
    sig_tests = {}
    physio_anchors = [m for m in MODALITIES if m in ("ecg", "eda", "resp")]
    for physio in physio_anchors:
        for other in ("video", "audio"):
            a, b = per_fold_test_acc[physio], per_fold_test_acc[other]
            if len(a) >= 2:
                t_stat, p_val = stats.ttest_rel(a, b)
                wins = sum(x > y for x, y in zip(a, b))
                mean_diff = float(np.mean(a) - np.mean(b))
                key = f"{physio}_vs_{other}"
                sig_tests[key] = {"wins": f"{wins}/{len(a)}", "mean_diff": mean_diff,
                                   "paired_ttest_p": float(p_val)}
                print(f"  {physio:5s} vs {other:5s}: wins {wins}/{len(a)} folds, "
                      f"mean diff={mean_diff:+.3f}, paired t-test p={p_val:.3f}")
    if args.n_folds < 8:
        print(f"\n  NOTE: only {args.n_folds} folds -> low statistical power. Treat p-values "
              f"as suggestive, not confirmatory; rerun with more folds/seeds before citing "
              f"significance in the thesis.")

    with open(OUT_DIR / f"anchor_sweep_{args.target}.json", "w") as f:
        json.dump({
            "args": vars(args),
            "summary": summary,
            "per_fold_test_balanced_acc": per_fold_test_acc,
            "paired_significance_tests": sig_tests,
        }, f, indent=2)


if __name__ == "__main__":
    main()
