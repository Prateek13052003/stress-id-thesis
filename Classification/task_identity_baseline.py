"""
Tests a specific, falsifiable claim: does knowing ONLY the task name (no
physiological signal at all) let a classifier match the deep model's
cross-task accuracy? If yes, the pooled 11-task benchmark is not actually
measuring stress detection from signal — it's measuring task-recognition,
because binary-stress is itself largely determined by which task a row
belongs to (see the crosstab this script also prints: Relax/Breathing are
83-87% non-stress, Counting/Math/Speaking/Stroop are 65-77% stress, by
protocol design, not by individual physiology).

Two versions of the baseline, both subject-wise GroupKFold (no leakage):
  1. Majority-vote lookup: for each test row, predict the majority
     binary-stress label that task had IN THE TRAINING FOLD ONLY. Zero
     model, zero physiological signal — pure task-name lookup.
  2. RandomForest on one-hot task identity as the only feature (same
     classifier used in dataset_audit.py, for a fair apples-to-apples
     comparison against the physiological-feature RF result already
     computed there).

Usage:
    python3 -m Classification.task_identity_baseline
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

from Classification.metrics_utils import within_task_balanced_accuracy

DATASET_DIR = "Dataset"


def majority_vote_baseline(task, y, groups, n_splits=5, seed=0):
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(task, y, groups):
        train_task, train_y = task.iloc[train_idx], y.iloc[train_idx]
        majority = train_y.groupby(train_task).agg(lambda s: s.mode().iloc[0])
        overall_majority = train_y.mode().iloc[0]
        preds = task.iloc[test_idx].map(majority).fillna(overall_majority).astype(int)
        within_task_bacc, _ = within_task_balanced_accuracy(
            y.iloc[test_idx], preds, task.iloc[test_idx])
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], preds),
            "f1": f1_score(y.iloc[test_idx], preds, average="weighted"),
            "within_task_balanced_acc": within_task_bacc,
        })
    return pd.DataFrame(scores)


def rf_on_task_onehot(task, y, groups, n_splits=5, seed=0):
    X = pd.get_dummies(task)
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = RandomForestClassifier(max_depth=5, random_state=seed)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        within_task_bacc, _ = within_task_balanced_accuracy(
            y.iloc[test_idx], preds, task.iloc[test_idx])
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], preds),
            "f1": f1_score(y.iloc[test_idx], preds, average="weighted"),
            "within_task_balanced_acc": within_task_bacc,
        })
    return pd.DataFrame(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="binary-stress", choices=["binary-stress", "affect3-class"])
    args = ap.parse_args()
    n_classes = 3 if args.label == "affect3-class" else 2
    chance = 1 / n_classes

    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0).dropna()
    y = labels[args.label].astype(int)
    task = y.index.to_series().apply(lambda i: i.split("_", 1)[1])
    groups = y.index.to_series().apply(lambda i: i.split("_", 1)[0])

    print(f"LABEL = {args.label}  (chance level = {chance:.3f})")
    print(f"n={len(y)} rows, {groups.nunique()} subjects, {task.nunique()} tasks")
    print(f"Overall class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print()

    print("=" * 70)
    print("1. Majority-vote-per-task lookup (zero physiological signal)")
    print("=" * 70)
    mv = majority_vote_baseline(task, y, groups, n_splits=5, seed=0)
    print(f"pooled balanced_acc      = {mv['balanced_acc'].mean():.3f} +/- {mv['balanced_acc'].std():.3f}")
    print(f"f1                       = {mv['f1'].mean():.3f} +/- {mv['f1'].std():.3f}")
    print(f"WITHIN-TASK balanced_acc = {mv['within_task_balanced_acc'].mean():.3f} "
          f"+/- {mv['within_task_balanced_acc'].std():.3f}  "
          f"(should be ~{chance:.3f}: this baseline has zero information once task is held constant)")
    print(f"per-fold pooled balanced_acc: {mv['balanced_acc'].round(3).tolist()}")
    print()

    print("=" * 70)
    print("2. RandomForest on one-hot task identity only (same RF config as "
          "Classification/dataset_audit.py)")
    print("=" * 70)
    rf = rf_on_task_onehot(task, y, groups, n_splits=5, seed=0)
    print(f"pooled balanced_acc      = {rf['balanced_acc'].mean():.3f} +/- {rf['balanced_acc'].std():.3f}")
    print(f"f1                       = {rf['f1'].mean():.3f} +/- {rf['f1'].std():.3f}")
    print(f"WITHIN-TASK balanced_acc = {rf['within_task_balanced_acc'].mean():.3f} "
          f"+/- {rf['within_task_balanced_acc'].std():.3f}  "
          f"(should be ~{chance:.3f}: this baseline has zero information once task is held constant)")
    print(f"per-fold pooled balanced_acc: {rf['balanced_acc'].round(3).tolist()}")

    # --- save proof to a file, not just the terminal ---
    import os
    os.makedirs("outputs", exist_ok=True)
    suffix = "" if args.label == "binary-stress" else f"_{args.label}"

    summary = pd.DataFrame([
        {"label": args.label, "chance": chance, "baseline": "majority_vote_per_task",
         "pooled_balanced_acc": mv["balanced_acc"].mean(),
         "pooled_balanced_acc_std": mv["balanced_acc"].std(),
         "within_task_balanced_acc": mv["within_task_balanced_acc"].mean(),
         "within_task_balanced_acc_std": mv["within_task_balanced_acc"].std()},
        {"label": args.label, "chance": chance, "baseline": "random_forest_on_task_onehot",
         "pooled_balanced_acc": rf["balanced_acc"].mean(),
         "pooled_balanced_acc_std": rf["balanced_acc"].std(),
         "within_task_balanced_acc": rf["within_task_balanced_acc"].mean(),
         "within_task_balanced_acc_std": rf["within_task_balanced_acc"].std()},
    ])
    summary.to_csv(f"outputs/task_identity_baseline_summary{suffix}.csv", index=False)

    with open(f"outputs/task_identity_baseline_log{suffix}.txt", "w") as f:
        f.write(f"TASK-IDENTITY BASELINE: how well can you guess {args.label} "
                "using ONLY the task name (Baseline/Relax/Counting1/...), with "
                "zero physiological/audio/video signal at all?\n\n")
        f.write(f"chance level = {chance:.3f}\n")
        f.write(f"n={len(y)} rows, {groups.nunique()} subjects, {task.nunique()} tasks\n\n")
        f.write("1. Majority-vote-per-task (predict each task's most common label)\n")
        f.write(f"   pooled balanced_acc = {mv['balanced_acc'].mean():.3f} +/- {mv['balanced_acc'].std():.3f}\n")
        f.write(f"   per-fold: {mv['balanced_acc'].round(3).tolist()}\n\n")
        f.write("2. RandomForest trained only on one-hot task identity\n")
        f.write(f"   pooled balanced_acc = {rf['balanced_acc'].mean():.3f} +/- {rf['balanced_acc'].std():.3f}\n")
        f.write(f"   per-fold: {rf['balanced_acc'].round(3).tolist()}\n\n")
        f.write(f"For reference: chance level = {chance:.3f}.\n")

    print(f"\nSaved proof to: outputs/task_identity_baseline_summary{suffix}.csv")
    print(f"Saved proof to: outputs/task_identity_baseline_log{suffix}.txt")


if __name__ == "__main__":
    main()
