"""
Tests whether the task-identity confound (task_identity_baseline.py) is
specific to POOLING all 11 tasks together, or present everywhere.

All 7 "talking" tasks (Counting1-3, Math, Reading, Speaking, Stroop — the
same 7 tasks StressID has audio for) have binary-stress majority class = 1
(range 54.7%-76.6% stress, see crosstab). A per-task-majority-vote baseline
therefore degenerates to a CONSTANT "always predict stressed" classifier on
this subset, which is mathematically guaranteed exactly balanced_acc=0.5,
regardless of prevalence. Unlike the full 11-task pool (where Relax/Breathing
pull one way and the talking tasks pull the other, letting task-identity
alone reach ~0.70), pooled evaluation restricted to this subset should NOT
have a free task-identity ceiling — so a model actually beating 0.5 here,
at n=448 (much better powered than the n~12-13 within-task cells), would be
real evidence of physiological signal.

This subset is also the one ADAPT calls X*_test — StressID's audio is only
recorded for these 7 tasks, so their reported 69.5 BAcc is presumably
computed on a talking-tasks-dominated population already (worth confirming
against their paper, not assumed).

Usage:
    python3 -m Classification.talking_subset_eval
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Classification.metrics_utils import within_task_balanced_accuracy

DATASET_DIR = "Dataset"
FEAT_DIR = "Feature Extraction/Features"
TALKING_TASKS = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]


def majority_vote_baseline(task, y, groups, n_splits=5, seed=0):
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(task, y, groups):
        train_task, train_y = task.iloc[train_idx], y.iloc[train_idx]
        majority = train_y.groupby(train_task).agg(lambda s: s.mode().iloc[0])
        overall_majority = train_y.mode().iloc[0]
        preds = task.iloc[test_idx].map(majority).fillna(overall_majority).astype(int)
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], preds),
            "f1": f1_score(y.iloc[test_idx], preds, average="weighted"),
        })
    return pd.DataFrame(scores)


def rf_on_task_onehot(task, y, groups, n_splits=5, seed=0):
    X = pd.get_dummies(task)
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = RandomForestClassifier(max_depth=5, random_state=seed)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], preds),
            "f1": f1_score(y.iloc[test_idx], preds, average="weighted"),
        })
    return pd.DataFrame(scores)


def rf_on_physio(X_physio, y, groups, n_splits=5, seed=0):
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X_physio, y, groups):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(max_depth=5, random_state=seed)),
        ])
        pipe.fit(X_physio.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X_physio.iloc[test_idx])
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], preds),
            "f1": f1_score(y.iloc[test_idx], preds, average="weighted"),
        })
    return pd.DataFrame(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="binary-stress", choices=["binary-stress", "affect3-class"])
    args = ap.parse_args()
    chance = 1 / 3 if args.label == "affect3-class" else 0.5

    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0).dropna()
    task_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[1])
    is_talking = task_all.isin(TALKING_TASKS)

    labels = labels[is_talking]
    task = task_all[is_talking]
    groups = labels.index.to_series().apply(lambda i: i.split("_", 1)[0])
    y = labels[args.label].astype(int)

    print(f"LABEL = {args.label}  (chance level ~ {chance:.3f})")
    print(f"n={len(y)} rows, {groups.nunique()} subjects, {task.nunique()} tasks "
          f"(talking-only subset)")
    print(f"Overall class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print()

    print("=" * 70)
    print("1. Majority-vote-per-task lookup (zero physiological signal) -- "
          "for binary-stress this collapses to exactly 0.5 since every talking "
          "task's majority is 'stressed'; for affect3-class this is not "
          "guaranteed, the printed number is the real empirical floor")
    print("=" * 70)
    mv = majority_vote_baseline(task, y, groups)
    print(f"POOLED balanced_acc = {mv['balanced_acc'].mean():.4f} +/- {mv['balanced_acc'].std():.4f}")
    print(f"per-fold: {mv['balanced_acc'].round(4).tolist()}")
    print()

    print("=" * 70)
    print("2. RandomForest on one-hot task identity only")
    print("=" * 70)
    rf_task = rf_on_task_onehot(task, y, groups)
    print(f"POOLED balanced_acc = {rf_task['balanced_acc'].mean():.4f} +/- {rf_task['balanced_acc'].std():.4f}")
    print(f"per-fold: {rf_task['balanced_acc'].round(4).tolist()}")
    print()

    print("=" * 70)
    print("3. RandomForest on all_physiological_features.csv (real signal, subject-wise GroupKFold)")
    print("=" * 70)
    X_physio = pd.read_csv(f"{FEAT_DIR}/all_physiological_features.csv", index_col=0)
    idx = X_physio.index.intersection(y.index)
    X_physio, y_physio, groups_physio = X_physio.loc[idx], y.loc[idx], groups.loc[idx]
    print(f"(after intersecting with available physio features: n={len(idx)})")
    rf_physio = rf_on_physio(X_physio, y_physio, groups_physio)
    print(f"POOLED balanced_acc = {rf_physio['balanced_acc'].mean():.4f} +/- {rf_physio['balanced_acc'].std():.4f}")
    print(f"f1                   = {rf_physio['f1'].mean():.4f} +/- {rf_physio['f1'].std():.4f}")
    print(f"per-fold: {rf_physio['balanced_acc'].round(4).tolist()}")
    print()
    print(f"Delta vs task-identity ceiling (majority-vote): "
          f"{rf_physio['balanced_acc'].mean() - mv['balanced_acc'].mean():+.4f}")
    print(f"Delta vs theoretical chance ({chance:.3f}): {rf_physio['balanced_acc'].mean() - chance:+.4f}")

    import os
    os.makedirs("outputs", exist_ok=True)
    suffix = "" if args.label == "binary-stress" else f"_{args.label}"
    out = pd.DataFrame([
        {"label": args.label, "chance": chance, "test": "majority_vote_per_task", "n": len(y),
         "balanced_acc": mv["balanced_acc"].mean(), "balanced_acc_std": mv["balanced_acc"].std()},
        {"label": args.label, "chance": chance, "test": "rf_on_task_identity_only", "n": len(y),
         "balanced_acc": rf_task["balanced_acc"].mean(), "balanced_acc_std": rf_task["balanced_acc"].std()},
        {"label": args.label, "chance": chance, "test": "rf_on_real_physio_features", "n": len(idx),
         "balanced_acc": rf_physio["balanced_acc"].mean(), "balanced_acc_std": rf_physio["balanced_acc"].std()},
    ])
    out.to_csv(f"outputs/talking_subset_eval_summary{suffix}.csv", index=False)
    print(f"\nSaved proof to: outputs/talking_subset_eval_summary{suffix}.csv")


if __name__ == "__main__":
    main()
