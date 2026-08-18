"""
On the talking-tasks-only subset (task-identity confound neutralized, see
talking_subset_eval.py: majority-vote/RF-on-task both = exactly 0.500 here),
compares LEAKY task-wise splits (matching the StressID paper's own protocol,
Section 4: "10 random splits, using 80% of the tasks for training, and 20%
for testing") against HONEST subject-wise splits, same features, same
classifier, same n_splits — to isolate how much of the gap between the
literature's reported numbers (StressID ~0.64-0.65 BAcc, ADAPT 0.695 on this
same talking-only 370-task population) and this project's subject-wise
finding (~0.50, chance) is explained by subject-level leakage alone.

Usage:
    python3 -m Classification.talking_subset_leaky_test
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATASET_DIR = "Dataset"
FEAT_DIR = "Feature Extraction/Features"
TALKING_TASKS = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]


def eval_splits(X, y, groups, leaky, n_splits=10, test_size=0.2, seed=0):
    scores = []
    if leaky:
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        split_iter = splitter.split(X, y, groups)

    for train_idx, test_idx in split_iter:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(max_depth=5, random_state=seed)),
        ])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        train_subj = set(groups.iloc[train_idx])
        test_subj = set(groups.iloc[test_idx])
        overlap = len(train_subj & test_subj) / len(test_subj)
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1": f1_score(y.iloc[test_idx], pred, average="weighted"),
            "subject_overlap": overlap,
        })
    return pd.DataFrame(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="binary-stress", choices=["binary-stress", "affect3-class"])
    args = ap.parse_args()

    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0).dropna()
    task_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[1])
    labels = labels[task_all.isin(TALKING_TASKS)]
    groups_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[0])
    y_all = labels[args.label].astype(int)

    X = pd.read_csv(f"{FEAT_DIR}/all_physiological_features.csv", index_col=0)
    idx = X.index.intersection(y_all.index)
    X, y, groups = X.loc[idx], y_all.loc[idx], groups_all.loc[idx]

    print(f"Talking-tasks-only subset, physiological features: n={len(y)}, "
          f"{groups.nunique()} subjects")
    print(f"StressID paper protocol match: 10 splits, 80% train / 20% test\n")

    print("=" * 70)
    print("LEAKY split (ShuffleSplit, matches paper's described protocol: "
          "plain random 80/20 over tasks, no subject grouping)")
    print("=" * 70)
    leaky = eval_splits(X, y, groups, leaky=True, n_splits=10, test_size=0.2, seed=0)
    print(f"balanced_acc = {leaky['balanced_acc'].mean():.4f} +/- {leaky['balanced_acc'].std():.4f}")
    print(f"f1           = {leaky['f1'].mean():.4f} +/- {leaky['f1'].std():.4f}")
    print(f"avg subject overlap between train/test: {leaky['subject_overlap'].mean():.1%}")
    print(f"per-split balanced_acc: {leaky['balanced_acc'].round(4).tolist()}")
    print()

    print("=" * 70)
    print("HONEST split (GroupShuffleSplit, subject-wise, same n_splits/test_size)")
    print("=" * 70)
    honest = eval_splits(X, y, groups, leaky=False, n_splits=10, test_size=0.2, seed=0)
    print(f"balanced_acc = {honest['balanced_acc'].mean():.4f} +/- {honest['balanced_acc'].std():.4f}")
    print(f"f1           = {honest['f1'].mean():.4f} +/- {honest['f1'].std():.4f}")
    print(f"avg subject overlap between train/test: {honest['subject_overlap'].mean():.1%}")
    print(f"per-split balanced_acc: {honest['balanced_acc'].round(4).tolist()}")
    print()

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    gap = leaky["balanced_acc"].mean() - honest["balanced_acc"].mean()
    print(f"Leakage-alone inflation on this talking-only, confound-neutralized subset: {gap:+.4f}")
    if args.label == "binary-stress":
        print(f"For reference: StressID paper reports ~0.62-0.65 BAcc on this same 370-task "
              f"population (their Table 3, physio-only row: 0.58; best fusion row: 0.65) "
              f"using the SAME leaky protocol as our 'LEAKY' condition above.")
        print(f"ADAPT reports 0.695 BAcc on this same population (their Table 1, StressID column).")

    import os
    os.makedirs("outputs", exist_ok=True)
    suffix = "" if args.label == "binary-stress" else f"_{args.label}"
    out = pd.DataFrame([
        {"label": args.label, "split": "leaky_task_wise", "n": len(y),
         "balanced_acc": leaky["balanced_acc"].mean(), "balanced_acc_std": leaky["balanced_acc"].std(),
         "avg_subject_overlap": leaky["subject_overlap"].mean()},
        {"label": args.label, "split": "honest_subject_wise", "n": len(y),
         "balanced_acc": honest["balanced_acc"].mean(), "balanced_acc_std": honest["balanced_acc"].std(),
         "avg_subject_overlap": honest["subject_overlap"].mean()},
    ])
    out.to_csv(f"outputs/talking_subset_leaky_test_summary{suffix}.csv", index=False)
    print(f"\nSaved proof to: outputs/talking_subset_leaky_test_summary{suffix}.csv")


if __name__ == "__main__":
    main()
