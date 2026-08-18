"""
Quantifies two problems in the existing benchmark (see CLAUDE.md):

1. Task-wise leakage: classification_physio.ipynb and
   classification_multimodal_binary_stress.ipynb call make_nclassif_random_splits*,
   which does a plain train_test_split over subject_task rows. Since each subject
   contributes up to 12 rows (one per task), the same subject's physiology ends up
   in both train and test, inflating scores. make_classification.py already has
   make_nclassif (GroupKFold by subject) sitting unused.

2. Missing-modality data loss: the multimodal notebook inner-joins modalities,
   silently dropping any subject_task missing one modality (see
   build_multimodal_dataset.py output: only 52.9% of rows survive).

This script re-runs the SAME physiological features + same classifier
(RandomForest) under both split strategies and reports the score gap directly,
plus modality-missingness stats. Writes results to outputs/audit_log.txt.
"""

import argparse
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Classification.metrics_utils import within_task_balanced_accuracy

FEAT_DIR = "Feature Extraction/Features"
DATASET_DIR = "Dataset"


def make_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(max_depth=5, random_state=0)),
    ])


def eval_splits(X, y, groups, group_aware, n_splits=10, seed=0, task=None):
    scores = []
    if group_aware:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)

    for train_idx, test_idx in split_iter:
        pipe = make_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        row = {
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1": f1_score(y.iloc[test_idx], pred, average="weighted"),
        }
        if task is not None:
            wt_bacc, _ = within_task_balanced_accuracy(y.iloc[test_idx], pred, task.iloc[test_idx])
            row["within_task_balanced_acc"] = wt_bacc
        scores.append(row)
    return pd.DataFrame(scores)


def subject_overlap_check(X, groups, seed=0):
    """Directly demonstrates the leakage: how many test-fold subjects also
    appear in that fold's training set, under plain KFold vs GroupKFold."""
    groups = pd.Series(groups, index=X.index)

    kfold_overlaps = []
    for train_idx, test_idx in KFold(n_splits=10, shuffle=True, random_state=seed).split(X):
        train_subj = set(groups.iloc[train_idx])
        test_subj = set(groups.iloc[test_idx])
        kfold_overlaps.append(len(train_subj & test_subj) / len(test_subj))

    group_overlaps = []
    for train_idx, test_idx in GroupKFold(n_splits=10).split(X, groups=groups):
        train_subj = set(groups.iloc[train_idx])
        test_subj = set(groups.iloc[test_idx])
        group_overlaps.append(len(train_subj & test_subj) / len(test_subj))

    return sum(kfold_overlaps) / len(kfold_overlaps), sum(group_overlaps) / len(group_overlaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="binary-stress", choices=["binary-stress", "affect3-class"])
    args = ap.parse_args()
    chance = 1 / 3 if args.label == "affect3-class" else 0.5

    log = []

    def w(line=""):
        print(line)
        log.append(line)

    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0).dropna()
    X = pd.read_csv(f"{FEAT_DIR}/all_physiological_features.csv", index_col=0)
    idx = X.index.intersection(labels.index)
    X, y = X.loc[idx], labels.loc[idx, args.label]
    task = X.index.to_series().apply(lambda i: i.split("_", 1)[1])

    w(f"LABEL = {args.label}  (chance level = {chance:.3f})")
    w("")

    w("=" * 70)
    w("1. LEAKAGE CHECK: fraction of test-fold subjects also seen in training")
    w("=" * 70)
    groups = [i.split("_")[0] for i in X.index]
    kfold_overlap, group_overlap = subject_overlap_check(X, groups)
    w(f"Plain KFold (what the notebooks currently use):  {kfold_overlap:.1%} of each "
      f"test fold's subjects were ALSO in that fold's training set")
    w(f"GroupKFold (subject-wise, correct):               {group_overlap:.1%}")
    w("")

    w("=" * 70)
    w("2. SCORE INFLATION: same features (all_physiological_features.csv), "
      "same RF(max_depth=5), 10 splits")
    w("=" * 70)
    leaky = eval_splits(X, y, groups, group_aware=False)
    honest = eval_splits(X, y, groups, group_aware=True, task=task)
    w(f"Plain KFold      balanced_acc = {leaky['balanced_acc'].mean():.3f} "
      f"(+/- {leaky['balanced_acc'].std():.3f}),  f1 = {leaky['f1'].mean():.3f}")
    w(f"GroupKFold       balanced_acc = {honest['balanced_acc'].mean():.3f} "
      f"(+/- {honest['balanced_acc'].std():.3f}),  f1 = {honest['f1'].mean():.3f}")
    w(f"Leakage inflation on balanced_acc: "
      f"{leaky['balanced_acc'].mean() - honest['balanced_acc'].mean():+.3f}")
    w("")

    w("=" * 70)
    w("2b. TASK-IDENTITY CONFOUND CHECK (see Classification/task_identity_baseline.py): "
      "knowing ONLY the task name can predict the label with zero physiological signal, "
      "because the label distribution differs by task (protocol design). WITHIN-TASK "
      f"balanced_acc neutralizes this: it is exactly {chance:.3f} (chance) for "
      "task-identity-only baselines, so any score above that here is real "
      "physiological signal, not task lookup.")
    w("=" * 70)
    w(f"GroupKFold (subject-wise, honest split), all_physiological_features.csv, RF:")
    w(f"  pooled balanced_acc      = {honest['balanced_acc'].mean():.3f} "
      f"(+/- {honest['balanced_acc'].std():.3f})")
    w(f"  WITHIN-TASK balanced_acc = {honest['within_task_balanced_acc'].mean():.3f} "
      f"(+/- {honest['within_task_balanced_acc'].std():.3f})   [chance = {chance:.3f}]")
    w("  (task-identity-only baselines score exactly chance-level on this metric "
      "— see Classification/task_identity_baseline.py)")
    w("")

    w("=" * 70)
    w("3. MISSING-MODALITY DATA LOSS (from build_multimodal_dataset.py)")
    w("=" * 70)
    try:
        mask = pd.read_csv(f"{FEAT_DIR}/multimodal_5modality_mask.csv", index_col=0)
        w("Per-modality availability:")
        w(mask.mean().round(3).to_string())
        n_complete = mask.all(axis=1).sum()
        w(f"Rows with all 5 modalities: {n_complete}/{len(mask)} ({n_complete / len(mask):.1%})")
        w("-> classification_multimodal_binary_stress.ipynb's inner-join keeps only "
          "this subset; the rest is silently discarded, not treated as a "
          "missing-modality case.")
    except FileNotFoundError:
        w("Run Feature Extraction/build_multimodal_dataset.py first for this section.")

    suffix = "" if args.label == "binary-stress" else f"_{args.label}"
    with open(f"outputs/audit_log{suffix}.txt", "w") as f:
        f.write("\n".join(log) + "\n")

    summary = pd.DataFrame([
        {"label": args.label, "chance": chance, "condition": "leaky_plain_kfold", "n": len(y),
         "balanced_acc": leaky["balanced_acc"].mean(), "balanced_acc_std": leaky["balanced_acc"].std()},
        {"label": args.label, "chance": chance, "condition": "honest_groupkfold", "n": len(y),
         "balanced_acc": honest["balanced_acc"].mean(), "balanced_acc_std": honest["balanced_acc"].std()},
        {"label": args.label, "chance": chance, "condition": "honest_groupkfold_within_task", "n": len(y),
         "balanced_acc": honest["within_task_balanced_acc"].mean(),
         "balanced_acc_std": honest["within_task_balanced_acc"].std()},
    ])
    summary.to_csv(f"outputs/dataset_audit_summary{suffix}.csv", index=False)
    print(f"\nSaved: outputs/audit_log{suffix}.txt, outputs/dataset_audit_summary{suffix}.csv")


if __name__ == "__main__":
    main()
