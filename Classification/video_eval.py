"""
Same rigor as w2v_audio_eval.py and talking_subset_leaky_test.py, applied to
video (84-dim OpenFace Action Units + gaze features, mean+std over each task).

Video is different from audio in one important way: it covers ALL 11 tasks
(603 rows, 55 subjects), not just the 7 talking ones. So two tests:

  1. Talking-tasks-only subset (comparable to the physio/audio results already
     gathered): honest subject-wise split vs leaky task-wise split.
  2. Full 11-task population: honest split, reporting BOTH pooled
     balanced_acc (which the task-identity confound inflates, per
     Classification/task_identity_baseline.py) AND within-task balanced_acc
     (Classification/metrics_utils.py) which neutralizes that confound. This
     is the test audio could never run, since audio doesn't exist outside
     the talking tasks.

Usage:
    python3 -m Classification.video_eval
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, ShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from Classification.metrics_utils import within_task_balanced_accuracy

DATASET_DIR = "Dataset"
FEAT_DIR = "Feature Extraction/Features"
TALKING_TASKS = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]


def make_pipeline(seed=0):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(max_depth=5, random_state=seed)),
    ])


def eval_shuffle_splits(X, y, groups, leaky, n_splits=10, test_size=0.2, seed=0, task=None):
    scores = []
    if leaky:
        split_iter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed).split(X, y)
    else:
        split_iter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed).split(X, y, groups)

    for train_idx, test_idx in split_iter:
        pipe = make_pipeline(seed)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        overlap = len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) / len(set(groups.iloc[test_idx]))
        row = {
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1": f1_score(y.iloc[test_idx], pred, average="weighted"),
            "subject_overlap": overlap,
        }
        if task is not None:
            wt_bacc, _ = within_task_balanced_accuracy(y.iloc[test_idx], pred, task.iloc[test_idx])
            row["within_task_balanced_acc"] = wt_bacc
        scores.append(row)
    return pd.DataFrame(scores)


def eval_groupkfold(X, y, groups, task, n_splits=5, seed=0):
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pipe = make_pipeline(seed)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        wt_bacc, _ = within_task_balanced_accuracy(y.iloc[test_idx], pred, task.iloc[test_idx])
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1": f1_score(y.iloc[test_idx], pred, average="weighted"),
            "within_task_balanced_acc": wt_bacc,
        })
    return pd.DataFrame(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="binary-stress", choices=["binary-stress", "affect3-class"])
    args = ap.parse_args()

    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0).dropna()
    y_all = labels[args.label].astype(int)
    groups_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[0])
    task_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[1])

    video = pd.read_csv(f"{FEAT_DIR}/video11tasks_aus_gaze_mean_std.csv", index_col=0)

    print(f"Video rows available: {len(video)}\n")

    # --- Test 1: talking-tasks-only subset, comparable to physio/audio ---
    idx_talk = video.index.intersection(y_all[task_all.isin(TALKING_TASKS)].index)
    y_t, groups_t = y_all.loc[idx_talk], groups_all.loc[idx_talk]
    print(f"{'=' * 70}\n1a. Video, talking-tasks-only subset (n={len(idx_talk)}, "
          f"{groups_t.nunique()} subjects), HONEST split\n{'=' * 70}")
    honest_t = eval_shuffle_splits(video.loc[idx_talk], y_t, groups_t, leaky=False)
    print(f"balanced_acc = {honest_t['balanced_acc'].mean():.4f} +/- {honest_t['balanced_acc'].std():.4f}")
    print(f"per-split: {honest_t['balanced_acc'].round(4).tolist()}")

    print(f"\n{'=' * 70}\n1b. Video, talking-tasks-only subset, LEAKY split\n{'=' * 70}")
    leaky_t = eval_shuffle_splits(video.loc[idx_talk], y_t, groups_t, leaky=True)
    print(f"balanced_acc = {leaky_t['balanced_acc'].mean():.4f} +/- {leaky_t['balanced_acc'].std():.4f}")
    print(f"avg subject overlap: {leaky_t['subject_overlap'].mean():.1%}")
    print(f"per-split: {leaky_t['balanced_acc'].round(4).tolist()}")
    print(f"\nLeakage-alone inflation: {leaky_t['balanced_acc'].mean() - honest_t['balanced_acc'].mean():+.4f}")

    # --- Test 2: full 11-task population, pooled vs within-task ---
    idx_full = video.index.intersection(y_all.index)
    y_f, groups_f, task_f = y_all.loc[idx_full], groups_all.loc[idx_full], task_all.loc[idx_full]
    print(f"\n{'=' * 70}\n2. Video, FULL 11-task population (n={len(idx_full)}, "
          f"{groups_f.nunique()} subjects), HONEST GroupKFold, pooled vs within-task\n{'=' * 70}")
    full_honest = eval_groupkfold(video.loc[idx_full], y_f, groups_f, task_f, n_splits=5)
    print(f"POOLED balanced_acc      = {full_honest['balanced_acc'].mean():.4f} "
          f"+/- {full_honest['balanced_acc'].std():.4f}")
    print(f"WITHIN-TASK balanced_acc = {full_honest['within_task_balanced_acc'].mean():.4f} "
          f"+/- {full_honest['within_task_balanced_acc'].std():.4f}")
    print(f"(gap between pooled and within-task = how much of the pooled score is "
          f"task-identity, not real signal: "
          f"{full_honest['balanced_acc'].mean() - full_honest['within_task_balanced_acc'].mean():+.4f})")

    import os
    os.makedirs("outputs", exist_ok=True)
    suffix = "" if args.label == "binary-stress" else f"_{args.label}"
    out = pd.DataFrame([
        {"label": args.label, "test": "talking_subset_honest", "n": len(idx_talk),
         "balanced_acc": honest_t["balanced_acc"].mean(), "balanced_acc_std": honest_t["balanced_acc"].std()},
        {"label": args.label, "test": "talking_subset_leaky", "n": len(idx_talk),
         "balanced_acc": leaky_t["balanced_acc"].mean(), "balanced_acc_std": leaky_t["balanced_acc"].std()},
        {"label": args.label, "test": "full_11task_pooled", "n": len(idx_full),
         "balanced_acc": full_honest["balanced_acc"].mean(), "balanced_acc_std": full_honest["balanced_acc"].std()},
        {"label": args.label, "test": "full_11task_within_task", "n": len(idx_full),
         "balanced_acc": full_honest["within_task_balanced_acc"].mean(),
         "balanced_acc_std": full_honest["within_task_balanced_acc"].std()},
    ])
    out.to_csv(f"outputs/video_eval_summary{suffix}.csv", index=False)
    print(f"\nSaved proof to: outputs/video_eval_summary{suffix}.csv")


if __name__ == "__main__":
    main()
