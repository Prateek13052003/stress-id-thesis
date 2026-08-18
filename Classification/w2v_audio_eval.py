"""
Tests whether richer audio features (Wav2Vec 2.0 embeddings, pretrained,
512-dim mean + 512-dim std pooled over time = 1024-dim) reveal real
physiological/paralinguistic stress signal that the handcrafted audio
features (140-dim, HCfeatures.csv) and physiological signals could not, once
leakage and the task-identity confound are controlled for.

Audio in StressID exists ONLY for the 7 talking tasks (378 rows, 54
subjects) — so this population is automatically the same
confound-neutralized "talking-only" domain as talking_subset_eval.py
(majority-vote-per-task = exactly 0.5 here too, same reasoning).

Three conditions, all on the exact same row subset for a fair comparison:
  1. W2V features, HONEST (subject-wise GroupKFold) split — the real test.
  2. HC audio features, HONEST split — same population, weaker features,
     for a direct "does richer embedding help" comparison.
  3. W2V features, LEAKY split (matches StressID paper's own protocol,
     Table 2 reports F1=0.70/Acc=0.66 for their "W2V 2.0 classifier" row)
     — a sanity check that this implementation is comparable to theirs
     before trusting the honest-split number.

Usage:
    python3 -m Classification.w2v_audio_eval
"""

import argparse
from ast import literal_eval

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


def load_w2v():
    df = pd.read_csv(f"{FEAT_DIR}/W2Vfeatures.csv", header=None, index_col=0)
    df.index = [i.split(".")[0] for i in df.index]
    mean_vecs = np.stack(df[1].apply(literal_eval).values)
    std_vecs = np.stack(df[2].apply(literal_eval).values)
    combined = np.concatenate([mean_vecs, std_vecs], axis=1)
    cols = [f"w2v_mean_{i}" for i in range(512)] + [f"w2v_std_{i}" for i in range(512)]
    return pd.DataFrame(combined, index=df.index, columns=cols)


def load_hc_audio():
    df = pd.read_csv(f"{FEAT_DIR}/HCfeatures.csv", header=None, index_col=0)
    df.index = [i.split(".")[0] for i in df.index]
    return df


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
        overlap = len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) / len(set(groups.iloc[test_idx]))
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
    y_all = labels[args.label].astype(int)
    groups_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[0])

    w2v = load_w2v()
    hc_audio = load_hc_audio()

    idx = w2v.index.intersection(y_all.index)
    print(f"W2V audio rows available: {len(w2v)}, after intersecting with labels: {len(idx)}")
    y = y_all.loc[idx]
    groups = groups_all.loc[idx]
    task = idx.to_series().apply(lambda i: i.split("_", 1)[1])
    print(f"n={len(idx)} rows, {groups.nunique()} subjects, tasks={sorted(task.unique())}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")
    print()

    print("=" * 70)
    print("1. W2V features (1024-dim), HONEST subject-wise split")
    print("=" * 70)
    w2v_honest = eval_splits(w2v.loc[idx], y, groups, leaky=False)
    print(f"balanced_acc = {w2v_honest['balanced_acc'].mean():.4f} +/- {w2v_honest['balanced_acc'].std():.4f}")
    print(f"f1           = {w2v_honest['f1'].mean():.4f} +/- {w2v_honest['f1'].std():.4f}")
    print(f"avg subject overlap: {w2v_honest['subject_overlap'].mean():.1%}")
    print(f"per-split: {w2v_honest['balanced_acc'].round(4).tolist()}")
    print()

    print("=" * 70)
    print("2. HC audio features (140-dim), HONEST split, SAME rows")
    print("=" * 70)
    hc_idx = hc_audio.index.intersection(idx)
    hc_honest = eval_splits(hc_audio.loc[hc_idx], y.loc[hc_idx], groups.loc[hc_idx], leaky=False)
    print(f"balanced_acc = {hc_honest['balanced_acc'].mean():.4f} +/- {hc_honest['balanced_acc'].std():.4f}")
    print(f"f1           = {hc_honest['f1'].mean():.4f} +/- {hc_honest['f1'].std():.4f}")
    print(f"per-split: {hc_honest['balanced_acc'].round(4).tolist()}")
    print()

    chance = 1 / 3 if args.label == "affect3-class" else 0.5
    print("=" * 70)
    if args.label == "binary-stress":
        print("3. W2V features, LEAKY split (sanity check vs StressID paper's own "
              "W2V baseline: Table 2 reports F1=0.70, Acc(balanced)=0.66)")
    else:
        print("3. W2V features, LEAKY split")
    print("=" * 70)
    w2v_leaky = eval_splits(w2v.loc[idx], y, groups, leaky=True)
    print(f"balanced_acc = {w2v_leaky['balanced_acc'].mean():.4f} +/- {w2v_leaky['balanced_acc'].std():.4f}")
    print(f"f1           = {w2v_leaky['f1'].mean():.4f} +/- {w2v_leaky['f1'].std():.4f}")
    print(f"avg subject overlap: {w2v_leaky['subject_overlap'].mean():.1%}")
    print(f"per-split: {w2v_leaky['balanced_acc'].round(4).tolist()}")
    print()

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"W2V honest vs chance ({chance:.3f}): {w2v_honest['balanced_acc'].mean() - chance:+.4f}")
    print(f"W2V honest vs HC honest:  {w2v_honest['balanced_acc'].mean() - hc_honest['balanced_acc'].mean():+.4f}")
    print(f"W2V leaky vs honest:      {w2v_leaky['balanced_acc'].mean() - w2v_honest['balanced_acc'].mean():+.4f}"
          f"  (leakage-alone inflation)")

    import os
    os.makedirs("outputs", exist_ok=True)
    suffix = "" if args.label == "binary-stress" else f"_{args.label}"
    out = pd.DataFrame([
        {"label": args.label, "chance": chance, "test": "w2v_honest_subject_wise", "n": len(idx),
         "balanced_acc": w2v_honest["balanced_acc"].mean(), "balanced_acc_std": w2v_honest["balanced_acc"].std()},
        {"label": args.label, "chance": chance, "test": "hc_audio_honest_subject_wise", "n": len(hc_idx),
         "balanced_acc": hc_honest["balanced_acc"].mean(), "balanced_acc_std": hc_honest["balanced_acc"].std()},
        {"label": args.label, "chance": chance, "test": "w2v_leaky_task_wise", "n": len(idx),
         "balanced_acc": w2v_leaky["balanced_acc"].mean(), "balanced_acc_std": w2v_leaky["balanced_acc"].std()},
    ])
    out.to_csv(f"outputs/w2v_audio_eval_summary{suffix}.csv", index=False)
    print(f"\nSaved proof to: outputs/w2v_audio_eval_summary{suffix}.csv")


if __name__ == "__main__":
    main()
