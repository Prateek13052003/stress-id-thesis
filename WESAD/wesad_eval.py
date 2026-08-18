"""
Positive control for the StressID investigation: WESAD (Schmidt et al., 2018)
is a well-established stress dataset where subject-wise-validated real signal
is already known to exist in the literature (commonly ~80-95% accuracy for
stress-vs-non-stress with LOSO/subject-wise validation).

This script:
  1. Loads each subject's WESAD.pkl (chest ECG/EDA/Resp at 700Hz + labels).
  2. Cuts non-overlapping 60s windows within each labeled condition
     (1=baseline, 2=stress, 3=amusement; other label values are transition/
     ignored segments per the WESAD readme).
  3. Extracts handcrafted features using the SAME feature-extraction code
     used for StressID (Feature Extraction/physiological/*.py) -- copied
     into this folder and patched for compatibility with the current
     scipy/pandas versions (scipy.integrate.trapz was removed;
     DataFrame.set_axis(inplace=True) was removed). Originals untouched.
  4. Builds the standard binary task: stress (2) vs non-stress (1+3).
  5. Evaluates with RandomForest under LEAKY (plain KFold) and HONEST
     (subject-wise GroupKFold) splits, exactly mirroring
     Classification/dataset_audit.py's methodology on StressID.

If HONEST balanced_acc is well above 0.5 here, it validates that this
project's evaluation methodology correctly detects real signal when it is
genuinely present -- which is what gives the StressID "no signal detected"
finding its force.

Usage:
    python3 -m WESAD.wesad_eval
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecg_features import get_ecg_features
from eda_features import get_eda_features
from respiration_features import get_resp_features

WESAD_DIR = Path.home() / "Downloads" / "WESAD_download" / "extracted" / "WESAD"
FS = 700  # chest sampling rate, Hz
WINDOW_SEC = 60
KEEP_LABELS = {1: "baseline", 2: "stress", 3: "amusement"}
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_subject(sid):
    path = WESAD_DIR / sid / f"{sid}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    chest = d["signal"]["chest"]
    label = np.asarray(d["label"])
    return {
        "ecg": np.asarray(chest["ECG"]).reshape(-1),
        "eda": np.asarray(chest["EDA"]).reshape(-1),
        "resp": np.asarray(chest["Resp"]).reshape(-1),
        "label": label,
    }


def window_subject(sid, sig):
    """Cuts non-overlapping WINDOW_SEC windows within each labeled condition
    run. Returns list of (window_id, condition_label, ecg, eda, resp)."""
    win_len = WINDOW_SEC * FS
    label = sig["label"]
    windows = []
    n = len(label)
    i = 0
    while i < n:
        cur = label[i]
        j = i
        while j < n and label[j] == cur:
            j += 1
        # [i, j) is one contiguous run of the same label
        if cur in KEEP_LABELS:
            start = i
            w = 0
            while start + win_len <= j:
                wid = f"{sid}_{KEEP_LABELS[cur]}{w}"
                windows.append((
                    wid, cur,
                    sig["ecg"][start:start + win_len],
                    sig["eda"][start:start + win_len],
                    sig["resp"][start:start + win_len],
                ))
                start += win_len
                w += 1
        i = j
    return windows


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def make_pipeline(seed=0):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(max_depth=5, random_state=seed)),
    ])


def eval_splits(X, y, groups, group_aware, n_splits=10, seed=0):
    scores = []
    if group_aware:
        split_iter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)
    for train_idx, test_idx in split_iter:
        pipe = make_pipeline(seed)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        train_subj = set(np.array(groups)[train_idx])
        test_subj = set(np.array(groups)[test_idx])
        overlap = len(train_subj & test_subj) / len(test_subj)
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1": f1_score(y.iloc[test_idx], pred, average="weighted"),
            "subject_overlap": overlap,
        })
    return pd.DataFrame(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None, help="Optional path to cache extracted features CSV")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    cache_path = Path(args.cache) if args.cache else OUT_DIR / "wesad_features_cache.csv"

    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        feat = pd.read_csv(cache_path, index_col=0)
    else:
        subjects = sorted(
            [p.name for p in WESAD_DIR.iterdir() if p.is_dir() and p.name.startswith("S")],
            key=lambda s: int(s[1:]),
        )
        print(f"Found {len(subjects)} subjects: {subjects}")

        all_windows = []
        for sid in subjects:
            print(f"  loading {sid}...")
            sig = load_subject(sid)
            wins = window_subject(sid, sig)
            print(f"    {len(wins)} windows of {WINDOW_SEC}s "
                  f"({sum(1 for w in wins if w[1] == 2)} stress, "
                  f"{sum(1 for w in wins if w[1] == 1)} baseline, "
                  f"{sum(1 for w in wins if w[1] == 3)} amusement)")
            all_windows.extend(wins)

        print(f"\nTotal windows: {len(all_windows)}")
        print("Extracting features (ECG/EDA/Resp, same code as StressID)...")

        ecg_dict, eda_dict, rsp_dict, cond = {}, {}, {}, {}
        for wid, label, ecg, eda, rsp in all_windows:
            ecg_dict[wid] = nk.ecg_clean(zscore(ecg), sampling_rate=FS, method="biosppy")
            eda_dict[wid] = nk.eda_clean(zscore(eda), sampling_rate=FS, method="biosppy")
            rsp_dict[wid] = nk.rsp_clean(zscore(rsp), sampling_rate=FS, method="biosppy")
            cond[wid] = label

        df_ecg = get_ecg_features(ecg_dict, FS)
        print(f"  ECG features: {df_ecg.shape}")
        df_eda = get_eda_features(eda_dict, FS)
        print(f"  EDA features: {df_eda.shape}")
        df_rsp = get_resp_features(rsp_dict, FS)
        print(f"  Resp features: {df_rsp.shape}")

        feat = pd.concat([df_ecg, df_rsp], axis=1).merge(df_eda, left_index=True, right_index=True)
        feat["condition"] = feat.index.map(cond)
        feat.to_csv(cache_path)
        print(f"Cached features to {cache_path}")

    feat = feat.dropna(subset=["condition"])
    y = (feat["condition"].astype(int) == 2).astype(int)  # stress vs non-stress
    X = feat.drop(columns=["condition"])
    groups = feat.index.to_series().apply(lambda i: i.split("_")[0])

    print(f"\nn={len(y)} windows, {groups.nunique()} subjects")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}  (1=stress)")

    print("\n" + "=" * 70)
    print("LEAKY split (plain KFold, no subject grouping)")
    print("=" * 70)
    leaky = eval_splits(X, y, groups, group_aware=False, n_splits=args.n_splits, seed=args.seed)
    print(f"balanced_acc = {leaky['balanced_acc'].mean():.4f} +/- {leaky['balanced_acc'].std():.4f}")
    print(f"f1           = {leaky['f1'].mean():.4f} +/- {leaky['f1'].std():.4f}")
    print(f"avg subject overlap: {leaky['subject_overlap'].mean():.1%}")

    print("\n" + "=" * 70)
    print("HONEST split (GroupKFold, subject-wise)")
    print("=" * 70)
    honest = eval_splits(X, y, groups, group_aware=True, n_splits=args.n_splits, seed=args.seed)
    print(f"balanced_acc = {honest['balanced_acc'].mean():.4f} +/- {honest['balanced_acc'].std():.4f}")
    print(f"f1           = {honest['f1'].mean():.4f} +/- {honest['f1'].std():.4f}")
    print(f"avg subject overlap: {honest['subject_overlap'].mean():.1%}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Leakage-alone inflation: {leaky['balanced_acc'].mean() - honest['balanced_acc'].mean():+.4f}")
    print(f"HONEST delta vs chance (0.5): {honest['balanced_acc'].mean() - 0.5:+.4f}")

    summary = pd.DataFrame([
        {"condition": "leaky_kfold", "n": len(y),
         "balanced_acc": leaky["balanced_acc"].mean(), "balanced_acc_std": leaky["balanced_acc"].std()},
        {"condition": "honest_groupkfold", "n": len(y),
         "balanced_acc": honest["balanced_acc"].mean(), "balanced_acc_std": honest["balanced_acc"].std()},
    ])
    summary.to_csv(OUT_DIR / "wesad_eval_summary.csv", index=False)
    print(f"\nSaved: outputs/wesad_eval_summary.csv (also under WESAD/outputs/)")


if __name__ == "__main__":
    main()
