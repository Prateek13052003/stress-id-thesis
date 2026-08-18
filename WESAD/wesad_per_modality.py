"""
Breaks the WESAD positive-control result (wesad_eval.py) down per modality
(ECG-only, EDA-only, Resp-only, and all-combined), same honest-vs-leaky
methodology, reusing the already-extracted feature cache (no re-extraction).

Column groups are positional, matching the exact concat order used in
wesad_eval.py: pd.concat([df_ecg(46 cols), df_rsp(62 cols)], axis=1)
    .merge(df_eda(24 cols)) -> ['condition'] last.

Usage:
    python3 -m WESAD.wesad_per_modality
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CACHE = OUT_DIR / "wesad_features_cache.csv"

N_ECG, N_RESP, N_EDA = 46, 62, 24


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
        scores.append({
            "balanced_acc": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1": f1_score(y.iloc[test_idx], pred, average="weighted"),
        })
    return pd.DataFrame(scores)


def main():
    feat = pd.read_csv(CACHE, index_col=0).dropna(subset=["condition"])
    y = (feat["condition"].astype(int) == 2).astype(int)
    groups = feat.index.to_series().apply(lambda i: i.split("_")[0])

    feature_cols = [c for c in feat.columns if c != "condition"]
    assert len(feature_cols) == N_ECG + N_RESP + N_EDA, \
        f"expected {N_ECG + N_RESP + N_EDA} feature cols, got {len(feature_cols)}"

    modalities = {
        "ecg": feature_cols[:N_ECG],
        "resp": feature_cols[N_ECG:N_ECG + N_RESP],
        "eda": feature_cols[N_ECG + N_RESP:N_ECG + N_RESP + N_EDA],
        "all_combined": feature_cols,
    }

    print(f"n={len(y)} windows, {groups.nunique()} subjects, "
          f"class balance: {y.value_counts(normalize=True).round(3).to_dict()} (1=stress)\n")

    rows = []
    header = f"{'modality':14s} {'n_features':>10s} {'leaky_bacc':>14s} {'honest_bacc':>14s} {'leak_inflation':>15s} {'vs_chance':>10s}"
    print(header)
    print("-" * len(header))
    for name, cols in modalities.items():
        X = feat[cols]
        leaky = eval_splits(X, y, groups, group_aware=False)
        honest = eval_splits(X, y, groups, group_aware=True)
        l_mean, l_std = leaky["balanced_acc"].mean(), leaky["balanced_acc"].std()
        h_mean, h_std = honest["balanced_acc"].mean(), honest["balanced_acc"].std()
        print(f"{name:14s} {len(cols):10d} {l_mean:.3f}+-{l_std:.3f}  {h_mean:.3f}+-{h_std:.3f}  "
              f"{l_mean - h_mean:+14.3f} {h_mean - 0.5:+9.3f}")
        rows.append({
            "modality": name, "n_features": len(cols), "n": len(y),
            "leaky_balanced_acc": l_mean, "leaky_balanced_acc_std": l_std,
            "honest_balanced_acc": h_mean, "honest_balanced_acc_std": h_std,
            "leak_inflation": l_mean - h_mean, "honest_vs_chance": h_mean - 0.5,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "wesad_per_modality_summary.csv", index=False)
    print(f"\nSaved: WESAD/outputs/wesad_per_modality_summary.csv")


if __name__ == "__main__":
    main()
