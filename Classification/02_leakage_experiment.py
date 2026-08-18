"""
02_leakage_experiment.py
========================
StressID benchmark audit -- Step 2.  THESIS KA HEADLINE EXPERIMENT.

Sawaal:  StressID ke published baselines subject leakage se kitne inflate
         hue hain?

Method:  Bilkul same features, same classifiers, same metrics.
         SIRF split strategy badalti hai:

             random        -> jo notebooks abhi use karte hain (leaky)
             groupshuffle  -> subject-wise, koi subject dono taraf nahi

         Gap = leakage ka size.

Modalities:  ecg, eda, resp, video, audio  (5 -- paper ke 3 nahi)
             plus 'physio_all' taaki paper se seedha compare ho sake.

Output:  outputs/leakage_*.csv  aur terminal pe ek clean table.

Chalane ka tareeka (Classification/ folder ke andar se):
    python 02_leakage_experiment.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from make_classification_fixed import (
    run_classification, subjects_from_index,
)


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

FEATURES_DIR = Path("../Feature Extraction/Features")
LABELS_PATH  = Path(r"/Users/prateekchoudhary/Downloads/transfer_12029461_files_10a77b58/StressID Dataset/labels.csv")
OUTPUT_DIR   = Path("outputs")

LABEL_COLUMN = "binary-stress"      # ya "affect3-class"
N_SPLITS     = 10
SEED         = 0

MODALITIES = {
    "ecg":        "ecg_features.csv",
    "eda":        "eda_features.csv",
    "resp":       "resp_features.csv",
    "physio_all": "all_physiological_features.csv",
    "video":      "video11tasks_aus_gaze_mean_std.csv",
    "audio":      "HCfeatures.csv",
}

# Notebooks jo classifiers use karte hain -- wahi rakhe hain
CLASSIFIERS = [
    RandomForestClassifier(max_depth=5, random_state=0),
    KNeighborsClassifier(n_neighbors=3),
    SVC(gamma="auto", kernel="rbf", random_state=0),
    MLPClassifier(max_iter=5000, random_state=0, hidden_layer_sizes=[]),
]


# ----------------------------------------------------------------------

def clean_index(idx):
    out = []
    for v in idx:
        s = str(v).strip()
        for ext in (".wav", ".csv", ".mp4", ".txt"):
            if s.lower().endswith(ext):
                s = s[: -len(ext)]
                break
        out.append(s)
    return out


def load_features(path, name):
    if not path.exists():
        print(f"  [SKIP] {name} -> file nahi mili: {path}")
        return None

    df = pd.read_csv(path, sep=",", header=0, index_col=0)
    cols = [str(c) for c in df.columns]
    numeric_like = sum(c.replace(".", "", 1).replace("-", "", 1).isdigit()
                       for c in cols)
    if len(cols) and numeric_like / len(cols) > 0.8:
        df = pd.read_csv(path, sep=",", header=None, index_col=0)
        df.columns = [f"{name}_{i}" for i in range(df.shape[1])]

    df.index = clean_index(df.index)
    df = df[~df.index.duplicated(keep="first")]
    return df.apply(pd.to_numeric, errors="coerce")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not LABELS_PATH.exists():
        sys.exit(f"ERROR: labels.csv nahi mila -> {LABELS_PATH.resolve()}")

    labels = pd.read_csv(LABELS_PATH, sep=",", header=0, index_col=0).dropna()
    labels.index = clean_index(labels.index)

    if LABEL_COLUMN not in labels.columns:
        sys.exit(f"ERROR: '{LABEL_COLUMN}' labels mein nahi. "
                 f"Available: {list(labels.columns)}")

    print("=" * 72)
    print(f"LEAKAGE EXPERIMENT   label='{LABEL_COLUMN}'   "
          f"n_splits={N_SPLITS}   seed={SEED}")
    print("=" * 72)

    all_results, summary_rows = [], []

    for name, fname in MODALITIES.items():
        X = load_features(FEATURES_DIR / fname, name)
        if X is None:
            continue

        common = X.index.intersection(labels.index)
        if len(common) < 30:
            print(f"  [SKIP] {name}: sirf {len(common)} labeled rows")
            continue

        Xm = X.loc[common]
        ym = labels.loc[common, LABEL_COLUMN]
        n_subj = len(set(subjects_from_index(Xm.index)))

        print(f"\n{'=' * 72}")
        print(f"MODALITY: {name}   "
              f"{Xm.shape[0]} tasks x {Xm.shape[1]} features   "
              f"{n_subj} subjects")
        print(f"  class balance: "
              f"{dict(ym.value_counts().sort_index())}")
        print("=" * 72)

        per_mode = {}
        for mode in ("random", "groupshuffle"):
            print(f"\n  [{mode}]")
            res, _, info = run_classification(
                Xm, ym,
                mode=mode,
                n_splits=N_SPLITS,
                seed=SEED,
                resample=False,
                list_classifiers=CLASSIFIERS,
                verbose=False,
            )
            res["modality"] = name
            all_results.append(res)
            per_mode[mode] = res

            n_leaky = int(info["leaky"].sum())
            print(f"    splits with subject overlap: "
                  f"{n_leaky}/{len(info)}")
            print(f"    mean overlapping subjects  : "
                  f"{info['overlapping_subjects'].mean():.1f}")

            for clf, grp in res.groupby("classifier"):
                print(f"      {clf:24s} "
                      f"F1 {grp['f1-score'].mean():.3f} "
                      f"(+/-{grp['f1-score'].std():.3f})   "
                      f"BAcc {grp['accuracy'].mean():.3f} "
                      f"(+/-{grp['accuracy'].std():.3f})")

        # gap
        for clf in per_mode["random"]["classifier"].unique():
            r = per_mode["random"]
            g = per_mode["groupshuffle"]
            r_f1 = r.loc[r["classifier"] == clf, "f1-score"].mean()
            g_f1 = g.loc[g["classifier"] == clf, "f1-score"].mean()
            r_ac = r.loc[r["classifier"] == clf, "accuracy"].mean()
            g_ac = g.loc[g["classifier"] == clf, "accuracy"].mean()
            summary_rows.append({
                "modality": name,
                "classifier": clf,
                "f1_leaky": round(r_f1, 3),
                "f1_clean": round(g_f1, 3),
                "f1_drop": round(r_f1 - g_f1, 3),
                "acc_leaky": round(r_ac, 3),
                "acc_clean": round(g_ac, 3),
                "acc_drop": round(r_ac - g_ac, 3),
            })

    if not summary_rows:
        sys.exit("\nKoi modality process nahi hui. Paths check karo.")

    results = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 72)
    print("SUMMARY -- leaky (random split) vs clean (subject-wise split)")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\n" + "-" * 72)
    print("Average inflation across all modalities and classifiers:")
    print(f"  F1        : {summary['f1_drop'].mean():+.3f}")
    print(f"  Bal. Acc  : {summary['acc_drop'].mean():+.3f}")
    print("-" * 72)
    print("Positive drop = published numbers inflated the.")

    results.to_csv(OUTPUT_DIR / "leakage_raw_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "leakage_summary.csv", index=False)
    print(f"\nSaved:")
    print(f"  {OUTPUT_DIR / 'leakage_raw_results.csv'}")
    print(f"  {OUTPUT_DIR / 'leakage_summary.csv'}")


if __name__ == "__main__":
    main()
