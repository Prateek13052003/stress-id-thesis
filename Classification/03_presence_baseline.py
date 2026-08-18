"""
03_presence_baseline.py
=======================
StressID benchmark audit -- Step 3.  CONTROL BASELINES.

Sawaal:  StressID pe reported performance ka kitna hissa asli
         physiological / behavioural signal se aata hai, aur kitna
         sirf METADATA se?

Audit (script 01) ne dikhaya:
    audio present -> 69.6% stress      audio absent -> 32.6% stress
    Counting2     -> 76.6% stress      Relax        -> 12.9% stress

Yaani ek model jo asli data dekhe hi na -- sirf ye jaane ki
"kaunsi file maujood hai" ya "ye kaunsa task hai" -- wo bhi
achha score la sakta hai.

Ye script paanch baselines chalata hai, sab SUBJECT-WISE splits pe:

    1. majority        -> hamesha most-common class            (floor)
    2. presence        -> sirf 5 bits: ecg/eda/resp/video/audio
    3. task_identity   -> sirf 11-dim one-hot: kaunsa task hai
    4. presence+task   -> dono metadata saath
    5. real_features   -> asli physiological features           (ceiling)

Agar (2) ya (3) ka score (5) ke kareeb aa jaaye, toh benchmark
mostly metadata measure kar raha hai, stress nahi.

PEHLE `01_audit_dataset.py` chalao -- ye uski
outputs/availability_matrix.csv use karta hai.

Chalane ka tareeka (Classification/ folder se):
    python3 -u 03_presence_baseline.py 2>&1 | tee outputs/presence_log.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from make_classification_fixed import run_classification, subjects_from_index


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

FEATURES_DIR = Path("../Feature Extraction/Features")
LABELS_PATH  = Path(r"/Users/prateekchoudhary/Downloads/transfer_12029461_files_10a77b58/StressID Dataset/labels.csv")
OUTPUT_DIR   = Path("outputs")
AVAIL_PATH   = OUTPUT_DIR / "availability_matrix.csv"

REAL_FEATURES_FILE = "all_physiological_features.csv"

LABEL_COLUMN = "binary-stress"
MODALITY_COLS = ["ecg", "eda", "resp", "video", "audio"]

N_SPLITS = 10
SEED = 0


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


def load_real_features(path, name):
    if not path.exists():
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


def evaluate(name, X, y, classifiers, note=""):
    """Ek baseline ko subject-wise splits pe chalao."""
    print(f"\n{'-' * 68}")
    print(f"BASELINE: {name}   ({X.shape[1]} features, {len(X)} tasks)")
    if note:
        print(f"  {note}")
    print(f"{'-' * 68}")

    res, _, info = run_classification(
        X, y,
        mode="groupshuffle",
        n_splits=N_SPLITS,
        seed=SEED,
        resample=False,
        list_classifiers=classifiers,
        impute=False,          # metadata mein NaN nahi hota
        scale=False,
        verbose=False,
    )

    assert info["overlapping_subjects"].sum() == 0, "subject leak detected!"

    rows = []
    for clf, grp in res.groupby("classifier"):
        f1, ba = grp["f1-score"], grp["accuracy"]
        print(f"  {clf:24s} F1 {f1.mean():.3f} (+/-{f1.std():.3f})   "
              f"BAcc {ba.mean():.3f} (+/-{ba.std():.3f})")
        rows.append({"baseline": name, "classifier": clf,
                     "f1": round(f1.mean(), 3), "f1_std": round(f1.std(), 3),
                     "bacc": round(ba.mean(), 3), "bacc_std": round(ba.std(), 3),
                     "n_features": X.shape[1], "n_tasks": len(X)})
    return rows


# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not AVAIL_PATH.exists():
        sys.exit(f"ERROR: {AVAIL_PATH} nahi mila.\n"
                 f"Pehle chalao:  python3 01_audit_dataset.py")

    avail = pd.read_csv(AVAIL_PATH, index_col=0)
    labels = pd.read_csv(LABELS_PATH, sep=",", header=0, index_col=0).dropna()
    labels.index = clean_index(labels.index)
    y_all = labels[LABEL_COLUMN]

    common = avail.index.intersection(y_all.index)
    avail = avail.loc[common]
    y = y_all.loc[common].astype(int)

    print("=" * 68)
    print(f"CONTROL BASELINES   label='{LABEL_COLUMN}'   "
          f"n_splits={N_SPLITS}   subject-wise splits")
    print("=" * 68)
    print(f"  labeled tasks : {len(y)}")
    print(f"  subjects      : {len(set(subjects_from_index(y.index)))}")
    print(f"  class balance : {dict(y.value_counts().sort_index())}   "
          f"(majority = {y.value_counts(normalize=True).max():.3f})")

    # ------------------------------------------------------------------
    # feature blocks
    # ------------------------------------------------------------------
    X_presence = avail[MODALITY_COLS].astype(float)

    n_pat = X_presence.astype(int).astype(str).agg("".join, axis=1).nunique()
    print(f"  distinct presence patterns among labeled tasks : {n_pat}")

    X_task = pd.get_dummies(avail["task"], prefix="task").astype(float)
    X_task.index = avail.index
    print(f"  distinct tasks : {X_task.shape[1]}")

    X_both = pd.concat([X_presence, X_task], axis=1)

    # ------------------------------------------------------------------
    # run baselines
    # ------------------------------------------------------------------
    all_rows = []

    all_rows += evaluate(
        "1_majority",
        X_presence, y,
        [DummyClassifier(strategy="most_frequent")],
        note="floor -- kuch nahi seekhta, hamesha majority class",
    )

    all_rows += evaluate(
        "2_presence_only",
        X_presence, y,
        [LogisticRegression(max_iter=2000),
         RandomForestClassifier(max_depth=5, random_state=0)],
        note="input = sirf 5 bits (kaunsi modality maujood hai)",
    )

    all_rows += evaluate(
        "3_task_identity",
        X_task, y,
        [LogisticRegression(max_iter=2000),
         RandomForestClassifier(max_depth=5, random_state=0)],
        note="input = sirf one-hot task label, koi signal nahi",
    )

    all_rows += evaluate(
        "4_presence_plus_task",
        X_both, y,
        [LogisticRegression(max_iter=2000),
         RandomForestClassifier(max_depth=5, random_state=0)],
        note="dono metadata saath",
    )

    # real features -- ceiling
    Xr = load_real_features(FEATURES_DIR / REAL_FEATURES_FILE, "phys")
    if Xr is not None:
        idx = Xr.index.intersection(y.index)
        all_rows += evaluate(
            "5_real_features",
            Xr.loc[idx], y.loc[idx],
            [LogisticRegression(max_iter=2000),
             RandomForestClassifier(max_depth=5, random_state=0)],
            note=f"ceiling -- asli physiological signal ({REAL_FEATURES_FILE})",
        )
        # note: yahan impute/scale off hai taaki comparison uniform rahe;
        # script 02 ne inhi features pe impute+scale ke saath 0.680 diya tha
    else:
        print(f"\n  [SKIP] real features nahi mile: {REAL_FEATURES_FILE}")

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    summary = pd.DataFrame(all_rows)

    print("\n" + "=" * 68)
    print("SUMMARY -- sab subject-wise splits pe")
    print("=" * 68)
    print(summary.to_string(index=False))

    best = summary.groupby("baseline")[["f1", "bacc"]].max()
    print("\n" + "-" * 68)
    print("Har baseline ka BEST score:")
    print(best.to_string())

    if "5_real_features" in best.index:
        real_f1 = best.loc["5_real_features", "f1"]
        real_ba = best.loc["5_real_features", "bacc"]
        meta_f1 = best.loc[["2_presence_only", "3_task_identity",
                            "4_presence_plus_task"], "f1"].max()
        meta_ba = best.loc[["2_presence_only", "3_task_identity",
                            "4_presence_plus_task"], "bacc"].max()

        print("\n" + "-" * 68)
        print("METADATA vs REAL SIGNAL")
        print(f"  best metadata-only   : F1 {meta_f1:.3f}   BAcc {meta_ba:.3f}")
        print(f"  real features        : F1 {real_f1:.3f}   BAcc {real_ba:.3f}")
        print(f"  asli signal ka faayda: F1 {real_f1 - meta_f1:+.3f}   "
              f"BAcc {real_ba - meta_ba:+.3f}")
        print("-" * 68)
        gap = real_f1 - meta_f1
        if gap < 0:
            print("  => METADATA ASLI FEATURES SE BEHTAR HAI.")
            print("     Is benchmark pe reported performance ka bada hissa")
            print("     stress signal se nahi, dataset ki structure se aata hai.")
        elif gap < 0.05:
            print("  => Gap chhota hai. Benchmark ka bada hissa metadata")
            print("     se explain ho jaata hai, stress signal se nahi.")
        else:
            print("  => Asli features metadata se saaf upar hain.")

    summary.to_csv(OUTPUT_DIR / "presence_baseline_summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'presence_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
