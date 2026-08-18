"""
01_audit_dataset.py
===================
StressID benchmark audit — Step 1.

Ye script kuch nahi train karta. Sirf DATA ka sach nikalta hai:

  1. Har feature file mein kitne rows / columns hain
  2. Kitne subjects, kitne tasks
  3. AVAILABILITY MATRIX  -> subject x task x modality (5 modalities)
  4. Paper ke claims verify karta hai (711 / 715 / 587 / 385 / 370)
  5. Labels ke saath merge karke batata hai kitne rows actually usable hain

Output:  outputs/ folder mein CSV files + terminal pe summary.

Chalane ka tareeka (Classification/ folder ke andar se):
    python 01_audit_dataset.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# CONFIG  --  agar paths alag hain toh sirf yahan badlo
# ----------------------------------------------------------------------

FEATURES_DIR = Path("../Feature Extraction/Features")
LABELS_PATH  = Path(r"/Users/prateekchoudhary/Downloads/transfer_12029461_files_10a77b58/StressID Dataset/labels.csv")
OUTPUT_DIR   = Path("outputs")

# 5 modalities.  Paper 3 maanta hai (physio / video / audio),
# hum physiology ko ECG + EDA + RESP mein todte hain.
MODALITY_FILES = {
    "ecg":   "ecg_features.csv",
    "eda":   "eda_features.csv",
    "resp":  "resp_features.csv",
    "video": "video11tasks_aus_gaze_mean_std.csv",
    "audio": "HCfeatures.csv",
}

# Ye files audit mein padhi jaati hain par modality nahi maani jaatin
EXTRA_FILES = {
    "physio_all":     "all_physiological_features.csv",
    "video_7tasks":   "video7tasks_aus_gaze_mean_std.csv",
    "video_morestat": "video11tasks_aus_gaze_morestats.csv",
    "audio_w2v":      "W2Vfeatures.csv",
}

# Paper (Chaptoukaev et al., NeurIPS 2023) ke claims
PAPER_CLAIMS = {
    "physiological recordings (Sec 3.2.2)": 711,
    "physiological tasks (Sec 4.1)":        715,
    "video recordings":                     587,
    "audio recordings":                     385,
    "tasks with all modalities":            370,
}


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def banner(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def clean_index(idx):
    """
    Index ko 'subject_task' form mein laata hai.

    Audio files ka index 'abc1_Speaking.wav' jaisa hota hai,
    isliye extension hata dete hain.  Whitespace bhi clean.
    """
    out = []
    for v in idx:
        s = str(v).strip()
        # sirf trailing file extension hatao, task ke andar ka dot nahi
        for ext in (".wav", ".csv", ".mp4", ".txt"):
            if s.lower().endswith(ext):
                s = s[: -len(ext)]
                break
        out.append(s)
    return out


def load_feature_file(path, name):
    """
    Feature CSV ko robustly load karta hai.

    HCfeatures.csv / W2Vfeatures.csv header ke bina save hue hain,
    isliye pehle header ke saath try karte hain aur agar pehla column
    numeric lage toh header=None se dobara padhte hain.
    """
    if not path.exists():
        print(f"  [MISSING] {name:16s} -> {path}")
        return None

    df = pd.read_csv(path, sep=",", header=0, index_col=0)

    # Header detect: agar column names mostly numeric strings hain
    # toh asli header nahi tha
    cols = [str(c) for c in df.columns]
    numeric_like = sum(c.replace(".", "", 1).replace("-", "", 1).isdigit()
                       for c in cols)
    if len(cols) > 0 and numeric_like / len(cols) > 0.8:
        df = pd.read_csv(path, sep=",", header=None, index_col=0)
        df.columns = [f"{name}_{i}" for i in range(df.shape[1])]

    df.index = clean_index(df.index)

    # duplicate index rows -> ye apne aap mein ek finding hai
    n_dup = df.index.duplicated().sum()
    if n_dup:
        print(f"  [WARN]    {name:16s} -> {n_dup} duplicate index rows")

    return df


def split_subject_task(index):
    """'2ea4_Counting1' -> ('2ea4', 'Counting1')"""
    subs, tasks = [], []
    for v in index:
        parts = str(v).split("_", 1)
        subs.append(parts[0])
        tasks.append(parts[1] if len(parts) > 1 else "UNKNOWN")
    return subs, tasks


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not FEATURES_DIR.exists():
        sys.exit(f"ERROR: Features folder nahi mila -> {FEATURES_DIR.resolve()}\n"
                 f"Script ko Classification/ folder ke andar se chalao, "
                 f"ya upar CONFIG mein path theek karo.")

    # ------------------------------------------------------------------
    # 1. Saari feature files load karo
    # ------------------------------------------------------------------
    banner("1. FEATURE FILES")

    frames = {}
    for name, fname in {**MODALITY_FILES, **EXTRA_FILES}.items():
        df = load_feature_file(FEATURES_DIR / fname, name)
        if df is not None:
            frames[name] = df
            subs, _ = split_subject_task(df.index)
            print(f"  {name:16s} {df.shape[0]:5d} rows x {df.shape[1]:5d} cols"
                  f"   |  {len(set(subs)):3d} subjects  |  {fname}")

    if not frames:
        sys.exit("ERROR: ek bhi feature file load nahi hui.")

    # ------------------------------------------------------------------
    # 2. Labels
    # ------------------------------------------------------------------
    banner("2. LABELS")

    if not LABELS_PATH.exists():
        sys.exit(f"ERROR: labels.csv nahi mila -> {LABELS_PATH.resolve()}\n"
                 f"CONFIG mein LABELS_PATH theek karo.")

    labels_raw = pd.read_csv(LABELS_PATH, sep=",", header=0, index_col=0)
    labels_raw.index = clean_index(labels_raw.index)
    labels = labels_raw.dropna()

    print(f"  labels.csv          : {len(labels_raw)} rows")
    print(f"  after .dropna()     : {len(labels)} rows"
          f"   ({len(labels_raw) - len(labels)} rows mein NaN tha)")
    print(f"  columns             : {list(labels.columns)}")

    for col in ("binary-stress", "affect3-class"):
        if col in labels.columns:
            vc = labels[col].value_counts().sort_index()
            dist = "  ".join(f"{k}={v}" for k, v in vc.items())
            print(f"  {col:20s}: {dist}")

    # ------------------------------------------------------------------
    # 3. Row-count mystery:  773 vs 711 vs 715
    # ------------------------------------------------------------------
    banner("3. ROW COUNT AUDIT  (paper ke claims vs asli data)")

    label_idx = set(labels.index)

    print(f"  {'file':18s} {'rows':>6s} {'labeled':>9s} {'unlabeled':>10s}")
    print("  " + "-" * 46)
    for name, df in frames.items():
        n_lab = len(set(df.index) & label_idx)
        print(f"  {name:18s} {len(df):6d} {n_lab:9d} {len(df) - n_lab:10d}")

    print()
    print("  Paper claims:")
    for k, v in PAPER_CLAIMS.items():
        print(f"    {k:42s} = {v}")

    # ------------------------------------------------------------------
    # 4. AVAILABILITY MATRIX
    # ------------------------------------------------------------------
    banner("4. AVAILABILITY MATRIX  (subject x task x 5 modalities)")

    # saare tasks ka union jo kisi bhi modality mein maujood hain
    all_tasks = set()
    for name in MODALITY_FILES:
        if name in frames:
            all_tasks |= set(frames[name].index)
    all_tasks = sorted(all_tasks)

    subs, tasks = split_subject_task(all_tasks)
    avail = pd.DataFrame({"subject": subs, "task": tasks}, index=all_tasks)

    for name in MODALITY_FILES:
        avail[name] = 0
        if name in frames:
            present = set(frames[name].index)
            avail[name] = [1 if t in present else 0 for t in all_tasks]

    mod_cols = list(MODALITY_FILES.keys())
    avail["n_modalities"] = avail[mod_cols].sum(axis=1)
    avail["has_label"] = [1 if t in label_idx else 0 for t in all_tasks]

    # pattern string, e.g. "11101"  -> combination analysis ke liye
    avail["pattern"] = avail[mod_cols].astype(str).agg("".join, axis=1)

    print(f"  total task rows : {len(avail)}")
    print(f"  subjects        : {avail['subject'].nunique()}")
    print(f"  unique tasks    : {avail['task'].nunique()}")
    print()

    print("  Per-modality availability:")
    for m in mod_cols:
        n = int(avail[m].sum())
        pct = 100.0 * n / len(avail)
        print(f"    {m:8s} {n:5d} / {len(avail)}  ({pct:5.1f}%)"
              f"   -> missing {100 - pct:5.1f}%")

    print()
    print("  Kitni modalities ek saath:")
    for k, v in avail["n_modalities"].value_counts().sort_index().items():
        print(f"    {k} modalities : {v:5d} tasks")

    n_complete = int((avail["n_modalities"] == len(mod_cols)).sum())
    n_complete_lab = int(((avail["n_modalities"] == len(mod_cols)) &
                          (avail["has_label"] == 1)).sum())
    print()
    print(f"  Saari 5 modalities present        : {n_complete}")
    print(f"  Saari 5 + label bhi present       : {n_complete_lab}")
    print(f"  (paper 3-modality complete = 370 bolta hai)")

    # ------------------------------------------------------------------
    # 5. Missingness structured hai ya random?
    # ------------------------------------------------------------------
    banner("5. MISSINGNESS PER TASK TYPE  (structured hai ya random?)")

    per_task = avail.groupby("task")[mod_cols].mean().round(3)
    per_task["n"] = avail.groupby("task").size()
    print(per_task.to_string())
    print()
    print("  Agar audio ka column 0 ya 1 pe clamp hai (beech mein nahi),")
    print("  toh missingness PROTOCOL-DRIVEN hai, random nahi.")
    print("  Yahi tumhari thesis ka Point 1 hai.")

    # ------------------------------------------------------------------
    # 6. Availability patterns
    # ------------------------------------------------------------------
    banner("6. TOP AVAILABILITY PATTERNS  (order: " + " ".join(mod_cols) + ")")

    pat = avail["pattern"].value_counts()
    for p, c in pat.head(15).items():
        active = [m for m, bit in zip(mod_cols, p) if bit == "1"]
        print(f"  {p}  n={c:5d}   {'+'.join(active) if active else '(none)'}")
    print()
    print(f"  Distinct patterns observed : {len(pat)}  (max possible = 31)")

    # ------------------------------------------------------------------
    # 7. Label imbalance per modality  -> shortcut ka khatra
    # ------------------------------------------------------------------
    if "binary-stress" in labels.columns:
        banner("7. SHORTCUT CHECK  (kya 'audio present' hi stress predict karta hai?)")

        lab_series = labels["binary-stress"]
        merged = avail.join(lab_series, how="inner")

        print(f"  labeled tasks : {len(merged)}")
        print()
        print(f"  {'modality':10s} {'present: stress%':>18s} {'absent: stress%':>18s}")
        print("  " + "-" * 48)
        for m in mod_cols:
            grp = merged.groupby(m)["binary-stress"].mean()
            p = 100 * grp.get(1, np.nan)
            a = 100 * grp.get(0, np.nan)
            print(f"  {m:10s} {p:17.1f}% {a:17.1f}%")
        print()
        print("  Bada gap = model sirf 'file hai ya nahi' dekh ke cheat kar sakta hai.")
        print("  Ye Method 2 ka shortcut-detection experiment hai.")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    banner("8. FILES SAVED")

    avail.to_csv(OUTPUT_DIR / "availability_matrix.csv")
    per_task.to_csv(OUTPUT_DIR / "missingness_per_task.csv")
    pat.to_frame("count").to_csv(OUTPUT_DIR / "availability_patterns.csv")

    summary = pd.DataFrame([
        {"file": n, "rows": len(d), "cols": d.shape[1],
         "subjects": len(set(split_subject_task(d.index)[0])),
         "labeled_rows": len(set(d.index) & label_idx)}
        for n, d in frames.items()
    ])
    summary.to_csv(OUTPUT_DIR / "file_summary.csv", index=False)

    for f in ["availability_matrix.csv", "missingness_per_task.csv",
              "availability_patterns.csv", "file_summary.csv"]:
        print(f"  {OUTPUT_DIR / f}")

    print()
    print("Audit complete.")


if __name__ == "__main__":
    main()
