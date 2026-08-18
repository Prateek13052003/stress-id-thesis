"""
Pulls together every honest-vs-leaky / chance-level result from this whole
investigation into ONE table. Reads the already-saved output files from:
  - dataset_audit.py, task_identity_baseline.py
  - talking_subset_eval.py, talking_subset_leaky_test.py
  - w2v_audio_eval.py, video_eval.py
  - Method/adapt_variant/train_talking_subset.py (ecg + resp, honest + leaky)

Does NOT re-run any experiment -- purely aggregates already-computed,
already-verified numbers into one file for easy presentation. Works for
either label (--label binary-stress / affect3-class); run the underlying
scripts with the matching --label first.

Usage (run from repo root):
    python3 -m Classification.master_summary --label affect3-class
"""

import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="binary-stress", choices=["binary-stress", "affect3-class"])
    args = ap.parse_args()
    label = args.label
    chance = 1 / 3 if label == "affect3-class" else 0.5
    suffix = "" if label == "binary-stress" else f"_{label}"

    rows = []

    def add(modality, feature_type, condition, n, bacc, std, source_file):
        rows.append({
            "label": label, "chance": round(chance, 4),
            "modality": modality, "feature_type": feature_type, "condition": condition,
            "n": n, "balanced_acc": round(bacc, 4), "balanced_acc_std": round(std, 4),
            "source_file": source_file,
        })

    # --- task-identity-only baselines (full 11-task pool) ---
    f = f"outputs/task_identity_baseline_summary{suffix}.csv"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add("task-name-only (no signal)", r["baseline"], "pooled, honest split", None,
            r["pooled_balanced_acc"], r["pooled_balanced_acc_std"], f)
        add("task-name-only (no signal)", r["baseline"], "WITHIN-TASK, honest split (empirical floor)", None,
            r["within_task_balanced_acc"], r["within_task_balanced_acc_std"], f)

    # --- physio, full pool, leaky vs honest (dataset_audit.py) ---
    f = f"outputs/dataset_audit_summary{suffix}.csv"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add("physiological (ECG+EDA+RESP)", "handcrafted, full 11-task pool", r["condition"], r["n"],
            r["balanced_acc"], r["balanced_acc_std"], f)

    # --- talking-subset physio: majority-vote / task-onehot / real features ---
    f = f"outputs/talking_subset_eval_summary{suffix}.csv"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add("physiological (talking-subset)", r["test"], "honest, confound-controlled", r["n"],
            r["balanced_acc"], r["balanced_acc_std"], f)

    # --- talking-subset physio: leaky vs honest ---
    f = f"outputs/talking_subset_leaky_test_summary{suffix}.csv"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add("physiological (talking-subset)", "real features", r["split"], r["n"],
            r["balanced_acc"], r["balanced_acc_std"], f)

    # --- audio: W2V vs HC, honest and leaky ---
    f = f"outputs/w2v_audio_eval_summary{suffix}.csv"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add("audio (talking-subset)", r["test"], "see test name", r["n"],
            r["balanced_acc"], r["balanced_acc_std"], f)

    # --- video: talking-subset + full-task pooled/within-task ---
    f = f"outputs/video_eval_summary{suffix}.csv"
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        add("video (AU + gaze)", r["test"], "see test name", r["n"],
            r["balanced_acc"], r["balanced_acc_std"], f)

    # --- deep model (Method/adapt_variant), talking subset, honest + leaky ---
    deep_files = {
        ("deep model, ECG-anchor", "honest"):
            f"Method/adapt_variant/outputs/results_talking-subset_{label}_anchor-ecg.json",
        ("deep model, ECG-anchor", "leaky"):
            f"Method/adapt_variant/outputs/results_talking-subset_{label}_anchor-ecg-leaky.json",
        ("deep model, RESP-anchor", "honest"):
            f"Method/adapt_variant/outputs/results_talking-subset_{label}_anchor-resp.json",
    }
    for (modality, condition), path in deep_files.items():
        if not os.path.exists(path):
            continue
        with open(path) as file:
            d = json.load(file)
        s = d["summary"]["full"]
        add(modality, "5-modality fusion", condition, d.get("n_rows"),
            s["balanced_acc_mean"], s["balanced_acc_std"], path)

    # --- deep model, full 11-task pool (Method/adapt_variant/train.py) ---
    path = f"Method/adapt_variant/outputs/results_{label}_anchor-ecg.json"
    if os.path.exists(path):
        with open(path) as file:
            d = file and json.load(file)
        s = d["summary"]["full"]
        add("deep model, ECG-anchor", "5-modality fusion", "full 11-task pool, pooled", None,
            s["balanced_acc_mean"], s["balanced_acc_std"], path)
        if "within_task_balanced_acc_mean" in s:
            add("deep model, ECG-anchor", "5-modality fusion", "full 11-task pool, WITHIN-TASK", None,
                s["within_task_balanced_acc_mean"], s["within_task_balanced_acc_std"], path)

    out = pd.DataFrame(rows)
    out_path = f"outputs/MASTER_all_results_summary{suffix}.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}  ({len(out)} rows)")
    print(f"\nChance level for comparison = {chance:.3f}")


if __name__ == "__main__":
    main()
