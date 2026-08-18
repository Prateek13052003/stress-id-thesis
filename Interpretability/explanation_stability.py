"""
Explanation Stability Score (ISS).

Question: is a model's per-modality attribution (which sensor "mattered
most") a reproducible property of the data, or an artifact of one particular
training run? Earlier in this project, Shapley utilization + head-ablation on
the StressID deep model gave a DIFFERENT "most important modality/head" every
time it was retrained with a new seed -- because that model had no real
signal to explain (Method/adapt_variant/interpretability_stability.py). This
script generalizes that check into a reusable score, and applies it to BOTH
StressID (no signal, established) and WESAD (real signal, established) for a
direct, controlled contrast.

Method: fix ONE subject-wise train/test split (--split-seed). Train the SAME
classifier (RandomForest) N times (--n-seeds) with different random seeds
(weight init / bootstrap randomness only -- the split never changes). For
each seed, compute EXACT Shapley per-modality utilization: with M modalities
there are 2^M coalitions, small enough to enumerate exactly. The value
function v(S) is the trained pipeline's predicted probability of the true
class when only the modalities in S are "visible" -- modalities NOT in S have
their columns replaced with the TRAINING-SET MEAN (the standard SHAP
baseline-value convention) before scaling, which the pipeline's own
StandardScaler then maps to ~0 (an uninformative input), before calling
predict_proba. No retraining per coalition -- one trained model per seed.

Three statistics across the N seeds:
  1. top1_agreement: fraction of seeds whose single highest-mean-utilization
     modality is the SAME modality. Chance baseline for M modalities = 1/M.
  2. mean_pairwise_rank_correlation: average Spearman correlation between
     every pair of seeds' modality-utilization ranking vectors. Range
     [-1, 1]; near 0 = no reproducible ranking, near 1 = fully reproducible.
  3. magnitude_validity: fraction of seeds where the top modality's Shapley
     contribution is significantly > 0 (one-sample t-test over the test
     samples, p<0.05). ADDED AFTER AN IMPORTANT FALSE START: with
     RandomForest, statistics 1 and 2 alone came out at a PERFECT 1.0 on
     BOTH StressID (no known signal) and WESAD (strong known signal) --
     RF is a low-variance bagged ensemble, so it ranks the same modality
     "top" every seed regardless of whether that modality's contribution is
     actually distinguishable from zero. Statistic 3 is what actually
     separated the two datasets (StressID's top-modality contribution was
     never significant; WESAD's always was) -- see the two per-dataset
     summary CSVs for the real numbers.

Final score: explanation_stability_score = rank_stability * magnitude_validity
(rank_stability itself is the average of top1_agreement, rescaled against its
chance baseline, and mean_pairwise_rank_correlation). Multiplying by
magnitude_validity means a model that is reproducibly stable but never
significant collapses toward 0, instead of the misleading 1.0 rank-stability
alone would have given.

Usage:
    python3 -m Interpretability.explanation_stability --dataset stressid --n-seeds 10
    python3 -m Interpretability.explanation_stability --dataset wesad --n-seeds 10
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs"


# ----------------------------------------------------------------------
# Dataset loaders -- each returns (X, y, groups, modality_column_groups)
# ----------------------------------------------------------------------

TALKING_TASKS = ["Counting1", "Counting2", "Counting3", "Math", "Reading", "Speaking", "Stroop"]


def load_stressid():
    feat_dir = ROOT / "Feature Extraction" / "Features"
    dataset_dir = ROOT / "Dataset"

    labels = pd.read_csv(dataset_dir / "labels.csv", index_col=0).dropna()
    task_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[1])
    labels = labels[task_all.isin(TALKING_TASKS)]
    y_all = labels["binary-stress"].astype(int)
    groups_all = labels.index.to_series().apply(lambda i: i.split("_", 1)[0])

    groups_cols = {}
    blocks = []
    for name in ["ecg", "eda", "resp"]:
        df = pd.read_csv(feat_dir / f"{name}_features.csv", index_col=0)
        df.columns = [f"{name}_{c}" for c in df.columns]
        groups_cols[name] = list(df.columns)
        blocks.append(df)
    X_all = pd.concat(blocks, axis=1)

    idx = X_all.index.intersection(y_all.index)
    X, y, groups = X_all.loc[idx], y_all.loc[idx], groups_all.loc[idx]
    return X, y, groups, groups_cols


def load_wesad():
    cache = ROOT / "WESAD" / "outputs" / "wesad_features_cache.csv"
    feat = pd.read_csv(cache, index_col=0).dropna(subset=["condition"])
    y = (feat["condition"].astype(int) == 2).astype(int)
    groups = feat.index.to_series().apply(lambda i: i.split("_")[0])

    n_ecg, n_resp, n_eda = 46, 62, 24
    cols = [c for c in feat.columns if c != "condition"]
    assert len(cols) == n_ecg + n_resp + n_eda
    groups_cols = {
        "ecg": cols[:n_ecg],
        "resp": cols[n_ecg:n_ecg + n_resp],
        "eda": cols[n_ecg + n_resp:n_ecg + n_resp + n_eda],
    }
    X = feat[cols]
    return X, y, groups, groups_cols


DATASETS = {"stressid": load_stressid, "wesad": load_wesad}


# ----------------------------------------------------------------------
# Exact Shapley with mean-baseline masking (no retraining per coalition)
# ----------------------------------------------------------------------

def make_pipeline(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(max_depth=5, random_state=seed)),
    ])


def true_class_prob(pipe, X, y, present_modalities, all_modalities, groups_cols, train_means):
    X_masked = X.copy()
    for m in all_modalities:
        if m not in present_modalities:
            X_masked[groups_cols[m]] = train_means[groups_cols[m]].values
    proba = pipe.predict_proba(X_masked)
    classes = list(pipe.named_steps["clf"].classes_)
    idx = [classes.index(v) for v in y]
    return proba[np.arange(len(y)), idx]


def exact_shapley_per_modality(pipe, X_test, y_test, groups_cols, train_means):
    """Returns dict[modality] -> per-TEST-SAMPLE array of exact Shapley
    contributions (not just the mean), so callers can test significance."""
    modalities = list(groups_cols.keys())
    n = len(modalities)
    v = {}
    for r in range(n + 1):
        for S in itertools.combinations(modalities, r):
            v[frozenset(S)] = true_class_prob(pipe, X_test, y_test, set(S), modalities, groups_cols, train_means)

    n_samples = len(y_test)
    shapley = {m: np.zeros(n_samples) for m in modalities}
    fact = math.factorial
    for i in modalities:
        others = [m for m in modalities if m != i]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                weight = fact(len(S)) * fact(n - len(S) - 1) / fact(n)
                shapley[i] += weight * (v[S | {i}] - v[S])
    return shapley


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--split-seed", type=int, default=1999)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    X, y, groups, groups_cols = DATASETS[args.dataset]()
    modalities = list(groups_cols.keys())
    print(f"Dataset={args.dataset}  n={len(y)}  subjects={groups.nunique()}  modalities={modalities}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.split_seed)
    train_pos, test_pos = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
    y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
    print(f"Fixed split: {len(set(groups.iloc[train_pos]))} train subjects, "
          f"{len(set(groups.iloc[test_pos]))} test subjects (split_seed={args.split_seed}, "
          f"unchanged across all {args.n_seeds} training seeds below)\n")

    train_means = X_train.mean()

    per_seed_utilization = []
    per_seed_top_pvalue = []
    for seed in range(args.n_seeds):
        pipe = make_pipeline(seed)
        pipe.fit(X_train, y_train)
        shapley_raw = exact_shapley_per_modality(pipe, X_test, y_test, groups_cols, train_means)
        util = {m: arr.mean() for m, arr in shapley_raw.items()}
        per_seed_utilization.append(util)

        top = max(util, key=util.get)
        # Magnitude significance: is the TOP modality's per-sample Shapley
        # distribution significantly greater than 0 (one-sample t-test over
        # the test-set samples)? Rank-stability alone doesn't check this --
        # a low-variance model (like RF) can rank the same modality "top"
        # every seed even when its contribution is ~0 or negative.
        t_stat, p_val = ttest_1samp(shapley_raw[top], 0.0, alternative="greater")
        per_seed_top_pvalue.append(p_val)
        sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
        print(f"  seed {seed:2d}: " + "  ".join(f"{m}={v:+.4f}" for m, v in util.items()) +
              f"   top={top}  p={p_val:.4f} [{sig}]")

    util_df = pd.DataFrame(per_seed_utilization)

    print(f"\n{'=' * 70}\nSTABILITY ACROSS {args.n_seeds} SEEDS (dataset={args.dataset})\n{'=' * 70}")
    print("\nMean +/- std utilization per modality across seeds:")
    for m in modalities:
        print(f"  {m:6s} mean={util_df[m].mean():+.4f}  std={util_df[m].std():.4f}")

    # --- statistic 1: top-1 agreement ---
    top_per_seed = util_df.idxmax(axis=1)
    top_counts = top_per_seed.value_counts()
    top1_agreement = top_counts.max() / args.n_seeds
    chance_top1 = 1 / len(modalities)
    print(f"\nTop-1 agreement: {top1_agreement:.2f}  (chance baseline for {len(modalities)} "
          f"modalities = {chance_top1:.2f})")
    print(f"  Top modality counts: {top_counts.to_dict()}")

    # --- statistic 2: mean pairwise Spearman rank correlation ---
    rank_matrix = util_df[modalities].rank(axis=1)
    corrs = []
    for i in range(args.n_seeds):
        for j in range(i + 1, args.n_seeds):
            rho, _ = spearmanr(rank_matrix.iloc[i], rank_matrix.iloc[j])
            corrs.append(rho)
    mean_rank_corr = float(np.nanmean(corrs))
    print(f"Mean pairwise rank correlation across seed-pairs: {mean_rank_corr:+.3f}  "
          f"(range [-1,1]; near 0 = no reproducible ranking)")

    rank_stability = 0.5 * ((top1_agreement - chance_top1) / (1 - chance_top1)) + 0.5 * max(mean_rank_corr, 0)

    # --- statistic 3: magnitude significance (the piece rank-stability alone misses) ---
    # A low-variance model (e.g. RandomForest) can rank the SAME modality
    # "top" every single seed even when its Shapley contribution is ~0 or
    # negative -- that happened for StressID below. Rank-stability alone
    # would then (wrongly) score this as "fully reproducible". This checks
    # whether the top modality's contribution is actually distinguishable
    # from zero, per seed.
    magnitude_validity = float(np.mean([p < 0.05 for p in per_seed_top_pvalue]))
    mean_top_utilization = float(np.mean([util_df.iloc[s][top_per_seed.iloc[s]] for s in range(args.n_seeds)]))
    print(f"\nMagnitude validity: {magnitude_validity:.2f}  (fraction of seeds where the top "
          f"modality's contribution is significantly > 0, one-sample t-test, p<0.05)")
    print(f"Mean utilization of the (per-seed) top modality: {mean_top_utilization:+.4f}")

    explanation_stability_score = rank_stability * magnitude_validity
    print(f"\nRank stability alone           = {rank_stability:.3f}")
    print(f"EXPLANATION STABILITY SCORE (ISS) = rank_stability x magnitude_validity "
          f"= {explanation_stability_score:.3f}")
    print("  (0 = not reproducible AND/OR not significantly different from zero; "
          "1 = same modality ranks top every seed AND its contribution is real)")

    util_df.to_csv(OUT_DIR / f"explanation_stability_{args.dataset}_per_seed.csv", index_label="seed")
    summary = pd.DataFrame([{
        "dataset": args.dataset, "n_seeds": args.n_seeds, "modalities": ",".join(modalities),
        "top1_agreement": top1_agreement, "chance_top1": chance_top1,
        "mean_pairwise_rank_corr": mean_rank_corr,
        "rank_stability": rank_stability,
        "magnitude_validity": magnitude_validity,
        "mean_top_utilization": mean_top_utilization,
        "explanation_stability_score": explanation_stability_score,
        **{f"{m}_mean_utilization": util_df[m].mean() for m in modalities},
        **{f"{m}_std_utilization": util_df[m].std() for m in modalities},
    }])
    summary.to_csv(OUT_DIR / f"explanation_stability_{args.dataset}_summary.csv", index=False)
    print(f"\nSaved: Interpretability/outputs/explanation_stability_{args.dataset}_summary.csv")
    print(f"Saved: Interpretability/outputs/explanation_stability_{args.dataset}_per_seed.csv")


if __name__ == "__main__":
    main()
