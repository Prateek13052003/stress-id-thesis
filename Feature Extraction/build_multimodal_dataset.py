"""
Assemble the real 5-modality StressID feature table: ECG, EDA, RESP, Video, Audio.

Unlike Classification/classification_multimodal_binary_stress.ipynb, this does NOT
inner-join modalities (which silently drops every subject_task missing any one
modality). Instead it outer-joins on the full label index and keeps missing cells
as NaN, plus writes a boolean availability mask per modality. This mask is the
ground-truth missingness pattern needed for missing-modality benchmarking.

Outputs (into Feature Extraction/Features/):
  - multimodal_5modality_features.csv   feature matrix, NaN where a modality is absent
  - multimodal_5modality_mask.csv       bool, one column per modality, True = available
"""

import pandas as pd

FEAT_DIR = "Feature Extraction/Features"
DATASET_DIR = "Dataset"

MODALITY_FILES = {
    "ecg": f"{FEAT_DIR}/ecg_features.csv",
    "eda": f"{FEAT_DIR}/eda_features.csv",
    "resp": f"{FEAT_DIR}/resp_features.csv",
    "video": f"{FEAT_DIR}/video11tasks_aus_gaze_mean_std.csv",
}


def load_audio_features():
    x_audio = pd.read_csv(f"{FEAT_DIR}/HCfeatures.csv", header=None, index_col=0)
    x_audio.index = [i.split(".")[0] for i in x_audio.index]
    x_audio.columns = [f"audio_{c}" for c in x_audio.columns]
    return x_audio


def load_modality(name, path):
    df = pd.read_csv(path, index_col=0)
    df.columns = [f"{name}_{c}" for c in df.columns]
    return df


def main():
    labels = pd.read_csv(f"{DATASET_DIR}/labels.csv", index_col=0)
    labels_supp = pd.read_csv(f"{DATASET_DIR}/labels_supplementary.csv", index_col=0)
    labels = labels.join(labels_supp, how="left")

    modalities = {name: load_modality(name, path) for name, path in MODALITY_FILES.items()}
    modalities["audio"] = load_audio_features()

    # Outer-join on the label index: every subject_task in labels.csv is kept,
    # missing modalities become NaN rows rather than being dropped.
    X = pd.DataFrame(index=labels.index)
    mask = pd.DataFrame(index=labels.index)
    for name, df in modalities.items():
        df = df[~df.index.duplicated(keep="first")]
        aligned = df.reindex(labels.index)
        X = X.join(aligned)
        mask[name] = ~aligned.isna().all(axis=1)

    X.to_csv(f"{FEAT_DIR}/multimodal_5modality_features.csv")
    mask.to_csv(f"{FEAT_DIR}/multimodal_5modality_mask.csv")

    print(f"Assembled {X.shape[0]} subject_task rows x {X.shape[1]} feature columns")
    print(f"Modalities: {list(modalities.keys())}")
    print()
    print("Per-modality availability (subject_task rows with that modality present):")
    print(mask.mean().round(3))
    print()
    n_complete = mask.all(axis=1).sum()
    print(f"Rows with ALL 5 modalities present: {n_complete} / {len(mask)} "
          f"({n_complete / len(mask):.1%})")
    print("(this is the row count the current notebook's inner-join silently keeps —"
          " everything else is thrown away)")

    combo_counts = mask.groupby(list(mask.columns)).size().sort_values(ascending=False)
    print()
    print("Modality-availability combinations (top 10):")
    print(combo_counts.head(10))


if __name__ == "__main__":
    main()
