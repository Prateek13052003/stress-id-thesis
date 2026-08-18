"""
make_classification_fixed.py
============================
Original `make_classification.py` ka corrected version.

Original code mein teen problems the:

  BUG 1 -- SUBJECT LEAKAGE
      Notebooks `make_nclassif_random_splits*` call karte hain, jo plain
      `train_test_split` use karta hai.  Ek hi subject ke tasks train aur
      test dono mein chale jaate hain, isliye model subject ko pehchan
      leta hai stress ko nahi.  Sahi function (`make_nclassif`, GroupKFold
      ke saath) file mein maujood hai par KABHI CALL NAHI HOTA.

  BUG 2 -- DEAD SHUFFLE
      `make_nclassif` mein:
          X_shuffled, y_shuffled, groups_shuffled = shuffle(X, y, groups)
          group_kfold.split(X, y, groups)      # <- shuffled use hi nahi hue
      Shuffle ka koi asar nahi.

  BUG 3 -- FAKE VARIANCE
      GroupKFold deterministic hai.  10 folds = ek hi fixed partition ke
      10 tukde.  Jo "+/- std over 10 splits" report hota hai wo run-to-run
      variance NAHI hai.  Sahi tareeka: GroupShuffleSplit with different
      random_state per repetition.

Ye file teen split modes deti hai taaki teeno ko seedha compare kiya ja sake:

    "random"        -> original (leaky) baseline
    "groupkfold"    -> subject-wise, deterministic folds
    "groupshuffle"  -> subject-wise + asli randomness   <-- RECOMMENDED

Baaki sab (imputer, scaler, classifiers, metrics) original jaisa hi rakha
gaya hai taaki comparison fair rahe.
"""

import time
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, GroupKFold, GroupShuffleSplit,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, f1_score,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:                                   # pragma: no cover
    SMOTE = None
    _HAS_SMOTE = False


VALID_MODES = ("random", "groupkfold", "groupshuffle")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def subjects_from_index(index):
    """'2ea4_Counting1' -> '2ea4'.  Grouping isi pe hoti hai."""
    return np.array([str(v).split("_")[0] for v in index])


def default_classifiers():
    return [
        LogisticRegression(max_iter=2000),
        RandomForestClassifier(max_depth=5, random_state=0),
        AdaBoostClassifier(n_estimators=100, random_state=0),
    ]


def build_pipeline(model, feature_selector, impute, scale):
    """Original code jaisa hi pipeline.

    Zaroori baat: imputer aur scaler PIPELINE KE ANDAR hain, isliye wo
    sirf training fold pe fit hote hain.  Ye original code mein bhi sahi
    tha -- yahan preprocessing leakage nahi hai.
    """
    steps = []
    if impute:
        steps.append(("impute", IterativeImputer(random_state=0)))
    if scale:
        steps.append(("scale", StandardScaler()))

    if feature_selector == "l1":
        steps.append(("feature_selection", SelectFromModel(
            LogisticRegression(max_iter=5000, C=0.1, penalty="l1",
                               dual=False, solver="saga"))))
    elif feature_selector == "RFE":
        steps.append(("feature_selection", RFE(
            RandomForestClassifier(max_depth=5, random_state=0),
            n_features_to_select=20, step=2)))
    elif feature_selector == "RFECV":
        steps.append(("feature_selection", RFECV(
            RandomForestClassifier(max_depth=5, random_state=0),
            step=2, cv=2)))
    elif feature_selector == "PCA":
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))

    steps.append(("classification", model))
    return Pipeline(steps)


def iter_splits(X, y, mode, n_splits, test_size, seed):
    """(train_idx, test_idx) positional indices yield karta hai.

    mode = "random"        -> koi grouping nahi  (ORIGINAL, LEAKY)
    mode = "groupkfold"    -> subject-wise, deterministic
    mode = "groupshuffle"  -> subject-wise, har repetition alag random
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode '{mode}' invalid. Use one of {VALID_MODES}")

    n = len(y)
    positions = np.arange(n)

    if mode == "random":
        for s in range(n_splits):
            tr, te = train_test_split(
                positions, test_size=test_size, random_state=seed + s
            )
            yield s, tr, te

    elif mode == "groupkfold":
        groups = subjects_from_index(y.index)
        gkf = GroupKFold(n_splits=n_splits)
        for s, (tr, te) in enumerate(gkf.split(X, y, groups)):
            yield s, tr, te

    else:  # groupshuffle
        groups = subjects_from_index(y.index)
        gss = GroupShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=seed
        )
        for s, (tr, te) in enumerate(gss.split(X, y, groups)):
            yield s, tr, te


def leakage_report(y, train_idx, test_idx):
    """Ek split mein kitne subjects dono taraf hain -- proof ke liye."""
    subs = subjects_from_index(y.index)
    tr_s, te_s = set(subs[train_idx]), set(subs[test_idx])
    overlap = tr_s & te_s
    return {
        "train_subjects": len(tr_s),
        "test_subjects": len(te_s),
        "overlapping_subjects": len(overlap),
        "leaky": len(overlap) > 0,
    }


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------

def run_classification(
    X, y,
    mode="groupshuffle",
    n_splits=10,
    test_size=0.2,
    seed=0,
    resample=False,
    feature_selector=None,
    list_classifiers=None,
    impute=True,
    scale=True,
    verbose=True,
):
    """Ek modality pe saare classifiers evaluate karta hai.

    Returns
    -------
    results     : DataFrame  (per split, per classifier)
    conf_mats   : list of confusion matrices
    split_info  : DataFrame  (har split ka leakage audit)
    """
    if list_classifiers is None:
        list_classifiers = default_classifiers()

    X = pd.DataFrame(X)
    y = pd.Series(y)
    if not X.index.equals(y.index):
        raise ValueError("X aur y ka index match nahi karta -- pehle align karo.")

    # sklearn ko string column names chahiye hote hain kabhi-kabhi
    X = X.copy()
    X.columns = [str(c) for c in X.columns]

    rows, conf_mats, split_rows = [], [], []

    for s, tr, te in iter_splits(X, y, mode, n_splits, test_size, seed):
        x_train, x_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]

        info = leakage_report(y, tr, te)
        info.update({"split": s, "mode": mode,
                     "n_train": len(tr), "n_test": len(te)})
        split_rows.append(info)

        # SMOTE sirf training set pe -- split ke BAAD.
        # (original code mein bhi ye sahi tha)
        if resample:
            if not _HAS_SMOTE:
                raise ImportError(
                    "resample=True ke liye imbalanced-learn chahiye.\n"
                    "Install karo:  pip install imbalanced-learn"
                )
            if y_train.nunique() < 2:
                warnings.warn(f"Split {s}: train mein ek hi class, SMOTE skip.")
            else:
                sm = SMOTE(random_state=seed + s)
                xr, yr = sm.fit_resample(x_train, y_train)
                x_train = pd.DataFrame(xr, columns=x_train.columns)
                y_train = pd.Series(yr)

        if verbose:
            flag = "LEAKY" if info["leaky"] else "clean"
            print(f"  split {s + 1:2d}/{n_splits}  "
                  f"train={len(tr):4d} test={len(te):4d}  "
                  f"overlap={info['overlapping_subjects']:2d} [{flag}]")

        for model in list_classifiers:
            clf = build_pipeline(model, feature_selector, impute, scale)

            tic = time.perf_counter()
            clf.fit(x_train, y_train)
            toc = time.perf_counter()

            y_pred = clf.predict(x_test)
            conf_mats.append(confusion_matrix(y_test, y_pred))

            rows.append({
                "split": s,
                "mode": mode,
                "classifier": model.__class__.__name__,
                "f1-score": f1_score(y_test, y_pred, average="weighted"),
                "accuracy": balanced_accuracy_score(y_test, y_pred),
                "time": toc - tic,
            })

    return pd.DataFrame(rows), conf_mats, pd.DataFrame(split_rows)


def summarise(results):
    """Mean +/- std per classifier."""
    g = results.groupby("classifier")[["f1-score", "accuracy", "time"]]
    out = g.agg(["mean", "std"]).round(3)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


def compare_modes(
    X, y,
    modes=("random", "groupshuffle"),
    n_splits=10,
    seed=0,
    **kwargs,
):
    """Wahi data, wahi classifiers -- sirf split strategy badal ke chalao.

    Yahi thesis ka headline experiment hai:
    leaky vs clean ka gap.
    """
    all_res, all_info = [], []
    for m in modes:
        print(f"\n--- mode: {m} ---")
        res, _, info = run_classification(
            X, y, mode=m, n_splits=n_splits, seed=seed, **kwargs
        )
        all_res.append(res)
        all_info.append(info)

    results = pd.concat(all_res, ignore_index=True)
    info = pd.concat(all_info, ignore_index=True)

    # flat, readable table -- ek row per classifier
    table = []
    for clf, grp in results.groupby("classifier"):
        row = {"classifier": clf}
        for m in modes:
            sub = grp[grp["mode"] == m]
            row[f"f1_{m}"] = round(sub["f1-score"].mean(), 3)
            row[f"acc_{m}"] = round(sub["accuracy"].mean(), 3)
        if len(modes) == 2:
            a, b = modes
            row["f1_drop"] = round(row[f"f1_{a}"] - row[f"f1_{b}"], 3)
            row["acc_drop"] = round(row[f"acc_{a}"] - row[f"acc_{b}"], 3)
        table.append(row)

    return results, pd.DataFrame(table), info
