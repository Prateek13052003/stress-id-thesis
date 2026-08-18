"""
Within-task balanced accuracy: neutralizes the task-identity confound
confirmed in task_identity_baseline.py (knowing only which of the 11 tasks a
row belongs to gets balanced_acc ~0.69-0.71, matching the pooled cross-task
numbers everywhere else in this project).

Standard balanced_accuracy_score is computed by pooling ALL test rows across
all 11 tasks. Since task identity alone predicts binary-stress well (Relax/
Breathing are ~85% non-stress, Counting/Math/Stroop/Speaking are ~65-77%
stress, by protocol design), a pooled score cannot distinguish "the model
detected task identity" from "the model detected physiological stress".

Within-task balanced accuracy instead scores each task's rows SEPARATELY
(balanced_accuracy_score restricted to that task's subset), then averages
across tasks. Task identity is constant within one task's subset, so it
carries exactly zero information there: a classifier that only ever predicts
each task's majority class gets exactly 0.5 on every task (by definition of
balanced accuracy for a 2-class problem — 100% recall on one class, 0% on the
other, average 0.5). Any score reliably above 0.5 under this metric is
evidence of real within-task discrimination, not task lookup.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def within_task_balanced_accuracy(y_true, y_pred, task, min_rows_per_class=1):
    """
    y_true, y_pred: array-like, same length
    task: array-like, same length, task label per row
    Returns: (macro_avg_over_tasks, per_task_dict)

    Tasks where the test subset has only one true class present are skipped
    from the macro average (balanced_accuracy_score is undefined/degenerate
    there) but still reported in per_task_dict with a note, for transparency.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "task": np.asarray(task)})
    per_task = {}
    valid_scores = []
    for t, group in df.groupby("task"):
        n_classes_present = group["y_true"].nunique()
        if n_classes_present < 2:
            per_task[t] = {"balanced_acc": None, "n": len(group),
                            "note": "only one class present in this task's test rows, skipped"}
            continue
        score = balanced_accuracy_score(group["y_true"], group["y_pred"])
        per_task[t] = {"balanced_acc": float(score), "n": len(group)}
        valid_scores.append(score)

    macro_avg = float(np.mean(valid_scores)) if valid_scores else None
    return macro_avg, per_task
