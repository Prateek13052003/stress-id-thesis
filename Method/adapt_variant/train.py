"""
Two-stage training of the physio-anchored ADAPT variant on StressID, with
subject-wise splits (no leakage — see Classification/dataset_audit.py for why
this matters).

Overfitting fix (v2): the first version trained a fixed epoch count and
overfit badly (train balanced-acc ~0.90 vs test ~0.63 after 40+40 epochs,
no regularization tuning). This version:
  - carves a subject-disjoint INNER validation set out of the training
    subjects (never seen by the outer test set) and uses it for early
    stopping in both the anchoring and fusion stages;
  - shrinks the modality encoders (hidden 128->64) and adds dropout inside
    them (they had no dropout before);
  - restores the best-validation-epoch weights before evaluating on the
    untouched outer test set.

Simplification vs. the original ADAPT paper, still noted explicitly: modality
robustness is trained via random modality-dropout augmentation under plain
cross-entropy, rather than ADAPT's additional multi-view InfoNCE fusion loss.

Usage:
    python3 -m Method.adapt_variant.train --n-folds 5 --anchor ecg
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Method.adapt_variant.data import (
    MODALITIES, ModalityStats, StressIDMultimodalDataset,
    build_full_table, subject_groups,
)
from Method.adapt_variant.model import PhysioAnchoredADAPT, info_nce
from Classification.metrics_utils import within_task_balanced_accuracy

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    feats = {name: torch.stack([b[0][name] for b in batch]) for name in MODALITIES}
    avail = torch.stack([b[1] for b in batch])
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return feats, avail, y


def inner_train_val_split(train_idx, val_frac, seed):
    """Subject-wise split of the training subjects into fit/val, for early
    stopping only. The outer test set is untouched by this."""
    grp = subject_groups(train_idx)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    fit_pos, val_pos = next(splitter.split(train_idx, groups=grp))
    return train_idx[fit_pos], train_idx[val_pos]


@torch.no_grad()
def anchoring_val_loss(model, loader, device, anchor, temperature):
    model.encoders.eval()
    total, n_batches = 0.0, 0
    anchor_idx = model.modalities.index(anchor)
    for feats, avail, _ in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        avail = avail.to(device)
        embs = model.encode(feats)
        anchor_emb = embs[anchor]
        anchor_ok = avail[:, anchor_idx]
        loss, n_terms = 0.0, 0
        for name in model.modalities:
            if name == anchor:
                continue
            paired = anchor_ok & avail[:, model.modalities.index(name)]
            if paired.sum() < 2:
                continue
            loss = loss + info_nce(anchor_emb[paired], embs[name][paired], temperature).item()
            n_terms += 1
        if n_terms:
            total += loss / n_terms
            n_batches += 1
    model.encoders.train()
    return total / max(n_batches, 1)


def train_anchoring(model, train_loader, val_loader, device, anchor, max_epochs, lr, temperature, patience):
    opt = torch.optim.AdamW(model.encoders.parameters(), lr=lr, weight_decay=0.05)
    anchor_idx = model.modalities.index(anchor)

    best_val, best_state, bad_epochs, stopped_epoch = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        total, n_batches = 0.0, 0
        for feats, avail, _ in train_loader:
            feats = {k: v.to(device) for k, v in feats.items()}
            avail = avail.to(device)
            embs = model.encode(feats)
            anchor_emb = embs[anchor]
            anchor_ok = avail[:, anchor_idx]

            loss = torch.tensor(0.0, device=device)
            n_terms = 0
            for name in model.modalities:
                if name == anchor:
                    continue
                paired = anchor_ok & avail[:, model.modalities.index(name)]
                if paired.sum() < 2:
                    continue
                loss = loss + info_nce(anchor_emb[paired], embs[name][paired], temperature)
                n_terms += 1
            if n_terms == 0:
                continue
            loss = loss / n_terms

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1

        val_loss = anchoring_val_loss(model, val_loader, device, anchor, temperature)
        stopped_epoch = epoch + 1
        if val_loss < best_val - 1e-4:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.encoders.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.encoders.load_state_dict(best_state)
    return stopped_epoch


def modality_dropout(avail, p, generator):
    if p <= 0:
        return avail
    drop = torch.rand(avail.shape, generator=generator) < p
    return avail & ~drop.to(avail.device)


@torch.no_grad()
def eval_fusion_val(model, loader, device):
    model.eval()
    preds, ys = [], []
    for feats, avail, y in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        avail = avail.to(device)
        logits, _ = model(feats, avail)
        preds.append(logits.argmax(-1).cpu())
        ys.append(y)
    model.train()
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return balanced_accuracy_score(ys, preds)


def train_fusion(model, train_loader, val_loader, device, max_epochs, lr, dropout_p, patience, seed):
    for p in model.encoders.parameters():
        p.requires_grad_(False)
    model.encoders.eval()

    opt = torch.optim.AdamW(
        list(model.fusion.parameters()) + list(model.classifier.parameters()),
        lr=lr, weight_decay=0.05,
    )
    gen = torch.Generator().manual_seed(seed)

    best_val, best_state, bad_epochs, stopped_epoch = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        total, correct, n = 0.0, 0, 0
        for feats, avail, y in train_loader:
            feats = {k: v.to(device) for k, v in feats.items()}
            avail_aug = modality_dropout(avail, dropout_p, gen).to(device)
            y = y.to(device)

            with torch.no_grad():
                embs = model.encode(feats)
            stacked = torch.stack([embs[name] for name in model.modalities], dim=1)
            cls_out, _ = model.fusion(stacked, avail_aug)
            logits = model.classifier(cls_out)

            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            n += y.size(0)

        val_acc = eval_fusion_val(model, val_loader, device)
        stopped_epoch = epoch + 1
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {
                "fusion": copy.deepcopy(model.fusion.state_dict()),
                "classifier": copy.deepcopy(model.classifier.state_dict()),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.fusion.load_state_dict(best_state["fusion"])
        model.classifier.load_state_dict(best_state["classifier"])

    for p in model.encoders.parameters():
        p.requires_grad_(True)
    return stopped_epoch, best_val


@torch.no_grad()
def evaluate(model, loader, device, force_missing=None, task=None):
    """
    task: optional array aligned with the loader's (non-shuffled) iteration
    order, one task-name per sample. When given, also reports within-task
    balanced accuracy (Classification/metrics_utils.py) — the metric that
    actually distinguishes "the model uses physiological signal" from "the
    model recovered task identity", since binary-stress is itself strongly
    task-dependent in StressID (see Classification/task_identity_baseline.py).
    The caller MUST use a non-shuffled loader for `task` alignment to be valid.
    """
    model.eval()
    preds, ys = [], []
    for feats, avail, y in loader:
        feats = {k: v.to(device) for k, v in feats.items()}
        avail = avail.to(device)
        logits, _ = model(feats, avail, ablate_modalities=force_missing)
        preds.append(logits.argmax(-1).cpu())
        ys.append(y)
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    result = {
        "balanced_acc": balanced_accuracy_score(ys, preds),
        "f1": f1_score(ys, preds, average="weighted"),
        "n": int(len(ys)),
    }
    if task is not None:
        wt_bacc, per_task = within_task_balanced_accuracy(ys, preds, task)
        result["within_task_balanced_acc"] = wt_bacc
        result["within_task_per_task"] = per_task
    return result


def run_one_fold(X, mask, y, groups, train_idx, test_idx, args, device, anchor=None, verbose=True):
    anchor = anchor or args.anchor
    fit_idx, val_idx = inner_train_val_split(train_idx, args.val_frac, args.seed)

    stats = ModalityStats(X, mask, groups, fit_idx)
    fit_ds = StressIDMultimodalDataset(X.loc[fit_idx], mask.loc[fit_idx], y.loc[fit_idx], groups, stats)
    val_ds = StressIDMultimodalDataset(X.loc[val_idx], mask.loc[val_idx], y.loc[val_idx], groups, stats)
    test_ds = StressIDMultimodalDataset(X.loc[test_idx], mask.loc[test_idx], y.loc[test_idx], groups, stats)

    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    in_dims = {name: len(cols) for name, cols in groups.items()}
    n_classes = int(y.nunique())
    model = PhysioAnchoredADAPT(in_dims, n_classes=n_classes, modalities=MODALITIES).to(device)

    ep1 = train_anchoring(model, fit_loader, val_loader, device, anchor,
                           args.epochs_anchor, args.lr, args.temperature, args.patience)
    ep2, best_val_acc = train_fusion(model, fit_loader, val_loader, device,
                                      args.epochs_fusion, args.lr, args.dropout_p, args.patience, args.seed)

    if verbose:
        print(f"  anchor={anchor}  anchoring stopped @ epoch {ep1}, fusion stopped @ epoch {ep2} "
              f"(best val balanced_acc={best_val_acc:.3f})")

    task_test = pd.Series(test_idx).apply(lambda i: i.split("_", 1)[1]).values

    train_perf = evaluate(model, fit_loader, device)
    val_perf = evaluate(model, val_loader, device)
    results = {"train": train_perf, "val": val_perf,
               "full": evaluate(model, test_loader, device, task=task_test)}
    for scenario, missing in [
        ("no_video", ["video"]),
        ("no_audio", ["audio"]),
        ("real_life_no_video_no_audio", ["video", "audio"]),
    ]:
        results[scenario] = evaluate(model, test_loader, device, force_missing=missing, task=task_test)

    extra = {"stats": stats, "test_loader": test_loader, "test_ds": test_ds,
             "fit_idx": fit_idx, "val_idx": val_idx}
    return model, results, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="binary-stress")
    ap.add_argument("--anchor", default="ecg", choices=MODALITIES)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs-anchor", type=int, default=150)
    ap.add_argument("--epochs-fusion", type=int, default=150)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.08)
    ap.add_argument("--dropout-p", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1999)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    X, mask, y, groups = build_full_table(args.target)
    grp = subject_groups(X.index)

    OUT_DIR.mkdir(exist_ok=True)
    all_results = []

    if args.n_folds <= 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
        splits = splitter.split(X, y, grp)
    else:
        splitter = GroupKFold(n_splits=args.n_folds)
        splits = splitter.split(X, y, grp)

    for fold, (train_pos, test_pos) in enumerate(splits):
        train_idx = X.index[train_pos]
        test_idx = X.index[test_pos]
        print(f"\n=== Fold {fold + 1} ({len(set(grp[train_pos]))} train subjects, "
              f"{len(set(grp[test_pos]))} test subjects) ===")
        _, results, _ = run_one_fold(X, mask, y, groups, train_idx, test_idx, args, device)
        for scenario, metrics in results.items():
            wt = metrics.get("within_task_balanced_acc")
            wt_str = f"  within_task_balanced_acc={wt:.3f}" if wt is not None else ""
            print(f"  {scenario:32s} balanced_acc={metrics['balanced_acc']:.3f}  f1={metrics['f1']:.3f}{wt_str}")
        all_results.append(results)

    summary = {}
    for scenario in all_results[0]:
        accs = [r[scenario]["balanced_acc"] for r in all_results]
        f1s = [r[scenario]["f1"] for r in all_results]
        summary[scenario] = {
            "balanced_acc_mean": float(np.mean(accs)), "balanced_acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        }
        wt_vals = [r[scenario].get("within_task_balanced_acc") for r in all_results]
        if all(v is not None for v in wt_vals):
            summary[scenario]["within_task_balanced_acc_mean"] = float(np.mean(wt_vals))
            summary[scenario]["within_task_balanced_acc_std"] = float(np.std(wt_vals))

    print("\n=== Summary across folds ===")
    for scenario, s in summary.items():
        wt_str = ""
        if "within_task_balanced_acc_mean" in s:
            wt_str = (f"  within_task_balanced_acc={s['within_task_balanced_acc_mean']:.3f}"
                      f"+/-{s['within_task_balanced_acc_std']:.3f}")
        print(f"  {scenario:32s} balanced_acc={s['balanced_acc_mean']:.3f}+/-{s['balanced_acc_std']:.3f}  "
              f"f1={s['f1_mean']:.3f}+/-{s['f1_std']:.3f}{wt_str}")

    with open(OUT_DIR / f"results_{args.target}_anchor-{args.anchor}.json", "w") as f:
        json.dump({"args": vars(args), "per_fold": all_results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
