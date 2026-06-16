"""
Train the lightweight edge cognitive classifier on real MotoStudent Albacete
telemetry. 1D-CNN over multivariate windows. One run per seed.

Usage: python train_cognitive.py --seed 42
Writes: results/cognitive_seed_<seed>.json and (best) checkpoints/cog_seed_<seed>.pt
"""
import os
import json
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             accuracy_score)

from dataio import build_windows, STATE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)


class CNN1D(nn.Module):
    def __init__(self, n_feat, window, n_classes, cfg):
        super().__init__()
        c1, c2 = cfg["conv_channels"]
        k = cfg["kernel_size"]
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, c1, k, padding=p), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, k, padding=p), nn.ReLU(), nn.BatchNorm1d(c2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(c2, cfg["fc_hidden"]), nn.ReLU(),
            nn.Linear(cfg["fc_hidden"], n_classes),
        )

    def forward(self, x):           # x: (B, W, F)
        x = x.transpose(1, 2)       # (B, F, W)
        return self.head(self.net(x))


def grouped_split(groups, y, split_cfg, rng):
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(split_cfg["train_frac"] * n))
    n_va = int(round(split_cfg["val_frac"] * n))
    tr_g = set(uniq[:n_tr])
    va_g = set(uniq[n_tr:n_tr + n_va])
    te_g = set(uniq[n_tr + n_va:])
    tr = np.array([g in tr_g for g in groups])
    va = np.array([g in va_g for g in groups])
    te = np.array([g in te_g for g in groups])
    return tr, va, te


def run(seed, cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_all = json.load(f)
    cfg = cfg_all["cognitive"]
    set_seed(seed)
    rng = np.random.RandomState(seed)

    X, XL, y, g, meta = build_windows(cfg_path)
    n_feat = X.shape[2]
    W = X.shape[1]

    tr, va, te = grouped_split(g, y, cfg["split"], rng)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]

    device = "cpu"
    model = CNN1D(n_feat, W, 4, cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # class weights from training set
    counts = np.bincount(ytr, minlength=4).astype(float)
    weights = counts.sum() / (4 * np.maximum(counts, 1))
    wt = torch.tensor(weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=wt if cfg["training"]["class_weighting"] else None)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"]["weight_decay"])

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    Xte_t = torch.tensor(Xte, device=device)

    bs = cfg["training"]["batch_size"]
    best_va = -1.0
    best_state = None
    patience = cfg["training"]["early_stopping_patience"]
    bad = 0
    t0 = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        perm = rng.permutation(len(Xtr_t))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = crit(out, ytr_t[idx])
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            pva = model(Xva_t).argmax(1).cpu().numpy()
        _, _, f1, _ = precision_recall_fscore_support(yva, pva, labels=[0, 1, 2, 3],
                                                      average="macro", zero_division=0)
        if f1 > best_va:
            best_va = f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    # inference latency: per-window, single sample, averaged
    with torch.no_grad():
        # warmup
        for _ in range(20):
            _ = model(Xte_t[:1])
        n_lat = min(500, len(Xte_t))
        lt0 = time.perf_counter()
        for j in range(n_lat):
            _ = model(Xte_t[j:j + 1])
        lat_ms = (time.perf_counter() - lt0) / n_lat * 1000.0
        pte = model(Xte_t).argmax(1).cpu().numpy()

    prec, rec, f1, sup = precision_recall_fscore_support(
        yte, pte, labels=[0, 1, 2, 3], average=None, zero_division=0)
    macro = precision_recall_fscore_support(
        yte, pte, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    acc = accuracy_score(yte, pte)
    cm = confusion_matrix(yte, pte, labels=[0, 1, 2, 3]).tolist()

    result = {
        "seed": seed,
        "n_params": int(n_params),
        "model_kb": round(n_params * 4 / 1024.0, 2),
        "inference_latency_ms": round(float(lat_ms), 4),
        "train_time_s": round(train_time, 2),
        "n_windows": int(meta["n_windows"]),
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "accuracy": float(acc),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_class": {
            STATE_NAMES[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(4)},
        "confusion_matrix": cm,
        "class_counts": meta["class_counts"],
        "total_seconds": meta["total_seconds"],
        "feature_names": meta["feature_names"],
        "y_true": yte.tolist(),
        "y_pred": pte.tolist(),
    }

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"cognitive_seed_{seed}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save(best_state, os.path.join(HERE, "checkpoints", f"cog_seed_{seed}.pt"))
    logging.info("seed %d: macro-F1=%.4f acc=%.4f params=%d lat=%.3fms",
                 seed, result["macro_f1"], acc, n_params, lat_ms)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = run(args.seed, args.config)
    print(json.dumps({k: res[k] for k in
                      ["seed", "macro_f1", "accuracy", "n_params", "inference_latency_ms"]},
                     indent=2))


if __name__ == "__main__":
    main()
