"""
Ablation study: contribution of each architectural component.

Configurations:
  1. full          : learned CNN classifier on independent (non-label) features
                     + real PBFT consensus tier.
  2. no_consensus  : classifier consumes a SINGLE noisy node's input (no
                     agreement), modelling cognition on an unfiltered faulty
                     feed; resilience column = PBFT accuracy with the consensus
                     tier removed (i.e. Raft / single-node behaviour) at the
                     operating point.
  3. no_classifier : threshold-rule labeler (no learning) on the label-defining
                     channels -- an UPPER bound on what hand rules achieve since
                     it sees the very channels the labels are built from.

Cognitive macro-F1 is measured on real MotoStudent test data (independent
feature set, FIX M5). The resilience column is measured on the REAL PBFT/Raft
simulation (FIX M1-M4) at f/n = 0.33 (n=7, coordinated equivocation), which is
inside the BFT bound for n=7 (f_tol=2) -- PBFT must keep safety 1.0 there.
"""
import os
import json
import argparse
import numpy as np
import torch

from dataio import build_windows, STATE_NAMES
from train_cognitive import CNN1D, set_seed, grouped_split
from run_consensus_sim import simulate
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))


def _train_eval_classifier(X, y, g, cfg, seed, corrupt_test=False, rng=None):
    """Train classifier (same early-stopping protocol as train_cognitive.py),
    return macro-F1 on the (optionally corrupted) test set."""
    set_seed(seed)
    rng = rng or np.random.RandomState(seed)
    tr, va, te = grouped_split(g, y, cfg["split"], rng)
    model = CNN1D(X.shape[2], X.shape[1], 4, cfg["model"])
    counts = np.bincount(y[tr], minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"]["weight_decay"])
    Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
    Xva_t = torch.tensor(X[va])
    bs = cfg["training"]["batch_size"]
    patience = cfg["training"]["early_stopping_patience"]
    best_va, best_state, bad = -1.0, None, 0
    for ep in range(cfg["training"]["epochs"]):
        model.train()
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pva = model(Xva_t).argmax(1).numpy()
        _, _, f1v, _ = precision_recall_fscore_support(
            y[va], pva, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        if f1v > best_va:
            best_va = f1v
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    Xte = X[te].copy()
    if corrupt_test:
        # simulate raw single-node faulty input: inject heavy noise on a fraction
        n_te = len(Xte)
        k = int(0.20 * n_te)
        sel = rng.choice(n_te, k, replace=False)
        Xte[sel] += rng.normal(0, 3.0, Xte[sel].shape).astype(np.float32)
    with torch.no_grad():
        pte = model(torch.tensor(Xte)).argmax(1).numpy()
    _, _, f1, _ = precision_recall_fscore_support(
        y[te], pte, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    return float(f1)


def _threshold_rule_f1(XL, y, g, cfg, seed, rng, label_names):
    """Threshold-only classifier on the LABEL-DEFINING channels (no learning).

    This baseline deliberately uses the same channels the labels are built from
    (an upper bound for hand rules); it isolates the value of LEARNING from an
    INDEPENDENT feature set over the convenience of the label channels.
    """
    set_seed(seed)
    rng = rng or np.random.RandomState(seed)
    tr, va, te = grouped_split(g, y, cfg["split"], rng)
    # locate channels within the label-feature block
    def idx(name, default):
        return label_names.index(name) if name in label_names else default
    ia = idx("long_accel", 0)
    isp = idx("speed_kmh", 0)
    ith = idx("throttle", 0)
    XteL = XL[te]
    acc = XteL[:, :, ia].mean(axis=1)
    sp = XteL[:, :, isp].mean(axis=1)
    th = XteL[:, :, ith].mean(axis=1)
    pred = np.full(len(XteL), 1)  # default cornering
    pred[acc < -0.5] = 0                       # braking
    pred[(acc > 0.5) & (th > 0)] = 2           # accel
    pred[(np.abs(acc) <= 0.5) & (sp > 0.5)] = 3   # cruise (high norm speed)
    _, _, f1, _ = precision_recall_fscore_support(
        y[te], pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    return float(f1)


def run(seed, cfg_path):
    with open(cfg_path) as f:
        cfg_all = json.load(f)
    cfg = cfg_all["cognitive"]
    ccfg = cfg_all["consensus_sim"]
    X, XL, y, g, meta = build_windows(cfg_path)
    label_names = meta["label_feature_names"]

    # cognitive metrics (independent feature set)
    f1_full = _train_eval_classifier(X, y, g, cfg, seed, corrupt_test=False,
                                     rng=np.random.RandomState(seed))
    f1_noconsensus = _train_eval_classifier(X, y, g, cfg, seed, corrupt_test=True,
                                            rng=np.random.RandomState(seed))
    f1_thresh = _threshold_rule_f1(XL, y, g, cfg, seed,
                                   np.random.RandomState(seed), label_names)

    # resilience at f/n=0.33, n=7, equivocate (inside BFT bound: f_tol=2)
    pbft = simulate(ccfg, seed, 7, 0.33, "equivocate", "pbft")
    raft = simulate(ccfg, seed, 7, 0.33, "equivocate", "raft")

    out = {
        "full": {"macro_f1": f1_full,
                 "safety": pbft["safety"], "accuracy": pbft["accuracy"]},
        "no_consensus": {"macro_f1": f1_noconsensus,
                         "safety": raft["safety"], "accuracy": raft["accuracy"]},
        "no_classifier": {"macro_f1": f1_thresh,
                          "safety": None, "accuracy": None},
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"ablation_seed_{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    out = run(args.seed, args.config)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
