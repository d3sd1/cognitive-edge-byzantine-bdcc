"""
Statistical helpers: 95% confidence intervals (bootstrap and t), paired
significance tests, and effect sizes. Used by evaluate.py (FIX M6).
"""
import numpy as np
from scipy import stats as sps
from sklearn.metrics import precision_recall_fscore_support


def t_ci(vals, conf=0.95):
    """Two-sided t confidence interval for the mean of `vals`."""
    a = np.asarray(vals, dtype=float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "std": 0.0, "n": n}
    se = a.std(ddof=1) / np.sqrt(n)
    tcrit = sps.t.ppf(0.5 + conf / 2, df=n - 1)
    return {"mean": mean, "lo": mean - tcrit * se, "hi": mean + tcrit * se,
            "std": float(a.std(ddof=1)), "n": n}


def bootstrap_macro_f1_ci(y_true, y_pred, n_boot=2000, conf=0.95, seed=0):
    """Percentile bootstrap 95% CI for macro-F1 over pooled predictions."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], labels=[0, 1, 2, 3],
            average="macro", zero_division=0)
        stats.append(f1)
    stats = np.array(stats)
    lo = np.percentile(stats, 100 * (0.5 - conf / 2))
    hi = np.percentile(stats, 100 * (0.5 + conf / 2))
    _, _, point, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    return {"mean": float(point), "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "n_samples": int(n)}


def bootstrap_per_class_ci(y_true, y_pred, n_boot=2000, conf=0.95, seed=0):
    """Percentile bootstrap CI for per-class F1 (wide where support is low)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot = np.zeros((n_boot, 4))
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], labels=[0, 1, 2, 3],
            average=None, zero_division=0)
        boot[b] = f1
    _, _, point, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    out = {}
    names = ["braking", "cornering", "accel", "cruise"]
    for i, nm in enumerate(names):
        out[nm] = {
            "f1": float(point[i]),
            "lo": float(np.percentile(boot[:, i], 100 * (0.5 - conf / 2))),
            "hi": float(np.percentile(boot[:, i], 100 * (0.5 + conf / 2))),
            "support": int(sup[i]),
        }
    return out


def paired_test(a, b):
    """Paired comparison of two per-seed metric vectors.

    Returns paired t-test and Wilcoxon p-values, mean difference, and Cohen's
    d_z (paired effect size). Handles degenerate (zero-variance) cases.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    diff = a - b
    md = float(diff.mean())
    out = {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": md, "n": int(len(a))}
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    if sd == 0:
        out["t_p"] = 0.0 if md != 0 else 1.0
        out["cohens_dz"] = float("inf") if md != 0 else 0.0
        out["wilcoxon_p"] = None
        out["note"] = "zero within-pair variance (deterministic separation)"
        return out
    t, p = sps.ttest_rel(a, b)
    out["t_stat"] = float(t); out["t_p"] = float(p)
    out["cohens_dz"] = float(md / sd)
    try:
        if np.any(diff != 0):
            w, wp = sps.wilcoxon(a, b)
            out["wilcoxon_p"] = float(wp)
        else:
            out["wilcoxon_p"] = 1.0
    except ValueError:
        out["wilcoxon_p"] = None
    return out
