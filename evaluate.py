"""
Aggregate per-seed results into results/aggregated_results.json with 95%
confidence intervals (bootstrap / t), significance tests, and effect sizes
(FIX M6). Also aggregates the real PBFT/Raft consensus simulation (safety vs
accuracy, message counts), the network-modelled latency sweep, and the
ablation.
"""
import os
import json
import platform
import numpy as np

from stats_utils import (t_ci, bootstrap_macro_f1_ci, bootstrap_per_class_ci,
                         paired_test)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SEEDS = [42, 123, 456, 789, 1024]


def _load(prefix):
    out = []
    for s in SEEDS:
        p = os.path.join(RES, f"{prefix}_seed_{s}.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def _ms(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def aggregate_cognitive():
    runs = _load("cognitive")
    if not runs:
        return None
    out = {"n_seeds": len(runs)}
    # per-seed CIs (t over 5 seeds) for headline metrics
    out["macro_f1"] = {**_ms([r["macro_f1"] for r in runs]),
                       "ci95": t_ci([r["macro_f1"] for r in runs])}
    out["accuracy"] = {**_ms([r["accuracy"] for r in runs]),
                       "ci95": t_ci([r["accuracy"] for r in runs])}
    out["macro_precision"] = _ms([r["macro_precision"] for r in runs])
    out["macro_recall"] = _ms([r["macro_recall"] for r in runs])
    out["n_params"] = runs[0]["n_params"]
    out["model_kb"] = runs[0]["model_kb"]
    out["feature_names"] = runs[0].get("feature_names")
    out["inference_latency_ms"] = _ms([r["inference_latency_ms"] for r in runs])
    out["train_time_s"] = _ms([r["train_time_s"] for r in runs])
    out["splits"] = {"n_train": runs[0]["n_train"], "n_val": runs[0]["n_val"],
                     "n_test": runs[0]["n_test"], "n_windows": runs[0]["n_windows"]}
    out["total_seconds"] = runs[0]["total_seconds"]
    out["class_counts"] = runs[0]["class_counts"]

    # pooled predictions across seeds -> bootstrap CI (robust to std=0)
    yt = np.concatenate([np.array(r["y_true"]) for r in runs])
    yp = np.concatenate([np.array(r["y_pred"]) for r in runs])
    out["macro_f1_bootstrap_ci95"] = bootstrap_macro_f1_ci(yt, yp, seed=0)
    out["per_class_bootstrap_ci95"] = bootstrap_per_class_ci(yt, yp, seed=0)

    # aggregate confusion matrix
    cms = np.array([r["confusion_matrix"] for r in runs]).sum(axis=0).astype(float)
    cmn = cms / np.maximum(cms.sum(axis=1, keepdims=True), 1)
    out["confusion_matrix_sum"] = cms.tolist()
    out["confusion_matrix_norm"] = cmn.tolist()
    return out


def aggregate_latency():
    runs = _load("latency")
    if not runs:
        return None
    keys = list(runs[0]["results"].keys())
    out = {"n_seeds": len(runs),
           "n_channels": runs[0]["n_channels"],
           "link_latency_ms": runs[0]["link_latency_ms"],
           "aggregate_rate_sps": runs[0]["aggregate_rate_sps"],
           "by_config": {}}
    for k in keys:
        rows = [r["results"][k] for r in runs]
        out["by_config"][k] = {
            "protocol": rows[0]["protocol"],
            "n_nodes": rows[0]["n_nodes"],
            "network_latency_ms": rows[0]["network_latency_ms"],
            "compute_median_ms": _ms([x["compute_median_ms"] for x in rows]),
            "median_ms": {**_ms([x["median_ms"] for x in rows]),
                          "ci95": t_ci([x["median_ms"] for x in rows])},
            "mean_ms": _ms([x["mean_ms"] for x in rows]),
            "p95_ms": _ms([x["p95_ms"] for x in rows]),
            "p99_ms": _ms([x["p99_ms"] for x in rows]),
            "exceeds_5ms_budget": any(x["exceeds_5ms_budget"] for x in rows),
            "sustained_throughput_sps": _ms(
                [x["sustained_throughput_sps"] for x in rows]),
        }
    return out


def aggregate_consensus():
    runs = _load("consensus")
    if not runs:
        return None
    keys = [(r["n_nodes"], r["fault_fraction"], r["behavior"])
            for r in runs[0]["sweep"]]
    agg = {}
    for k in keys:
        pf_saf, pf_acc, pf_live, pf_msg = [], [], [], []
        rf_saf, rf_acc, rf_live, rf_msg = [], [], [], []
        meta = None
        for run in runs:
            for row in run["sweep"]:
                if (row["n_nodes"], row["fault_fraction"], row["behavior"]) == k:
                    pf_saf.append(row["pbft_safety"]); pf_acc.append(row["pbft_accuracy"])
                    pf_live.append(row["pbft_liveness"]); pf_msg.append(row["pbft_avg_messages"])
                    rf_saf.append(row["raft_safety"]); rf_acc.append(row["raft_accuracy"])
                    rf_live.append(row["raft_liveness"]); rf_msg.append(row["raft_avg_messages"])
                    meta = {"f_actual": row["f_actual"], "f_tol": row["f_tol"],
                            "within_bft_bound": row["within_bft_bound"]}
        agg[f"n{k[0]}_f{k[1]:.2f}_{k[2]}"] = {
            "n_nodes": k[0], "fault_fraction": k[1], "behavior": k[2], **meta,
            "pbft_safety": _ms(pf_saf), "pbft_accuracy": _ms(pf_acc),
            "pbft_liveness": _ms(pf_live), "pbft_avg_messages": _ms(pf_msg),
            "raft_safety": _ms(rf_saf), "raft_accuracy": _ms(rf_acc),
            "raft_liveness": _ms(rf_live), "raft_avg_messages": _ms(rf_msg),
        }

    # representative table at n=7 over fractions (equivocate)
    rep = {}
    for frac in [0.10, 0.20, 0.33, 0.40, 0.50]:
        key = f"n7_f{frac:.2f}_equivocate"
        if key in agg:
            rep[f"{frac:.2f}"] = {
                "within_bound": agg[key]["within_bft_bound"],
                "f_actual": agg[key]["f_actual"], "f_tol": agg[key]["f_tol"],
                "pbft_safety": agg[key]["pbft_safety"],
                "pbft_accuracy": agg[key]["pbft_accuracy"],
                "raft_safety": agg[key]["raft_safety"],
                "raft_accuracy": agg[key]["raft_accuracy"],
            }

    # significance: PBFT vs Raft safety at n=7, equivocate, f/n=0.33 (in-bound)
    sig = {}
    k = "n7_f0.33_equivocate"
    pf_saf = [row["pbft_safety"] for run in runs for row in run["sweep"]
              if (row["n_nodes"], row["fault_fraction"], row["behavior"]) == (7, 0.33, "equivocate")]
    rf_saf = [row["raft_safety"] for run in runs for row in run["sweep"]
              if (row["n_nodes"], row["fault_fraction"], row["behavior"]) == (7, 0.33, "equivocate")]
    if pf_saf and rf_saf:
        sig["pbft_vs_raft_safety_n7_f0.33_equivocate"] = paired_test(pf_saf, rf_saf)
    return {"sweep": agg, "representative_n7_equivocate": rep, "significance": sig}


def aggregate_ablation():
    runs = _load("ablation")
    if not runs:
        return None
    out = {}
    cfg_names = list(runs[0].keys())
    series = {}
    for cfg_name in cfg_names:
        f1s = [r[cfg_name]["macro_f1"] for r in runs if r[cfg_name].get("macro_f1") is not None]
        saf = [r[cfg_name]["safety"] for r in runs if r[cfg_name].get("safety") is not None]
        acc = [r[cfg_name]["accuracy"] for r in runs if r[cfg_name].get("accuracy") is not None]
        series[cfg_name] = {"f1": f1s, "safety": saf, "accuracy": acc}
        out[cfg_name] = {
            "macro_f1": {**_ms(f1s), "ci95": t_ci(f1s)} if f1s else None,
            "safety": _ms(saf) if saf else None,
            "accuracy": _ms(acc) if acc else None,
        }
    # significance: full vs no_consensus (cognitive F1) and vs no_classifier
    out["significance"] = {}
    if series.get("full", {}).get("f1") and series.get("no_consensus", {}).get("f1"):
        out["significance"]["full_vs_no_consensus_f1"] = paired_test(
            series["full"]["f1"], series["no_consensus"]["f1"])
    if series.get("full", {}).get("f1") and series.get("no_classifier", {}).get("f1"):
        out["significance"]["learned_vs_threshold_f1"] = paired_test(
            series["full"]["f1"], series["no_classifier"]["f1"])
    return out


def main():
    agg = {
        "experiment": "cognitive-edge-byzantine-bdcc",
        "seeds": SEEDS,
        "hardware": {
            "cpu": "AMD Ryzen 9 5900X 12-Core (24 threads)",
            "ram_gb": 64,
            "gpu": "none (CPU-only inference)",
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "cognitive": aggregate_cognitive(),
        "latency": aggregate_latency(),
        "consensus": aggregate_consensus(),
        "ablation": aggregate_ablation(),
    }
    with open(os.path.join(RES, "aggregated_results.json"), "w") as f:
        json.dump(agg, f, indent=2)

    if agg["cognitive"]:
        c = agg["cognitive"]
        b = c["macro_f1_bootstrap_ci95"]
        print(f"COGNITIVE macro-F1 = {c['macro_f1']['mean']:.4f} "
              f"(boot 95% CI [{b['lo']:.4f},{b['hi']:.4f}]) "
              f"| feats={c['feature_names']} | params={c['n_params']}")
    if agg["latency"]:
        for k, v in agg["latency"]["by_config"].items():
            flag = " [>5ms]" if v["exceeds_5ms_budget"] else ""
            print(f"LAT {k}: e2e median={v['median_ms']['mean']:.3f}ms "
                  f"(net={v['network_latency_ms']:.2f}) p99~"
                  f"{v['p99_ms']['mean']:.3f}ms{flag}")
    if agg["consensus"]:
        for frac, v in agg["consensus"]["representative_n7_equivocate"].items():
            ib = "in" if v["within_bound"] else "OUT"
            print(f"CONS f/n={frac} (f={v['f_actual']}/bound {v['f_tol']}, {ib}-bound) "
                  f"PBFT safety={v['pbft_safety']['mean']*100:.1f}% "
                  f"acc={v['pbft_accuracy']['mean']*100:.1f}% | "
                  f"Raft safety={v['raft_safety']['mean']*100:.1f}% "
                  f"acc={v['raft_accuracy']['mean']*100:.1f}%")
    print("-> results/aggregated_results.json written")


if __name__ == "__main__":
    main()
