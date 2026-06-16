"""
Ingest -> consensus -> decision latency / throughput benchmark.

The stream is SYNTHETIC (generate_synthetic_stream.py); the MEASUREMENT is real:
we run the actual ingest normalization + the REAL PBFT / Raft consensus round
(run_consensus_sim.py) against the synthetic samples and time each decision
epoch, ADDING a modelled per-hop network latency for the protocol's critical
message path.

End-to-end latency per decision epoch =
    t_ingest_compute            (real, measured: windowed normalization)
  + t_consensus_compute         (real, measured: PBFT/Raft round execution)
  + network_latency             (modelled: sequential_hops * link_latency_ms)

The network term is the dominant cost the old benchmark ignored. PBFT pays
3 sequential one-way message delays (PRE-PREPARE -> PREPARE -> COMMIT), Raft
pays 2 (AppendEntries -> ack). The link latency is configurable in config.json
(`benchmark.link_latency_ms`) and documented there for an in-vehicle local
network (CAN-FD / switched automotive Ethernet). The resulting latency is
LARGER than a pure-compute number -- this is correct and honest.

Reports mean / median / p95 / p99 latency (ms) and sustained throughput, swept
over n in {4,7,10}. Honestly flags any case that exceeds the 5 ms budget.
"""
import os
import json
import time
import argparse
import numpy as np

from generate_synthetic_stream import generate, channel_specs
from run_consensus_sim import run_pbft_round, run_raft_round

HERE = os.path.dirname(os.path.abspath(__file__))


def run(seed, cfg_path):
    with open(cfg_path) as f:
        cfg_all = json.load(f)
    scfg = cfg_all["synthetic_stream"]
    bcfg = cfg_all["benchmark"]

    events, smeta = generate(scfg, seed)
    specs = channel_specs(scfg)
    n_ch = len(specs)
    cadence = bcfg["decision_cadence_hz"]
    dt = 1.0 / cadence
    f_byz = bcfg["byzantine_f"]
    link_ms = bcfg["link_latency_ms"]
    hops = {"pbft": bcfg["pbft_sequential_hops"],
            "raft": bcfg["raft_sequential_hops"]}

    ev = np.array([(t, c, v, q) for (t, c, v, q) in events], dtype=float)
    t_all = ev[:, 0]
    dur = scfg["duration_s"]
    n_epochs = int(dur * cadence)

    rng = np.random.RandomState(seed)
    results = {}
    for protocol in bcfg["protocols"]:
        net_ms = hops[protocol] * link_ms
        for n_nodes in bcfg["n_nodes_sweep"]:
            f_tol = (n_nodes - 1) // 3
            byz_idx = list(range(min(f_byz, f_tol)))
            compute_lat = []
            last = np.zeros(n_ch)
            qual = np.ones(n_ch)
            ptr = 0
            stale = np.zeros(n_ch)
            # warmup
            for w in range(bcfg["warmup_epochs"]):
                ptr, _ = _epoch(ev, t_all, w * dt, dt, last, qual, n_ch,
                                n_nodes, f_tol, byz_idx, protocol, stale, rng,
                                ptr)
            ptr = 0
            for e in range(n_epochs):
                t_lo = e * dt
                t0 = time.perf_counter()
                ptr, _ = _epoch(ev, t_all, t_lo, dt, last, qual, n_ch,
                                n_nodes, f_tol, byz_idx, protocol, stale, rng,
                                ptr, primary=e % n_nodes)
                compute_lat.append((time.perf_counter() - t0) * 1000.0)
            comp = np.array(compute_lat)
            # end-to-end = real compute + modelled network on critical path
            e2e = comp + net_ms
            wall = comp.sum() / 1000.0  # compute-bound wall time
            thru = smeta["n_samples"] / wall if wall > 0 else 0.0
            key = f"{protocol}_n{n_nodes}"
            results[key] = {
                "protocol": protocol,
                "n_nodes": n_nodes,
                "sequential_hops": hops[protocol],
                "link_latency_ms": link_ms,
                "network_latency_ms": round(net_ms, 4),
                "compute_median_ms": float(np.median(comp)),
                "compute_mean_ms": float(np.mean(comp)),
                "median_ms": float(np.median(e2e)),
                "mean_ms": float(np.mean(e2e)),
                "p95_ms": float(np.percentile(e2e, 95)),
                "p99_ms": float(np.percentile(e2e, 99)),
                "std_ms": float(np.std(e2e)),
                "exceeds_5ms_budget": bool(np.percentile(e2e, 99) > 5.0),
                "sustained_throughput_sps": float(thru),
                "channels": n_ch,
                "aggregate_input_rate_sps": smeta["aggregate_rate_sps"],
                "n_epochs": n_epochs,
            }

    out = {"seed": seed, "results": results,
           "n_channels": n_ch,
           "link_latency_ms": link_ms,
           "aggregate_rate_sps": smeta["aggregate_rate_sps"]}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"latency_seed_{seed}.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    return out


def _epoch(ev, t_all, t_lo, dt, last, qual, n_ch, n_nodes, f_tol, byz_idx,
           protocol, stale, rng, ptr, primary=0):
    """One ingest+consensus epoch; updates last/qual in place.
    Returns (new ptr, decided_values)."""
    t_hi = t_lo + dt
    n = len(t_all)
    while ptr < n and t_all[ptr] < t_lo:
        ptr += 1
    j = ptr
    while j < n and t_all[j] < t_hi:
        c = int(ev[j, 1])
        last[c] = ev[j, 2]
        qual[c] = ev[j, 3]
        j += 1
    truth = last.copy()
    if protocol == "pbft":
        decisions, _ = run_pbft_round(n_nodes, f_tol, byz_idx, "equivocate",
                                      truth, stale, rng, primary)
    else:
        decisions, _ = run_raft_round(n_nodes, f_tol, byz_idx, "equivocate",
                                      truth, stale, rng, primary)
    stale[:] = truth
    return j, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    out = run(args.seed, args.config)
    for k, r in out["results"].items():
        flag = " [>5ms!]" if r["exceeds_5ms_budget"] else ""
        print(f"{k}: e2e median={r['median_ms']:.3f}ms p99={r['p99_ms']:.3f}ms "
              f"(net={r['network_latency_ms']:.2f} compute={r['compute_median_ms']:.3f}) "
              f"thru={r['sustained_throughput_sps']:.0f} sps{flag}")


if __name__ == "__main__":
    main()
