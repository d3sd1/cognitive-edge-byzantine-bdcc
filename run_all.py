"""
One-command end-to-end reproduction of all experiments.

Runs, for each of the 5 fixed seeds (42, 123, 456, 789, 1024):
  1. cognitive classifier training/eval on real MotoStudent data
  2. Byzantine consensus simulation (sweep nodes x fault fraction x behavior)
  3. latency/throughput benchmark on the synthetic stream
  4. ablation study
Then aggregates results and generates all figures.

Usage:  python run_all.py
"""
import os
import sys
import time
import json
import logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_cognitive
import run_consensus_sim
import benchmark_latency
import ablation
import evaluate
import generate_figures

SEEDS = [42, 123, 456, 789, 1024]
CFG = os.path.join(HERE, "config.json")


def main():
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(HERE, "logs", f"experiment_{ts}.log")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()])

    t_start = time.time()
    for s in SEEDS:
        logging.info("=== SEED %d ===", s)
        t0 = time.time()
        train_cognitive.run(s, CFG)
        logging.info("  cognitive done (%.1fs)", time.time() - t0)
        t0 = time.time()
        run_consensus_sim.run(s, CFG)
        logging.info("  consensus done (%.1fs)", time.time() - t0)
        t0 = time.time()
        benchmark_latency.run(s, CFG)
        logging.info("  latency done (%.1fs)", time.time() - t0)
        t0 = time.time()
        ablation.run(s, CFG)
        logging.info("  ablation done (%.1fs)", time.time() - t0)

    logging.info("Aggregating results...")
    evaluate.main()
    logging.info("Generating figures...")
    generate_figures.main()

    total = time.time() - t_start
    logging.info("ALL DONE in %.1f s (%.1f min)", total, total / 60.0)
    # record runtime into aggregated results
    aggp = os.path.join(HERE, "results", "aggregated_results.json")
    with open(aggp) as f:
        agg = json.load(f)
    agg["total_runtime_s"] = round(total, 1)
    agg["total_runtime_min"] = round(total / 60.0, 2)
    with open(aggp, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nTotal wall-clock runtime: {total/60.0:.1f} min")


if __name__ == "__main__":
    main()
