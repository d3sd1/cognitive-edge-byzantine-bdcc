# Reproducibility package — Hierarchical Cognitive Edge Architecture for Byzantine-Resilient Big Data Telemetry

Reproducibility code for the manuscript *"A Hierarchical Cognitive Edge
Architecture for Byzantine-Resilient Big Data Telemetry in High-Performance
Racing Motorcycles"* (submitted to MDPI *Big Data and Cognitive Computing*).

The evaluation is split into **three deliberately separated planes of evidence**,
each labeled honestly as real, synthetic, or simulated:

| Plane | Tier | Data type | Script |
|-------|------|-----------|--------|
| Cognitive classification | Cognitive | **Real** MotoStudent Electric telemetry (Albacete) | `train_cognitive.py` |
| Throughput / latency | Ingest + Consensus | **Synthetic** 88-channel, >10k samples/s aggregate benchmark (compute measured, network modelled) | `generate_synthetic_stream.py` + `benchmark_latency.py` |
| Byzantine resilience | Consensus | **Simulated** fault injection (5 seeds) | `run_consensus_sim.py` |

No result borrows credibility from another plane. The consensus tier uses only
**public, published protocols** implemented as genuine message-passing state
machines: PBFT (three-phase pre-prepare/prepare/commit with a 2f+1 quorum
certificate and a rotating primary) and a Raft crash-fault baseline. No
proprietary or patent-pending agreement algorithm is included.

Key methodological points (addressing common reviewer concerns):

- **Real PBFT, not a robust aggregator.** `run_consensus_sim.py` exchanges
  explicit message objects, has O(n^2) message complexity, and decides only on a
  commit certificate. Faulty nodes perform true per-recipient equivocation
  (different values to different receivers), plus silent, stale, and an adaptive
  boundary adversary. We report **safety** (honest replicas deciding the same
  value) separately from **estimation accuracy** (decided value matches the
  fault-free truth), and sweep the fault fraction past the 1/3 bound so the real
  collapse is visible.
- **Network-aware latency.** `benchmark_latency.py` measures real ingest +
  consensus compute and adds a modelled per-hop link latency on the protocol's
  critical path (PBFT = 3 hops, Raft = 2 hops; `link_latency_ms` in
  `config.json`). End-to-end latency is therefore millisecond-scale, not the
  pure-compute sub-0.2 ms of a non-networked benchmark.
- **No circular cognition.** The classifier is trained only on channels that are
  **independent** of the labeling rule; the channels used to define the labels
  (throttle, battery current, motor speed -> speed/accel, power) are withheld.
- **Statistics.** `stats_utils.py` provides 95% bootstrap/t confidence
  intervals, paired significance tests, and Cohen's d_z effect sizes.

## Requirements

- Python 3.11+ (tested on 3.13.7)
- CPU-only is sufficient (no GPU required). Reference host: AMD Ryzen 9 5900X
  (12C/24T), 64 GB RAM, Windows 11.

## Setup

```bash
python -m venv .venv
# Windows:
.venv/Scripts/activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Real data (cognitive plane)

The cognitive classifier is trained on **MotoStudent Electric** powertrain/
electrical telemetry recorded at the Circuito de Albacete (~10 Hz, 13 channels,
~4760 s across 3 sessions). The dataset does not contain IMU/lean channels, so
dynamic states (`braking`, `cornering`, `accel`, `cruise`) are labeled from
physically grounded rules on speed, longitudinal acceleration, throttle and
battery current; the exact rule set is in `dataio.py` and `config.json`. To
avoid circular evaluation, those label-defining channels (and motor speed /
power, which determine them) are **excluded** from the classifier input; the
model uses only the six independent channels listed under `feature_channels` in
`config.json`.

The raw `.xlsm` session files are **not redistributed in this repository**
(student-competition data; redistribution subject to the data owners). They are
expected under `datasets/dataset_motostudent_electric/`. The labeling and
preprocessing code is fully published here so the pipeline is reproducible once
the data is available.

## Run everything (one command)

```bash
python run_all.py
```

This runs all 5 seeds (42, 123, 456, 789, 1024) for the cognitive classifier,
consensus simulation, latency benchmark and ablation, then aggregates results
and regenerates all figures.

Outputs:
- `results/aggregated_results.json` — mean ± std over 5 seeds
- `figures/*.pdf` — all paper figures
- `logs/experiment_<timestamp>.log`

**Expected runtime:** ~8.6 minutes on the reference host (CPU-only).

## Individual stages

```bash
python train_cognitive.py     --seed 42   # real-data classifier
python generate_synthetic_stream.py --seed 42   # synthetic stream metadata
python benchmark_latency.py   --seed 42   # ingest->consensus latency
python run_consensus_sim.py   --seed 42   # Byzantine fault injection sweep
python ablation.py            --seed 42   # tier-contribution ablation
python evaluate.py                        # aggregate across seeds
python generate_figures.py                # regenerate figures
```

## Files

| File | Purpose |
|------|---------|
| `config.json` | All hyperparameters (centralized) |
| `requirements.txt` | Pinned dependencies |
| `dataio.py` | MotoStudent loading + physically-grounded state labeling |
| `train_cognitive.py` | Lightweight 1D-CNN classifier (real data) |
| `generate_synthetic_stream.py` | Synthetic 88-channel high-rate stream |
| `run_consensus_sim.py` | Genuine PBFT / Raft message-passing state machines under Byzantine faults |
| `benchmark_latency.py` | Ingest→consensus→decision latency (real compute + modelled network) |
| `ablation.py` | Per-component ablation study |
| `stats_utils.py` | Bootstrap/t CIs, paired tests, effect sizes |
| `evaluate.py` | Aggregation (CIs, significance, effect sizes) |
| `generate_figures.py` | Publishable PDF figures |
| `run_all.py` | One-command orchestrator |

## Data availability

- Synthetic benchmark stream + all code: this repository (MIT license) and
  archived with a permanent DOI (Zenodo; see manuscript Data Availability).
- Real MotoStudent telemetry: described in full in the manuscript; raw
  redistribution subject to the data owners' permission.
- Raw production telemetry from the broader research platform is **not** released
  (intellectual-property / pending-patent constraints).

## License

MIT — see `LICENSE`.
