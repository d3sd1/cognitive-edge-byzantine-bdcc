"""
Synthetic high-rate benchmark stream generator (explicitly labeled SYNTHETIC).

Emits 80+ heterogeneous channels at mixed native rates whose AGGREGATE sum
exceeds 10,000 samples/s (not 10k per channel). Realistic value ranges and
noise, occasional quality-flag dropouts. No proprietary signal models are used.

Output: a list of (timestamp, channel_id, value, quality) tuples sorted by time,
or a compact numpy structure for the benchmark. Seeded for reproducibility.
"""
import os
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def channel_specs(cfg):
    """Expand rate_groups into per-channel specs with realistic ranges."""
    range_by_kind = {
        "imu":            (-30.0, 30.0),     # m/s^2 / rad/s
        "suspension":     (0.0, 120.0),      # mm travel
        "brake_pressure": (0.0, 250.0),      # bar
        "powertrain":     (-50.0, 650.0),    # A / V / Nm mixed
        "battery_cell":   (2.8, 4.2),        # V
        "temperature":    (10.0, 90.0),      # degC
        "gnss":           (-180.0, 180.0),   # deg / m
    }
    specs = []
    cid = 0
    for grp in cfg["rate_groups"]:
        lo, hi = range_by_kind[grp["kind"]]
        for _ in range(grp["count"]):
            specs.append({"id": cid, "hz": grp["hz"], "kind": grp["kind"],
                          "lo": lo, "hi": hi})
            cid += 1
    return specs


def generate(cfg, seed):
    rng = np.random.RandomState(seed)
    specs = channel_specs(cfg)
    dur = cfg["duration_s"]
    noise = cfg["noise_level"]
    drop = cfg["dropout_prob"]

    events = []  # (t, cid, value, quality)
    agg = 0.0
    for s in specs:
        hz = s["hz"]
        n = int(dur * hz)
        agg += hz
        t = np.arange(n) / hz
        # base signal: smooth sinusoidal carrier + slow drift, scaled to range
        span = s["hi"] - s["lo"]
        mid = 0.5 * (s["hi"] + s["lo"])
        f = rng.uniform(0.2, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        base = 0.4 * span * np.sin(2 * np.pi * f * t + phase)
        drift = 0.1 * span * np.sin(2 * np.pi * 0.05 * t)
        val = mid + base + drift + rng.normal(0, noise * span, n)
        val = np.clip(val, s["lo"], s["hi"])
        q = np.ones(n)
        dmask = rng.random(n) < drop
        q[dmask] = 0.0
        for i in range(n):
            events.append((t[i], s["id"], float(val[i]), float(q[i])))

    events.sort(key=lambda e: e[0])
    meta = {
        "seed": seed,
        "n_channels": len(specs),
        "duration_s": dur,
        "aggregate_rate_sps": float(agg),
        "n_samples": len(events),
        "channels_by_kind": {k: int(sum(1 for s in specs if s["kind"] == k))
                             for k in set(s["kind"] for s in specs)},
    }
    return events, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)["synthetic_stream"]
    events, meta = generate(cfg, args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
