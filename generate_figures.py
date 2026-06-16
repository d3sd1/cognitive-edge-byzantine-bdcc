"""
Generate all publishable figures (PDF, serif/Latin-Modern-like, 10pt) from
results/aggregated_results.json.

Figures:
  fig_architecture.pdf : three-tier schematic
  fig_confusion.pdf    : normalized confusion matrix (cognitive)
  fig_latency.pdf      : latency distribution PBFT vs Raft
  fig_byzantine.pdf    : consensus correctness vs fault fraction, 4 behaviors
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
SEEDS = [42, 123, 456, 789, 1024]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

STATES = ["Braking", "Cornering", "Accel.", "Cruise"]
CB = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]  # colorblind-safe


def load_agg():
    with open(os.path.join(RES, "aggregated_results.json")) as f:
        return json.load(f)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    tiers = [("Ingest tier\n(normalize 80+ channels)", 1.6, CB[0]),
             ("Consensus tier\n(PBFT / Raft, Byzantine-resilient)", 5.0, CB[1]),
             ("Cognitive tier\n(edge classifier:\nbraking/cornering/accel/cruise)", 8.4, CB[2])]
    for label, x, c in tiers:
        box = FancyBboxPatch((x - 1.35, 0.9), 2.7, 1.2,
                             boxstyle="round,pad=0.06", linewidth=1.2,
                             edgecolor=c, facecolor=c + "22")
        ax.add_patch(box)
        ax.text(x, 1.5, label, ha="center", va="center", fontsize=8.5)
    for x0, x1 in [(2.95, 3.65), (6.35, 7.05)]:
        ax.add_patch(FancyArrowPatch((x0, 1.5), (x1, 1.5),
                     arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#444"))
    ax.text(0.15, 1.5, "stream\nin", ha="center", va="center", fontsize=8, color="#666")
    ax.text(9.85, 1.5, "decision\nout", ha="center", va="center", fontsize=8, color="#666")
    ax.text(5.0, 2.55, "Trust before cognition", ha="center", fontsize=9, style="italic")
    fig.savefig(os.path.join(FIG, "fig_architecture.pdf"))
    plt.close(fig)


def fig_confusion(agg):
    cm = np.array(agg["cognitive"]["confusion_matrix_norm"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(STATES, rotation=30, ha="right"); ax.set_yticklabels(STATES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(FIG, "fig_confusion.pdf"))
    plt.close(fig)


def fig_latency(agg):
    """End-to-end latency (real compute + modelled network) vs node count,
    PBFT vs Raft, with the 5 ms budget line."""
    lat = agg["latency"]
    by = lat["by_config"]
    ns = sorted(set(v["n_nodes"] for v in by.values()))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    width = 0.36
    xpos = np.arange(len(ns))
    for k, (proto, name, c) in enumerate(
            [("pbft", "PBFT (3 hops)", CB[1]), ("raft", "Raft (2 hops)", CB[0])]):
        med = [by[f"{proto}_n{n}"]["median_ms"]["mean"] for n in ns]
        p99 = [by[f"{proto}_n{n}"]["p99_ms"]["mean"] for n in ns]
        err = [by[f"{proto}_n{n}"]["median_ms"]["std"] for n in ns]
        ax.bar(xpos + (k - 0.5) * width, med, width, yerr=err, capsize=3,
               label=f"{name} median", color=c, edgecolor="black", linewidth=0.5)
        ax.plot(xpos + (k - 0.5) * width, p99, "D", color="black", ms=4,
                label=f"{name} p99" if k == 0 else None)
    ax.axhline(5.0, color="red", ls="--", lw=1, label="5 ms budget")
    ax.set_xticks(xpos); ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("Ingest-to-decision latency (ms)")
    ax.set_title(f"End-to-end latency, synthetic {lat['n_channels']}-ch stream "
                 f"({lat['aggregate_rate_sps']:.0f} sps), link={lat['link_latency_ms']} ms/hop",
                 fontsize=8.5)
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig_latency.pdf"))
    plt.close(fig)


def fig_byzantine(agg):
    """PBFT vs Raft SAFETY (consensus agreement) vs fault fraction, four
    behaviours, crossing and exceeding the f/n = 1/3 BFT threshold. PBFT keeps
    safety = 1 inside the bound and collapses only when f exceeds floor((n-1)/3).
    """
    sweep = agg["consensus"]["sweep"]
    behaviors = ["silent", "stale", "equivocate", "adaptive"]
    titles = {"silent": "Silent omission", "stale": "Stale replay",
              "equivocate": "Per-recipient equivocation",
              "adaptive": "Adaptive boundary adversary"}
    n_nodes = 7
    fracs = sorted(set(v["fault_fraction"] for v in sweep.values()
                       if v["n_nodes"] == n_nodes))
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True, sharey=True)
    for ax, beh in zip(axes.flat, behaviors):
        pb_m, pb_s, rf_m, rf_s = [], [], [], []
        for f in fracs:
            cell = sweep[f"n{n_nodes}_f{f:.2f}_{beh}"]
            pb_m.append(cell["pbft_safety"]["mean"] * 100)
            pb_s.append(cell["pbft_safety"]["std"] * 100)
            rf_m.append(cell["raft_safety"]["mean"] * 100)
            rf_s.append(cell["raft_safety"]["std"] * 100)
        fr = np.array(fracs)
        ax.plot(fr, pb_m, "-o", color=CB[1], label="PBFT", ms=4)
        ax.fill_between(fr, np.array(pb_m) - np.array(pb_s),
                        np.array(pb_m) + np.array(pb_s), color=CB[1], alpha=0.2)
        ax.plot(fr, rf_m, "-s", color=CB[0], label="Raft", ms=4)
        ax.fill_between(fr, np.array(rf_m) - np.array(rf_s),
                        np.array(rf_m) + np.array(rf_s), color=CB[0], alpha=0.2)
        ax.axvline(1 / 3, color="gray", ls=":", lw=1)
        ax.text(0.335, 12, "$f/n=1/3$", rotation=90, fontsize=7, color="gray",
                ha="right", va="bottom")
        ax.set_title(titles[beh], fontsize=9)
        ax.set_ylim(-3, 104); ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Byzantine fault fraction $f/n$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Consensus safety (agreement, %)")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_byzantine.pdf"))
    plt.close(fig)


def fig_byzantine_accuracy(agg):
    """PBFT vs Raft estimation ACCURACY (decided value matches ground truth)
    vs fault fraction at n=7, equivocation -- a property of the estimator, not
    of agreement. Shown separately from safety (FIX: the two were conflated)."""
    sweep = agg["consensus"]["sweep"]
    n_nodes = 7
    beh = "equivocate"
    fracs = sorted(set(v["fault_fraction"] for v in sweep.values()
                       if v["n_nodes"] == n_nodes))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    pb = [sweep[f"n{n_nodes}_f{f:.2f}_{beh}"]["pbft_accuracy"]["mean"] * 100 for f in fracs]
    rf = [sweep[f"n{n_nodes}_f{f:.2f}_{beh}"]["raft_accuracy"]["mean"] * 100 for f in fracs]
    fr = np.array(fracs)
    ax.plot(fr, pb, "-o", color=CB[1], label="PBFT", ms=4)
    ax.plot(fr, rf, "-s", color=CB[0], label="Raft", ms=4)
    ax.axvline(1 / 3, color="gray", ls=":", lw=1)
    ax.text(0.335, 5, "$f/n=1/3$", rotation=90, fontsize=8, color="gray",
            ha="right", va="bottom")
    ax.set_xlabel("Byzantine fault fraction $f/n$")
    ax.set_ylabel("Estimation accuracy (%)")
    ax.set_title("Decided-value accuracy vs ground truth ($n{=}7$, equivocation)",
                 fontsize=9)
    ax.set_ylim(-3, 104); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_byzantine_accuracy.pdf"))
    plt.close(fig)


def fig_messages(agg):
    """PBFT O(n^2) vs Raft O(n) message complexity vs node count."""
    sweep = agg["consensus"]["sweep"]
    ns = sorted(set(v["n_nodes"] for v in sweep.values()))
    pb = [sweep[f"n{n}_f0.00_equivocate"]["pbft_avg_messages"]["mean"] for n in ns]
    rf = [sweep[f"n{n}_f0.00_equivocate"]["raft_avg_messages"]["mean"] for n in ns]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ns, pb, "-o", color=CB[1], label="PBFT ($O(n^2)$)", ms=5)
    ax.plot(ns, rf, "-s", color=CB[0], label="Raft ($O(n)$)", ms=5)
    ax.set_xlabel("Number of replicas $n$")
    ax.set_ylabel("Messages per consensus round")
    ax.set_xticks(ns); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_messages.pdf"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    agg = load_agg()
    fig_architecture()
    if agg.get("cognitive"):
        fig_confusion(agg)
    if agg.get("latency"):
        fig_latency(agg)
    if agg.get("consensus"):
        fig_byzantine(agg)
        fig_byzantine_accuracy(agg)
        fig_messages(agg)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
