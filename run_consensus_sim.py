"""
Byzantine consensus simulation (explicitly labeled SIMULATION).

This module implements GENUINE, public-literature consensus protocols as
message-passing state machines over n in-process replicas:

  - PBFT (Castro & Liskov, 1999): a three-phase agreement protocol
    (PRE-PREPARE / PREPARE / COMMIT) driven by a primary (leader) for the
    current view, with an explicit quorum of 2f+1 and a quorum certificate.
    Replicas exchange explicit Message objects; message complexity is O(n^2)
    in the PREPARE/COMMIT phases. A replica DECIDES a value only when it
    collects a commit certificate (>= 2f+1 matching COMMIT messages).
    Tolerated bound: f_tol = floor((n-1)/3); PBFT requires n >= 3*f_tol + 1.

  - Raft (Ongaro & Ousterhout, 2014): a crash-fault-tolerant leader-based
    protocol. The leader replicates a single value; followers append it
    without any cross-checking. Raft has NO defense against equivocation or
    fabrication (a Byzantine leader or follower can corrupt the decision),
    which is exactly what this study quantifies. Quorum 2f+1 (crash), but no
    Byzantine validation.

NOTE: these are generic, public-domain protocols. This file deliberately does
NOT implement any proprietary or patent-pending agreement algorithm.

Fault behaviors (Byzantine nodes):
  - silent    : node sends nothing (omission).
  - stale     : node sends a stale value from a previous round.
  - equivocate: TRUE equivocation -- the node sends DIFFERENT values to
                different recipients (per-recipient divergence), attempting
                to split honest replicas into disagreeing quorums.
  - adaptive  : worst-case adversary that places its value right at the
                decision boundary (just inside the quorum the honest nodes
                would form) to maximise the chance of a safety violation.

Two DISTINCT metrics are reported (the old code conflated them):
  (a) safety / agreement  : fraction of rounds in which ALL deciding honest
      replicas decide the SAME value (consensus safety property). For a
      correct PBFT this is 1.0 whenever actual_faulty <= floor((n-1)/3), and
      it only degrades when the fault count EXCEEDS the BFT bound.
  (b) estimation accuracy : fraction of rounds in which the decided value
      equals the fault-free ground-truth value (a property of the value the
      protocol agrees ON, not of agreement itself). The sweep crosses and
      EXCEEDS the f/n = 1/3 threshold so the real collapse is visible.
"""
import os
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
class Message:
    """An explicit protocol message exchanged between replicas."""
    __slots__ = ("phase", "view", "seq", "sender", "recipient", "value", "digest")

    def __init__(self, phase, view, seq, sender, recipient, value):
        self.phase = phase            # 'pre-prepare' | 'prepare' | 'commit'
        self.view = view
        self.seq = seq
        self.sender = sender
        self.recipient = recipient    # None = broadcast target id
        self.value = value            # np.ndarray or None
        self.digest = _digest(value)


def _digest(value):
    """Order-/precision-stable digest of a proposed value vector."""
    if value is None:
        return None
    v = np.asarray(value, dtype=float)
    # quantize so that honest near-identical proposals share a digest
    return tuple(np.round(v, 3).tolist())


# --------------------------------------------------------------------------
# PBFT replica state machine
# --------------------------------------------------------------------------
class PBFTReplica:
    """One PBFT replica. Honest replicas follow the protocol exactly."""

    def __init__(self, rid, n, f_tol):
        self.rid = rid
        self.n = n
        self.f_tol = f_tol
        self.quorum = 2 * f_tol + 1          # PBFT prepared/committed quorum
        self.reset_round()

    def reset_round(self):
        self.prepares = {}                   # digest -> set(senders)
        self.commits = {}                    # digest -> set(senders)
        self.preprepare_digest = None
        self.preprepare_value = None
        self.decided_value = None
        self.decided = False


def _committee_message_count(n):
    """O(n^2): pre-prepare (n-1) + prepare (n*(n-1)) + commit (n*(n-1))."""
    return (n - 1) + 2 * n * (n - 1)


def run_pbft_round(n, f_tol, byz_idx, behavior, truth, stale_value, rng,
                   primary):
    """
    One PBFT consensus round over n replicas. Returns:
      decided_values : list[np.ndarray | None] per honest replica (its decision)
      n_messages     : number of protocol messages exchanged this round.

    Honest replicas run real PRE-PREPARE / PREPARE / COMMIT with a 2f+1 quorum
    and only DECIDE on a commit certificate. Byzantine replicas may equivocate
    (per-recipient divergence), go silent, replay stale values, or act
    adaptively at the quorum boundary.
    """
    replicas = [PBFTReplica(i, n, f_tol) for i in range(n)]
    byz = set(byz_idx)
    honest = [i for i in range(n) if i not in byz]
    n_messages = 0

    # ---- value each node will propose / forward -------------------------
    def honest_value():
        return truth + rng.normal(0, 0.002, truth.shape)

    def byz_value_for(recipient):
        """Value a Byzantine node sends TO a specific recipient."""
        if behavior == "silent":
            return None
        if behavior == "stale":
            return stale_value.copy()
        if behavior == "equivocate":
            # per-recipient divergence: split honest replicas. Half get one
            # bogus value, half get another, both far from truth.
            if (recipient % 2) == 0:
                return truth + 5.0
            return truth - 5.0
        if behavior == "adaptive":
            # worst case: sit just at the digest boundary of the honest value
            # so that, combined with other faulty nodes, it can try to forge a
            # competing quorum. Send a value that quantizes DIFFERENTLY per
            # recipient to maximise quorum splitting near the boundary.
            eps = 0.004 * (1 if (recipient % 2 == 0) else -1)
            return truth + eps + (0.0 if recipient in honest else 6.0)
        return truth + 5.0

    # ---- PHASE 1: PRE-PREPARE (primary -> all) --------------------------
    if primary in byz:
        # Byzantine primary equivocates the pre-prepare per recipient.
        pp_values = {}
        for r in range(n):
            if r == primary:
                continue
            pp_values[r] = byz_value_for(r)
            n_messages += 1
    else:
        pv = honest_value()
        pp_values = {r: pv.copy() for r in range(n) if r != primary}
        n_messages += (n - 1)

    # primary's own accepted pre-prepare
    if primary not in byz:
        own_pp = pp_values.get((primary + 1) % n)  # any; all equal for honest
        own_pp = pv
    else:
        own_pp = None

    for r in replicas:
        if r.rid == primary:
            if primary not in byz:
                r.preprepare_value = pv
                r.preprepare_digest = _digest(pv)
            continue
        val = pp_values.get(r.rid)
        if val is not None:
            r.preprepare_value = val
            r.preprepare_digest = _digest(val)

    # ---- PHASE 2: PREPARE (all -> all) ----------------------------------
    # Each replica broadcasts PREPARE for the digest it accepted in pre-prepare.
    prepare_msgs = []   # (sender, recipient, digest)
    for s in range(n):
        if s in byz:
            if behavior == "silent":
                continue
            # Byzantine: send divergent PREPARE digests per recipient
            for rcpt in range(n):
                if rcpt == s:
                    continue
                d = _digest(byz_value_for(rcpt))
                prepare_msgs.append((s, rcpt, d))
                n_messages += 1
        else:
            d = replicas[s].preprepare_digest
            if d is None:
                continue
            for rcpt in range(n):
                if rcpt == s:
                    continue
                prepare_msgs.append((s, rcpt, d))
                n_messages += 1

    for (s, rcpt, d) in prepare_msgs:
        if d is None:
            continue
        replicas[rcpt].prepares.setdefault(d, set()).add(s)
    # a replica also counts its own pre-prepare digest as a self-prepare
    for r in replicas:
        if r.preprepare_digest is not None:
            r.prepares.setdefault(r.preprepare_digest, set()).add(r.rid)

    # A replica is "prepared" on digest d if it has >= quorum prepares for d.
    for r in replicas:
        if r.rid in byz:
            continue
        prepared_digest = None
        for d, senders in r.prepares.items():
            if len(senders) >= r.quorum:
                prepared_digest = d
                break
        r._prepared = prepared_digest

    # ---- PHASE 3: COMMIT (all -> all) -----------------------------------
    commit_msgs = []
    for s in range(n):
        if s in byz:
            if behavior == "silent":
                continue
            for rcpt in range(n):
                if rcpt == s:
                    continue
                d = _digest(byz_value_for(rcpt))
                commit_msgs.append((s, rcpt, d))
                n_messages += 1
        else:
            d = getattr(replicas[s], "_prepared", None)
            if d is None:
                continue
            for rcpt in range(n):
                if rcpt == s:
                    continue
                commit_msgs.append((s, rcpt, d))
                n_messages += 1

    for (s, rcpt, d) in commit_msgs:
        if d is None:
            continue
        replicas[rcpt].commits.setdefault(d, set()).add(s)
    for r in replicas:
        d = getattr(r, "_prepared", None)
        if d is not None:
            r.commits.setdefault(d, set()).add(r.rid)

    # DECIDE on a commit certificate (>= quorum matching COMMITs).
    decided_values = []
    # map digest back to a representative value
    digest_to_value = {}
    for r in replicas:
        if r.preprepare_value is not None:
            digest_to_value[r.preprepare_digest] = r.preprepare_value
    for r in replicas:
        if r.rid in byz:
            continue
        decided = None
        for d, senders in r.commits.items():
            if len(senders) >= r.quorum:
                decided = digest_to_value.get(d, None)
                break
        decided_values.append(decided)

    return decided_values, n_messages


def run_raft_round(n, f_tol, byz_idx, behavior, truth, stale_value, rng,
                   leader):
    """
    One Raft round. The leader's value is replicated to followers, which append
    it WITHOUT validation. Raft tolerates crashes but NOT Byzantine behaviour.
    A Byzantine leader dictates a corrupt value; a crashed leader yields no
    decision (in practice a re-election, modelled here as no-decision).
    Returns (decided_values_per_honest, n_messages).
    """
    byz = set(byz_idx)
    honest = [i for i in range(n) if i not in byz]
    n_messages = 0

    if leader in byz:
        if behavior == "silent":
            # crashed leader -> election fails this round, no decision
            return [None for _ in honest], n_messages
        # Byzantine leader: equivocates per follower (Raft cannot detect it)
        decided = []
        for h in honest:
            if behavior == "stale":
                decided.append(stale_value.copy())
            elif behavior == "equivocate":
                decided.append(truth + (5.0 if (h % 2 == 0) else -5.0))
            elif behavior == "adaptive":
                decided.append(truth + 6.0)
            else:
                decided.append(truth + 5.0)
            n_messages += 1
        return decided, n_messages

    # honest leader: append-only replication, quorum 2f+1 of acks
    val = truth + rng.normal(0, 0.002, truth.shape)
    n_acks = 0
    for f in range(n):
        if f == leader:
            continue
        n_messages += 1            # AppendEntries
        if f in byz and behavior == "silent":
            continue
        n_acks += 1                # ack (Byzantine followers still ack)
        n_messages += 1
    if (n_acks + 1) >= (2 * f_tol + 1):
        return [val.copy() for _ in honest], n_messages
    return [None for _ in honest], n_messages


def simulate(cfg, seed, n_nodes, fault_fraction, behavior, protocol):
    """
    Run n_epochs rounds. Returns a dict with safety and accuracy metrics and
    average message count. `protocol` in {'pbft','raft'}.
    """
    rng = np.random.RandomState(seed if protocol == "pbft" else seed + 100000)
    state_dim = cfg["state_dim"]
    n_epochs = cfg["n_epochs"]
    tol = cfg["match_tolerance"]

    f_actual = int(np.floor(fault_fraction * n_nodes))
    f_tol = (n_nodes - 1) // 3                  # PBFT-tolerable bound
    byz_idx = list(range(f_actual))

    agree_rounds = 0
    accurate_rounds = 0
    decided_rounds = 0
    msg_total = 0
    stale_value = np.zeros(state_dim)

    for e in range(n_epochs):
        truth = rng.normal(0, 1, state_dim)
        # rotate primary/leader across all nodes (views)
        primary = e % n_nodes
        if protocol == "pbft":
            decisions, nmsg = run_pbft_round(
                n_nodes, f_tol, byz_idx, behavior, truth, stale_value, rng,
                primary)
        else:
            decisions, nmsg = run_raft_round(
                n_nodes, f_tol, byz_idx, behavior, truth, stale_value, rng,
                primary)
        stale_value = truth
        msg_total += nmsg

        decided = [d for d in decisions if d is not None]
        if decided:
            decided_rounds += 1
            # safety: all deciding honest replicas decided the SAME value
            ref = decided[0]
            same = all(np.linalg.norm(d - ref) <= tol for d in decided)
            if same:
                agree_rounds += 1
            # accuracy: the decided value matches fault-free ground truth
            if same and np.linalg.norm(ref - truth) <= tol:
                accurate_rounds += 1
        else:
            # no decision this round: agreement vacuously holds (no split),
            # but accuracy does not (no correct value produced).
            agree_rounds += 1

    return {
        "safety": agree_rounds / n_epochs,
        "accuracy": accurate_rounds / n_epochs,
        "liveness": decided_rounds / n_epochs,
        "avg_messages": msg_total / n_epochs,
        "f_actual": f_actual,
        "f_tol": f_tol,
        "within_bft_bound": f_actual <= f_tol,
    }


def run(seed, cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)["consensus_sim"]
    results = {"seed": seed, "sweep": []}
    for n_nodes in cfg["n_nodes_sweep"]:
        for frac in cfg["fault_fractions"]:
            for behavior in cfg["fault_behaviors"]:
                pbft = simulate(cfg, seed, n_nodes, frac, behavior, "pbft")
                raft = simulate(cfg, seed, n_nodes, frac, behavior, "raft")
                results["sweep"].append({
                    "n_nodes": n_nodes, "fault_fraction": frac,
                    "behavior": behavior,
                    "f_actual": pbft["f_actual"], "f_tol": pbft["f_tol"],
                    "within_bft_bound": pbft["within_bft_bound"],
                    "pbft_safety": round(pbft["safety"], 6),
                    "pbft_accuracy": round(pbft["accuracy"], 6),
                    "pbft_liveness": round(pbft["liveness"], 6),
                    "pbft_avg_messages": round(pbft["avg_messages"], 2),
                    "raft_safety": round(raft["safety"], 6),
                    "raft_accuracy": round(raft["accuracy"], 6),
                    "raft_liveness": round(raft["liveness"], 6),
                    "raft_avg_messages": round(raft["avg_messages"], 2),
                })
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"consensus_seed_{seed}.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    res = run(args.seed, args.config)
    for row in res["sweep"]:
        if row["n_nodes"] == 7 and row["behavior"] == "equivocate":
            print(f"n=7 f={row['f_actual']} (bound {row['f_tol']}) "
                  f"f/n={row['fault_fraction']:.2f} | "
                  f"PBFT safety={row['pbft_safety']:.3f} acc={row['pbft_accuracy']:.3f} | "
                  f"Raft safety={row['raft_safety']:.3f} acc={row['raft_accuracy']:.3f}")


if __name__ == "__main__":
    main()
