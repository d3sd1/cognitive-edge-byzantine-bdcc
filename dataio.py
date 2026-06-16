"""
Data loading and physically-grounded state labeling for the cognitive tier.

Real data: MotoStudent Electric telemetry (Circuito de Albacete), powertrain /
electrical channels sampled at ~10 Hz. The dataset does NOT contain IMU/lean or
lateral-acceleration channels, so dynamic states are derived from physically
grounded rules on the available signals (speed, longitudinal acceleration,
throttle, torque, battery current). The cornering state is inferred from the
brake -> apex -> acceleration speed signature characteristic of a corner, which
is a recognized signal-only heuristic when lateral acceleration is unavailable.
All thresholds are centralized in config.json and published with the code.

States: 0=braking, 1=cornering, 2=accel, 3=cruise
"""
import os
import json
import numpy as np
import openpyxl

STATE_NAMES = ["braking", "cornering", "accel", "cruise"]


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_session(xlsm_path, channels, sheet):
    """Read raw triplet-organized channels from a MotoStudent Datos sheet."""
    wb = openpyxl.load_workbook(xlsm_path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    data = rows[1:]

    def col(i):
        return np.array(
            [(r[i] if (i < len(r) and isinstance(r[i], (int, float))) else np.nan)
             for r in data], dtype=float)

    series = {}
    for ch in channels:
        c = ch["col"]
        t = col(c)
        v = col(c + 1)
        mask = ~np.isnan(t) & ~np.isnan(v)
        series[ch["name"]] = (t[mask], v[mask])
    return series


def _resample_common(series, names, hz):
    """Resample all channels onto a common uniform time base via interpolation."""
    t0 = max(s[0].min() for s in series.values() if len(s[0]) > 1)
    t1 = min(s[0].max() for s in series.values() if len(s[0]) > 1)
    n = int((t1 - t0) * hz)
    if n < 100:
        return None, None
    tgrid = np.linspace(t0, t1, n)
    out = {}
    for nm in names:
        t, v = series[nm]
        order = np.argsort(t)
        out[nm] = np.interp(tgrid, t[order], v[order])
    return tgrid, out


def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _label_states(tgrid, ch, cfg):
    """Physically grounded state labeling on the resampled channels."""
    lab = cfg["labeling"]
    hz = cfg["resample_hz"]
    rpm_to_kmh = cfg["rpm_to_kmh"]

    speed = ch["motor_rpm"] * rpm_to_kmh           # km/h
    speed = np.clip(speed, 0, None)
    speed_s = _smooth(speed, lab["smooth_window"])
    # longitudinal acceleration in m/s^2 from speed (km/h -> m/s)
    v_ms = speed_s / 3.6
    accel = np.gradient(v_ms, 1.0 / hz)
    accel = _smooth(accel, lab["smooth_window"])

    throttle = ch["throttle"]
    bcur = ch["battery_current"]

    n = len(tgrid)
    labels = np.full(n, -1, dtype=int)

    running = speed_s >= lab["min_run_speed_kmh"]

    # braking: clear longitudinal deceleration with throttle released
    braking = running & (accel <= lab["brake_decel_thresh"]) & (throttle <= lab["throttle_on"])
    # accel: clear longitudinal acceleration with throttle applied
    accelerating = running & (accel >= lab["accel_thresh"]) & (throttle > lab["throttle_on"])
    # cruise: quasi-steady speed at HIGH speed (straights / top-speed running)
    cruise = (running
              & (np.abs(accel) <= lab["cruise_accel_band"])
              & (speed_s >= lab["cruise_min_speed_kmh"]))
    # cornering: quasi-steady speed at MODERATE/low speed (apex phase between a
    # brake event and an acceleration event), below the cruise-speed regime
    cornering = (running
                 & (np.abs(accel) <= lab["cruise_accel_band"])
                 & (speed_s < lab["cornering_max_speed_kmh"])
                 & (speed_s >= lab["min_run_speed_kmh"]))

    # assign with priority: braking, accel, then steady regimes by speed band
    labels[braking] = 0
    labels[accelerating & (labels < 0)] = 2
    labels[cruise & (labels < 0)] = 3
    labels[cornering & (labels < 0)] = 1

    # enforce minimum state run length to remove flicker
    labels = _despeckle(labels, lab["min_state_len"])

    # store derived channels
    derived = {
        "speed_kmh": speed_s,
        "long_accel": accel,
        "power_kw": ch["battery_voltage"] * bcur / 1000.0,
    }
    return labels, derived


def _despeckle(labels, min_len):
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_len and out[i] != -1:
            # merge tiny run into previous valid label if any
            prev = out[i - 1] if i > 0 else -1
            out[i:j] = prev
        i = j
    return out


def build_windows(cfg_path):
    """Returns X (N, W, F), y (N,), groups (N,) for grouped splitting, and meta."""
    cfg_all = _load_config(cfg_path)
    cfg = cfg_all["cognitive"]
    base = os.path.dirname(os.path.abspath(cfg_path))
    dataset_dir = os.path.normpath(os.path.join(base, cfg["dataset_dir"]))

    feat_names = cfg["feature_channels"]
    # label-defining channels are kept separately ONLY for the threshold-rule
    # ablation baseline; they are NEVER fed to the learned classifier.
    label_names = cfg.get("label_defining_channels",
                          ["throttle", "speed_kmh", "long_accel"])
    W = cfg["window"]
    stride = cfg["window_stride"]

    all_X, all_XL, all_y, all_g = [], [], [], []
    seg_counter = 0
    per_session = []
    total_seconds = 0.0

    for sess in cfg["session_files"]:
        path = os.path.join(dataset_dir, sess)
        if not os.path.exists(path):
            continue
        series = _read_session(path, cfg["channels"], cfg["sheet"])
        raw_names = [c["name"] for c in cfg["channels"]]
        tgrid, ch = _resample_common(series, raw_names, cfg["resample_hz"])
        if tgrid is None:
            continue
        labels, derived = _label_states(tgrid, ch, cfg)
        ch.update(derived)

        # build feature matrix (learned classifier inputs)
        feat = np.stack([ch[nm] for nm in feat_names], axis=1)  # (T, F)
        # label-defining channels, kept aside for the threshold-rule baseline
        labfeat = np.stack([ch[nm] for nm in label_names], axis=1)  # (T, FL)
        # per-session z-normalization on valid running region
        valid = labels >= 0
        if valid.sum() < 50:
            continue
        mu = feat[valid].mean(axis=0)
        sd = feat[valid].std(axis=0) + 1e-6
        feat = (feat - mu) / sd
        muL = labfeat[valid].mean(axis=0)
        sdL = labfeat[valid].std(axis=0) + 1e-6
        labfeat = (labfeat - muL) / sdL

        T = feat.shape[0]
        total_seconds += (tgrid[-1] - tgrid[0])

        # segment the session timeline into n_segments contiguous blocks (groups)
        n_seg = cfg["split"]["n_segments"]
        seg_bounds = np.linspace(0, T, n_seg + 1).astype(int)

        sess_windows = 0
        for k in range(W, T, stride):
            sl = slice(k - W, k)
            lab_win = labels[sl]
            if (lab_win < 0).any():
                continue
            # majority label in window
            vals, counts = np.unique(lab_win, return_counts=True)
            y = vals[np.argmax(counts)]
            seg = np.searchsorted(seg_bounds, k, side="right") - 1
            all_X.append(feat[sl])
            all_XL.append(labfeat[sl])
            all_y.append(int(y))
            all_g.append(seg_counter + seg)
            sess_windows += 1
        seg_counter += n_seg
        per_session.append({"session": sess, "duration_s": float(tgrid[-1] - tgrid[0]),
                            "windows": sess_windows})

    X = np.asarray(all_X, dtype=np.float32)
    XL = np.asarray(all_XL, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int64)
    g = np.asarray(all_g, dtype=np.int64)
    meta = {
        "n_windows": int(len(y)),
        "n_features": int(X.shape[2]) if len(X) else 0,
        "window": W,
        "feature_names": feat_names,
        "label_feature_names": label_names,
        "total_seconds": float(total_seconds),
        "total_laps_equiv": None,
        "per_session": per_session,
        "class_counts": {STATE_NAMES[i]: int((y == i).sum()) for i in range(4)},
    }
    return X, XL, y, g, meta


if __name__ == "__main__":
    import sys
    cfgp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "config.json")
    X, XL, y, g, meta = build_windows(cfgp)
    print("X", X.shape, "XL", XL.shape, "y", y.shape, "groups", len(np.unique(g)))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
