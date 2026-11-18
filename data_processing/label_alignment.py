"""Per-observation labels: assign every GPS point to an arc of its reference route.

The .route file lists the arcs the vehicle drove, in order, but it has far
fewer rows than the .track file, so labels per GPS point must be derived.

Method: monotone alignment solved with dynamic programming (Viterbi-style).
For GPS points i = 0..N-1 we choose route positions j_0 <= j_1 <= ... so that
  * positions never go backwards along the route (chronological order),
  * between consecutive points the route distance skipped is plausible for
    the time elapsed (max(MIN_SKIP_M, MAX_SPEED * dt)); after a long gap
    (dt > LONG_GAP_S) any forward jump is allowed,
and the total cost   sum_i min(dist(p_i, arc_j_i), DIST_CAP)
                   + SKIP_PENALTY * metres of route skipped
is minimal. Points whose chosen arc is further than UNALIGNED_M away get
true_arc_id = -1 (e.g. the vehicle parked off the road network).

Run from the repo root:
    python -m data_processing.label_alignment
Outputs:
    data/processed/obs.parquet          one row per GPS observation
    results/eda/label_summary.csv       per-trip label quality
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loader import list_trips, load_trip
from utils.geo import point_segment_distance

DIST_CAP = 100.0      # m, caps the emission cost so outliers don't dominate
UNALIGNED_M = 50.0    # m, beyond this the point gets label -1
MAX_SPEED = 50.0      # m/s, generous upper bound on vehicle speed
MIN_SKIP_M = 150.0    # m, minimum allowed skip per step (GPS noise, short arcs)
LONG_GAP_S = 60.0     # s, after a gap this long any forward jump is allowed
SKIP_PENALTY = 0.02   # cost per metre of route skipped (prefers staying put)

OUT_OBS = Path("data/processed/obs.parquet")
OUT_SUMMARY = Path("results/eda/label_summary.csv")


def align_trip(trip):
    """Return (route_idx per obs, distance to that arc per obs)."""
    tr = trip.track
    ra = trip.arcs.iloc[trip.route["arc_id"].to_numpy()]
    n, m = len(tr), len(ra)
    if m == 0:
        return np.full(n, -1), np.full(n, np.nan)

    seg_len = ra["length_m"].to_numpy()
    cum_start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    cum_end = cum_start + seg_len
    ax, ay, bx, by = (ra[c].to_numpy() for c in ("ax", "ay", "bx", "by"))
    px, py, t = tr["x"].to_numpy(), tr["y"].to_numpy(), tr["t"].to_numpy()

    def emission(i):
        d, _ = point_segment_distance(px[i], py[i], ax, ay, bx, by)
        return d

    back = np.zeros((n, m), dtype=np.int16 if m < 32767 else np.int32)  # offset to previous position
    cost = np.minimum(emission(0), DIST_CAP)  # free start anywhere on the route
    j_idx = np.arange(m)

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        new = np.full(m, np.inf)
        off = np.zeros(m, dtype=back.dtype)
        if dt > LONG_GAP_S:
            # any forward jump allowed: prefix minimum of previous cost
            run_min = np.minimum.accumulate(cost)
            arg = np.zeros(m, dtype=np.int64)
            best = 0
            for j in range(m):  # m is at most a few thousand; long gaps are rare
                if cost[j] < cost[best]:
                    best = j
                arg[j] = best
            new, off = run_min, (j_idx - arg).astype(back.dtype)
        else:
            max_skip = max(MIN_SKIP_M, MAX_SPEED * dt)
            lo = np.searchsorted(cum_end, cum_start - max_skip, side="left")
            w = int((j_idx - lo).max())
            for o in range(w + 1):
                src = j_idx - o
                ok = src >= lo
                if not ok.any():
                    continue
                s = np.where(ok, src, 0)
                skip = np.maximum(cum_start - cum_end[s], 0.0) if o > 0 else 0.0
                cand = np.where(ok, cost[s] + SKIP_PENALTY * skip, np.inf)
                better = cand < new
                new = np.where(better, cand, new)
                off = np.where(better, o, off)
        back[i] = off
        cost = new + np.minimum(emission(i), DIST_CAP)

    # backtrack from the best final position (free end)
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(n - 1, 0, -1):
        path[i - 1] = path[i] - back[i, path[i]]

    dist, _ = point_segment_distance(px, py, ax[path], ay[path], bx[path], by[path])
    return path, dist


def label_trip(trip):
    path, dist = align_trip(trip)
    route_arcs = trip.route["arc_id"].to_numpy()
    aligned = dist <= UNALIGNED_M
    df = trip.track[["obs_idx", "t", "lon", "lat", "x", "y"]].copy()
    df.insert(0, "trip_id", trip.trip_id)
    df.insert(0, "split", trip.split)
    df["route_idx"] = np.where(aligned, path, -1)
    df["true_arc_id"] = np.where(aligned, route_arcs[path] if len(route_arcs) else -1, -1)
    df["dist_to_true_m"] = np.where(aligned, dist, np.nan)

    visited = np.unique(path[aligned])
    steps = np.diff(path)
    summary = {
        "split": trip.split,
        "trip_id": trip.trip_id,
        "n_obs": len(df),
        "n_route_arcs": len(route_arcs),
        "unaligned_pct": float((~aligned).mean() * 100),
        "route_arcs_with_points_pct": float(len(visited) / max(len(route_arcs), 1) * 100),
        "median_dist_m": float(np.nanmedian(df["dist_to_true_m"])) if aligned.any() else np.nan,
        "p95_dist_m": float(np.nanpercentile(df["dist_to_true_m"], 95)) if aligned.any() else np.nan,
        "max_route_step": int(steps.max()) if len(steps) else 0,
    }
    return df, summary


def main():
    OUT_OBS.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    frames, rows = [], []
    t0 = time.time()
    for split, tid in list_trips():
        df, s = label_trip(load_trip(tid, split))
        frames.append(df)
        rows.append(s)
        print(f"{split:12s} {tid}  n={s['n_obs']:6d}  unaligned={s['unaligned_pct']:5.1f}%  "
              f"route covered={s['route_arcs_with_points_pct']:5.1f}%  ({time.time() - t0:.0f}s)")
    obs = pd.concat(frames, ignore_index=True)
    obs.to_parquet(OUT_OBS, index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT_SUMMARY, index=False)
    print(f"\nwrote {OUT_OBS} ({len(obs)} rows) and {OUT_SUMMARY}")
    print(summ.groupby("split")[["unaligned_pct", "route_arcs_with_points_pct", "median_dist_m"]].mean().round(2))
    print("overall unaligned %:", round((obs["true_arc_id"] == -1).mean() * 100, 2))


if __name__ == "__main__":
    main()
