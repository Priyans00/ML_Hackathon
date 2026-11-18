"""Per-trip data validation and inventory.

Run from the repo root:
    python -m data_processing.validate

Writes results/eda/validation.csv (one row per trip) and prints a summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loader import list_trips, load_trip

OUT = Path("results/eda")

# Trips are worldwide, so only check that coordinates are valid WGS84 values
# and not the (0, 0) "null island" placeholder.
LON_RANGE, LAT_RANGE = (-180.0, 180.0), (-90.0, 90.0)


def validate_trip(trip):
    tr, nd, ar, ro = trip.track, trip.nodes, trip.arcs, trip.route
    dt = np.diff(tr["t"].to_numpy())
    step = np.hypot(np.diff(tr["x"]), np.diff(tr["y"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)

    bad_coord = ~(tr["lon"].between(*LON_RANGE) & tr["lat"].between(*LAT_RANGE)) | ((tr["lon"] == 0) & (tr["lat"] == 0))
    arc_bad_node = ~(ar["u"].between(0, len(nd) - 1) & ar["v"].between(0, len(nd) - 1))
    route_bad_arc = ~ro["arc_id"].between(0, len(ar) - 1)

    # Route continuity: arc k's end node should equal arc k+1's start node.
    r = ro["arc_id"][~route_bad_arc].to_numpy()
    breaks = int(np.sum(ar["v"].to_numpy()[r[:-1]] != ar["u"].to_numpy()[r[1:]])) if len(r) > 1 else 0
    repeats = int(np.sum(r[:-1] == r[1:])) if len(r) > 1 else 0

    return {
        "split": trip.split,
        "trip_id": trip.trip_id,
        "n_obs": len(tr),
        "n_nodes": len(nd),
        "n_arcs": len(ar),
        "n_route_arcs": len(ro),
        "duration_s": float(tr["t"].iloc[-1] - tr["t"].iloc[0]) if len(tr) else 0.0,
        "track_nan": int(tr[["lon", "lat", "t"]].isna().sum().sum()),
        "node_nan": int(nd[["lon", "lat"]].isna().sum().sum()),
        "bad_coords": int(bad_coord.sum()),
        "duplicate_points": int(tr.duplicated(["lon", "lat", "t"]).sum()),
        "t_non_increasing": int(np.sum(dt <= 0)),
        "dt_1s_pct": float(np.mean(dt == 1) * 100) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "gaps_gt5s": int(np.sum(dt > 5)),
        "gaps_gt30s": int(np.sum(dt > 30)),
        "speed_p95_ms": float(np.nanpercentile(speed, 95)) if len(dt) else np.nan,
        "jumps_gt50ms": int(np.nansum(speed > 50)),
        "zero_len_arcs": int((ar["length_m"] == 0).sum()),
        "arc_bad_node_ref": int(arc_bad_node.sum()),
        "route_bad_arc_ref": int(route_bad_arc.sum()),
        "route_breaks": breaks,
        "route_repeats": repeats,
        "path_len_m": float(step.sum()),
        "centre_lon": trip.proj.lon0, "centre_lat": trip.proj.lat0,
        "lon_min": tr["lon"].min(), "lon_max": tr["lon"].max(),
        "lat_min": tr["lat"].min(), "lat_max": tr["lat"].max(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, tid in list_trips():
        rows.append(validate_trip(load_trip(tid, split)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "validation.csv", index=False)

    print(df.groupby("split").size().rename("trips").to_string(), "\n")
    cols = ["n_obs", "n_nodes", "n_arcs", "n_route_arcs", "duration_s", "dt_1s_pct", "dt_max_s"]
    print(df[cols].describe().round(1).to_string(), "\n")
    issues = ["track_nan", "node_nan", "bad_coords", "duplicate_points", "t_non_increasing",
              "gaps_gt5s", "gaps_gt30s", "jumps_gt50ms", "zero_len_arcs",
              "arc_bad_node_ref", "route_bad_arc_ref", "route_breaks", "route_repeats"]
    summary = pd.DataFrame({"total": df[issues].sum(), "trips_affected": (df[issues] > 0).sum()})
    print(summary.to_string())


if __name__ == "__main__":
    main()
