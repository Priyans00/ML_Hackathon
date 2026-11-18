"""Leakage-safe candidate-level feature construction."""
import numpy as np
import pandas as pd

from map_matching.candidates import CandidateConfig, CandidateIndex, generate_candidates
from utils.geo import bearing_deg

FEATURE_COLUMNS = [
    "candidate_rank", "distance_m", "projection_fraction", "arc_length_m",
    "candidate_bearing_deg", "gps_heading_deg", "heading_error_deg",
    "gps_step_m", "dt_s", "gps_speed_mps",
]


def _angle_error(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def build_feature_table(trip, config=None, labels=None):
    """Build stable candidate features; labels are attached only when supplied."""
    config = config or CandidateConfig()
    candidates = generate_candidates(trip, config=config, index=CandidateIndex.build(trip, config))
    if candidates.empty:
        return candidates.assign(**{c: pd.Series(dtype=float) for c in FEATURE_COLUMNS},
                                label=pd.Series(dtype="int8"))
    arcs = trip.arcs.set_index("arc_id")
    candidates["arc_length_m"] = candidates.arc_id.map(arcs.length_m).astype(float)
    ax = candidates.arc_id.map(arcs.ax).to_numpy(float)
    ay = candidates.arc_id.map(arcs.ay).to_numpy(float)
    bx = candidates.arc_id.map(arcs.bx).to_numpy(float)
    by = candidates.arc_id.map(arcs.by).to_numpy(float)
    candidates["candidate_bearing_deg"] = bearing_deg(bx - ax, by - ay)
    tr = trip.track.set_index("obs_idx")
    idx = candidates.obs_idx.to_numpy()
    previous = np.maximum(idx - 1, 0)
    dx = tr.loc[idx, "x"].to_numpy() - tr.loc[previous, "x"].to_numpy()
    dy = tr.loc[idx, "y"].to_numpy() - tr.loc[previous, "y"].to_numpy()
    step = np.hypot(dx, dy)
    dt = tr.loc[idx, "t"].to_numpy() - tr.loc[previous, "t"].to_numpy()
    moving = (step > 1e-9) & (dt > 0)
    gps_heading = np.full(len(idx), np.nan)
    gps_heading[moving] = bearing_deg(dx[moving], dy[moving])
    candidates["gps_heading_deg"] = gps_heading
    candidates["heading_error_deg"] = _angle_error(
        np.nan_to_num(gps_heading, nan=0.0), candidates.candidate_bearing_deg.to_numpy()
    )
    candidates["gps_step_m"] = step
    candidates["dt_s"] = np.where(dt > 0, dt, np.nan)
    speed = np.full(len(dt), np.nan, dtype=float)
    np.divide(step, dt, out=speed, where=dt > 0)
    candidates["gps_speed_mps"] = speed
    for column in FEATURE_COLUMNS:
        candidates[column] = candidates[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if labels is not None:
        label_map = labels.set_index("obs_idx").true_arc_id
        true_arc = candidates.obs_idx.map(label_map).fillna(-1)
        candidates["label"] = (candidates.arc_id == true_arc).astype("int8")
        candidates.loc[true_arc.to_numpy() < 0, "label"] = -1
    return candidates


def build_features_for_trips(trips, config=None, label_tables=None):
    """Build and concatenate tables for ``[(split, trip_id, trip), ...]``."""
    frames = []
    for split, trip_id, trip in trips:
        labels = None if label_tables is None else label_tables.get((split, trip_id))
        frame = build_feature_table(trip, config=config, labels=labels)
        if not frame.empty:
            frame.insert(1, "split", split)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
