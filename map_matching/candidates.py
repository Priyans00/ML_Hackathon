"""Deterministic spatially indexed road-candidate generation."""
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd

from utils.geo import point_segment_distance


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate search parameters in metres."""

    radius_m: float = 75.0
    top_k: int = 8
    cell_size_m: float = 100.0

    def __post_init__(self):
        if self.radius_m <= 0 or self.top_k <= 0 or self.cell_size_m <= 0:
            raise ValueError("radius_m, top_k, and cell_size_m must be positive")


class CandidateIndex:
    """Per-trip spatial index; arc geometry is never shared across trips."""

    def __init__(self, arcs, cell_size_m):
        self.arcs = arcs
        self.arc_lookup = arcs.set_index("arc_id", drop=False)
        self.cell_size_m = cell_size_m
        self.grid = _grid_index(arcs, cell_size_m)

    @classmethod
    def build(cls, trip, config):
        return cls(trip.arcs, config.cell_size_m)


def _grid_index(arcs, cell_size):
    grid = defaultdict(list)
    for row in arcs.itertuples(index=False):
        if not np.isfinite([row.ax, row.ay, row.bx, row.by]).all():
            continue
        x0, x1 = sorted((row.ax, row.bx))
        y0, y1 = sorted((row.ay, row.by))
        for ix in range(int(np.floor(x0 / cell_size)), int(np.floor(x1 / cell_size)) + 1):
            for iy in range(int(np.floor(y0 / cell_size)), int(np.floor(y1 / cell_size)) + 1):
                grid[(ix, iy)].append(int(row.arc_id))
    return grid


def generate_candidates(trip, config=None, index=None):
    """Return one row per nearby arc and observation.

    A uniform grid first filters segment bounding boxes, then exact
    point-to-segment distances select deterministic top-K candidates.
    """
    config = config or CandidateConfig()
    tr, arcs = trip.track, trip.arcs
    if index is None:
        valid_arcs = arcs[np.isfinite(arcs[["ax", "ay", "bx", "by"]]).all(axis=1)]
        index = CandidateIndex(valid_arcs, config.cell_size_m)
    rows = []
    for obs in tr.itertuples(index=False):
        if not np.isfinite([obs.x, obs.y]).all():
            continue
        cx, cy = int(np.floor(obs.x / config.cell_size_m)), int(np.floor(obs.y / config.cell_size_m))
        reach = int(np.ceil(config.radius_m / config.cell_size_m))
        ids = sorted({
            arc_id for ix in range(cx - reach, cx + reach + 1)
            for iy in range(cy - reach, cy + reach + 1)
            for arc_id in index.grid.get((ix, iy), ())
        })
        if not ids:
            continue
        nearby = index.arc_lookup.loc[ids]
        distance, fraction = point_segment_distance(
            obs.x, obs.y, nearby.ax.to_numpy(), nearby.ay.to_numpy(),
            nearby.bx.to_numpy(), nearby.by.to_numpy(),
        )
        keep = np.isfinite(distance) & (distance <= config.radius_m)
        if not keep.any():
            continue
        found = pd.DataFrame({
            "arc_id": nearby.arc_id.to_numpy()[keep].astype(np.int64),
            "distance_m": distance[keep].astype(float),
            "projection_fraction": fraction[keep].astype(float),
        }).sort_values(["distance_m", "arc_id"], kind="mergesort").head(config.top_k)
        found.insert(0, "candidate_rank", np.arange(1, len(found) + 1))
        found.insert(0, "obs_idx", int(obs.obs_idx))
        found.insert(0, "trip_id", trip.trip_id)
        found["t"] = float(obs.t)
        found["lon"], found["lat"] = float(obs.lon), float(obs.lat)
        rows.append(found)
    columns = [
        "trip_id", "obs_idx", "t", "lon", "lat", "arc_id",
        "candidate_rank", "distance_m", "projection_fraction",
    ]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)
