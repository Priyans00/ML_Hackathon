"""Loader for the four raw PathSnap file types.

Usage:
    from utils.data_loader import list_trips, load_trip
    for split, trip_id in list_trips():
        trip = load_trip(trip_id, split)
        trip.track, trip.nodes, trip.arcs, trip.route

Raw formats (headerless, whitespace separated):
    .track  lon lat t          one GPS observation per row (~1 Hz)
    .nodes  lon lat            row index = node_id
    .arcs   from_node to_node  row index = arc_id
    .route  arc_id             ground-truth arcs in travel order
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.geo import LocalProjection

DATA_ROOT = Path(__file__).resolve().parents[1] / "dataset"
SPLITS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    split: str
    track: pd.DataFrame  # obs_idx, lon, lat, t, x, y
    nodes: pd.DataFrame  # node_id, lon, lat, x, y
    arcs: pd.DataFrame   # arc_id, u, v, ax, ay, bx, by, length_m
    route: pd.DataFrame  # route_idx, arc_id  (empty if no .route file)
    proj: LocalProjection  # per-trip lon/lat <-> metres


def list_trips(splits=SPLITS, root=DATA_ROOT):
    """Return [(split, trip_id), ...] for every trip folder found."""
    out = []
    for split in splits:
        d = Path(root) / split
        if d.is_dir():
            out += [(split, p.name) for p in sorted(d.iterdir()) if p.is_dir()]
    return out


def _read(path, names, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(trip_id, split, root=DATA_ROOT):
    d = Path(root) / split / trip_id

    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"], float)
    track.insert(0, "obs_idx", np.arange(len(track)))
    proj = LocalProjection.for_points(track["lon"], track["lat"])
    track["x"], track["y"] = proj.to_metres(track["lon"], track["lat"])

    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], float)
    nodes.insert(0, "node_id", np.arange(len(nodes)))
    nodes["x"], nodes["y"] = proj.to_metres(nodes["lon"], nodes["lat"])

    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], np.int64)
    arcs.insert(0, "arc_id", np.arange(len(arcs)))
    nx_, ny_ = nodes["x"].to_numpy(), nodes["y"].to_numpy()
    arcs["ax"], arcs["ay"] = nx_[arcs["u"]], ny_[arcs["u"]]
    arcs["bx"], arcs["by"] = nx_[arcs["v"]], ny_[arcs["v"]]
    arcs["length_m"] = np.hypot(arcs["bx"] - arcs["ax"], arcs["by"] - arcs["ay"])

    route_path = d / f"{trip_id}.route"
    if route_path.exists():
        route = _read(route_path, ["arc_id"], np.int64)
    else:
        route = pd.DataFrame({"arc_id": pd.Series(dtype=np.int64)})
    route.insert(0, "route_idx", np.arange(len(route)))

    return Trip(trip_id, split, track, nodes, arcs, route, proj)
