"""Reusable Folium map for one trip.

    from eda.plot_trip import plot_trip
    m = plot_trip("00000048", "train")                         # GPS + network + reference route
    m = plot_trip("00000048", "train", obs=obs_df)             # also colours points by label
    m = plot_trip(tid, split, predicted_arcs=[...])           # adds a predicted-route layer
    m.save("results/maps/trip_00000048.html")

CLI:  python -m eda.plot_trip 00000048 train
"""
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from utils.data_loader import load_trip
from utils.geo import point_segment_distance

NETWORK_RADIUS_M = 300.0   # only draw road arcs this close to the GPS track
MAX_GPS_MARKERS = 2500     # thin very long tracks so the HTML stays light


def _arc_lines(trip, arc_ids):
    nd = trip.nodes
    ar = trip.arcs.set_index("arc_id").loc[arc_ids]
    return [[(nd.lat[u], nd.lon[u]), (nd.lat[v], nd.lon[v])] for u, v in zip(ar.u, ar.v)]


def _nearby_arcs(trip, radius=NETWORK_RADIUS_M):
    """Arc ids whose segment is within `radius` metres of any (thinned) GPS point."""
    tr, ar = trip.track, trip.arcs
    step = max(1, len(tr) // 1500)
    px, py = tr.x.to_numpy()[::step], tr.y.to_numpy()[::step]
    # quick bounding-box filter first
    pad = radius
    in_box = ((np.minimum(ar.ax, ar.bx) <= px.max() + pad) & (np.maximum(ar.ax, ar.bx) >= px.min() - pad) &
              (np.minimum(ar.ay, ar.by) <= py.max() + pad) & (np.maximum(ar.ay, ar.by) >= py.min() - pad))
    cand = ar[in_box]
    keep = np.zeros(len(cand), bool)
    for c in range(0, len(cand), 2000):
        a = cand.iloc[c:c + 2000]
        d, _ = point_segment_distance(px[:, None], py[:, None], a.ax.to_numpy()[None], a.ay.to_numpy()[None],
                                      a.bx.to_numpy()[None], a.by.to_numpy()[None])
        keep[c:c + 2000] = d.min(axis=0) <= radius
    return cand.arc_id.to_numpy()[keep]


def plot_trip(trip_id, split, obs=None, predicted_arcs=None, trip=None, show_network=True):
    trip = trip or load_trip(trip_id, split)
    tr = trip.track
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)

    if show_network:
        net = folium.FeatureGroup(name="Road network (nearby)", show=True)
        folium.PolyLine(_arc_lines(trip, _nearby_arcs(trip)), color="#555555", weight=1.5, opacity=0.6).add_to(net)
        net.add_to(m)

    if len(trip.route):
        ref = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_arc_lines(trip, trip.route.arc_id.to_numpy()), color="#1a9850", weight=5, opacity=0.8).add_to(ref)
        ref.add_to(m)

    if predicted_arcs is not None and len(predicted_arcs):
        pred = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_arc_lines(trip, np.unique(np.asarray(predicted_arcs))), color="#d73027", weight=3,
                        opacity=0.9, dash_array="6").add_to(pred)
        pred.add_to(m)

    step = max(1, len(tr) // MAX_GPS_MARKERS)
    pts = tr.iloc[::step]
    if obs is not None:
        lab = obs[obs.trip_id == trip_id].set_index("obs_idx").true_arc_id
        unaligned = lab.reindex(pts.obs_idx).to_numpy() == -1
    else:
        unaligned = np.zeros(len(pts), bool)
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for (_, r), bad in zip(pts.iterrows(), unaligned):
        folium.CircleMarker((r.lat, r.lon), radius=2, weight=0, fill=True, fill_opacity=0.8,
                            color="#e41a1c" if bad else "#2166ac",
                            tooltip=f"obs {int(r.obs_idx)}  t={r.t:.0f}s" + ("  (unlabelled)" if bad else "")).add_to(gps)
    gps.add_to(m)

    folium.Marker((tr.lat.iloc[0], tr.lon.iloc[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr.lat.iloc[-1], tr.lon.iloc[-1]), tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>Trip {trip_id} ({split})</b><br>
      <span style="color:#2166ac">&#9679;</span> GPS observation<br>
      <span style="color:#e41a1c">&#9679;</span> GPS, no label (&gt;50 m from route)<br>
      <span style="color:#1a9850">&#9644;</span> Reference route<br>
      <span style="color:#d73027">&#9476;</span> Predicted route<br>
      <span style="color:#888">&#9644;</span> Road network (within {NETWORK_RADIUS_M:.0f} m)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    return m


if __name__ == "__main__":
    tid, split = sys.argv[1], sys.argv[2]
    obs_path = Path("data/processed/obs.parquet")
    obs = pd.read_parquet(obs_path) if obs_path.exists() else None
    out = Path("results/maps") / f"trip_{tid}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_trip(tid, split, obs=obs).save(out)
    print("saved", out)
