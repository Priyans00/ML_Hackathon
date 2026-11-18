"""Exploratory data analysis for PathSnap (train + student_test partitions).

Run from the repo root, after data_processing.validate and
data_processing.label_alignment:
    python -m eda.run_eda

Outputs (results/eda/):
    network_stats.csv       per-trip road-network size and connectivity
    eda_summary.txt         headline numbers used in the report
    fig_*.png               plots for report section 2.2
    ../maps/overview_all_trips.html   Folium map of every trip's location
"""
from pathlib import Path

import folium
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.data_loader import list_trips, load_trip  # noqa: E402

OUT = Path("results/eda")
MAPS = Path("results/maps")
COLORS = {"train": "#2166ac", "student_test": "#d6604d"}


def network_stats(trip):
    ar = trip.arcs
    g = nx.DiGraph()
    g.add_nodes_from(range(len(trip.nodes)))
    g.add_edges_from(zip(ar.u, ar.v))
    deg = np.array([d for _, d in g.to_undirected(as_view=True).degree()])
    wcc = [len(c) for c in nx.weakly_connected_components(g)]
    scc = [len(c) for c in nx.strongly_connected_components(g)]
    pairs = set(zip(ar.u, ar.v))
    two_way = sum((v, u) in pairs for u, v in pairs)
    return {
        "split": trip.split, "trip_id": trip.trip_id,
        "n_nodes": len(trip.nodes), "n_arcs": len(ar),
        "isolated_nodes": int((deg == 0).sum()),
        "dead_end_nodes": int((deg == 1).sum()),
        "mean_degree": float(deg.mean()),
        "weak_components": len(wcc),
        "largest_wcc_pct": max(wcc) / len(trip.nodes) * 100,
        "strong_components": len(scc),
        "largest_scc_pct": max(scc) / len(trip.nodes) * 100,
        "two_way_arc_pct": two_way / max(len(pairs), 1) * 100,
        "self_loops": int((ar.u == ar.v).sum()),
        "duplicate_arcs": int(ar.duplicated(["u", "v"]).sum()),
        "arc_len_median_m": float(ar.length_m.median()),
        "route_len_km": float(ar.length_m.to_numpy()[trip.route.arc_id].sum() / 1000),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(OUT / "validation.csv", dtype={"trip_id": str})
    obs = pd.read_parquet("data/processed/obs.parquet")

    dts, speeds, net_rows, centres = [], [], [], []
    for split, tid in list_trips():
        trip = load_trip(tid, split)
        tr = trip.track
        dt = np.diff(tr.t.to_numpy())
        step = np.hypot(np.diff(tr.x), np.diff(tr.y))
        dts.append(pd.DataFrame({"split": split, "dt": dt}))
        speeds.append(pd.DataFrame({"split": split, "speed": step[dt > 0] / dt[dt > 0]}))
        net_rows.append(network_stats(trip))
        centres.append((split, tid, trip.proj.lat0, trip.proj.lon0, len(tr)))
        print("done", split, tid)
    dts, speeds = pd.concat(dts), pd.concat(speeds)
    net = pd.DataFrame(net_rows)
    net.to_csv(OUT / "network_stats.csv", index=False)

    # ---------- figures ----------
    def hist_by_split(ax, df, col, bins, log=False):
        for s, g in df.groupby("split"):
            ax.hist(g[col], bins=bins, alpha=0.6, label=s, color=COLORS[s], density=True)
        if log:
            ax.set_xscale("log")
        ax.legend()

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    hist_by_split(axs[0], val, "n_obs", np.logspace(2, 4.5, 30), log=True)
    axs[0].set(title="GPS observations per trip", xlabel="observations (log)", ylabel="density")
    hist_by_split(axs[1], val.assign(km=val.path_len_m / 1000), "km", 30)
    axs[1].set(title="GPS path length per trip", xlabel="km")
    hist_by_split(axs[2], val, "n_arcs", np.logspace(2.5, 5.6, 30), log=True)
    axs[2].set(title="Road-network arcs per trip", xlabel="arcs (log)")
    fig.tight_layout(); fig.savefig(OUT / "fig_trip_sizes.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    vc = dts.dt.clip(upper=61).round().value_counts().sort_index()
    axs[0].bar(vc.index, vc.values, color="#2166ac")
    axs[0].set_yscale("log")
    axs[0].set(title="Time between consecutive GPS points (both partitions)",
                xlabel="seconds (61 = 61 s or more)", ylabel="count (log)")
    sp = speeds[speeds.speed < 80]
    hist_by_split(axs[1], sp, "speed", 80)
    axs[1].set(title="Implied speed between consecutive points", xlabel="m/s (values >= 80 m/s hidden)")
    fig.tight_layout(); fig.savefig(OUT / "fig_sampling_speed.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    d = obs.dist_to_true_m.dropna()
    axs[0].hist(d.clip(upper=50), bins=50, color="#1a9850")
    axs[0].set(title="Distance from GPS point to its labelled route arc",
               xlabel="metres", ylabel="observations")
    lab = obs.groupby(["split", "trip_id"]).true_arc_id.apply(lambda s: (s == -1).mean() * 100).reset_index()
    lab = lab.sort_values("true_arc_id", ascending=False).head(12)
    axs[1].barh(lab.trip_id, lab.true_arc_id, color=[COLORS[s] for s in lab.split])
    axs[1].invert_yaxis()
    axs[1].set(title="Trips with most unlabelled points (>50 m from route)", xlabel="% of observations")
    fig.tight_layout(); fig.savefig(OUT / "fig_label_quality.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    axs[0].scatter(net.n_nodes, net.n_arcs, c=[COLORS[s] for s in net.split], s=18)
    axs[0].set(xscale="log", yscale="log", title="Network size", xlabel="nodes", ylabel="arcs")
    hist_by_split(axs[1], net, "largest_scc_pct", 20)
    axs[1].set(title="Largest strongly connected component", xlabel="% of nodes")
    hist_by_split(axs[2], net, "two_way_arc_pct", 20)
    axs[2].set(title="Arcs that also exist in reverse direction", xlabel="% of arcs")
    fig.tight_layout(); fig.savefig(OUT / "fig_network.png", dpi=120); plt.close(fig)

    gaps = val.sort_values("dt_max_s", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(gaps.trip_id, gaps.dt_max_s / 60, color=[COLORS[s] for s in gaps.split])
    ax.set_xscale("log"); ax.invert_yaxis()
    ax.set(title="Largest single time gap per trip (top 12)", xlabel="minutes (log)")
    fig.tight_layout(); fig.savefig(OUT / "fig_largest_gaps.png", dpi=120); plt.close(fig)

    # ---------- overview map ----------
    m = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    for s, tid, lat, lon, n in centres:
        folium.CircleMarker((lat, lon), radius=6, color=COLORS[s], fill=True, fill_opacity=0.9,
                            tooltip=f"{s} trip {tid}: {n} GPS points").add_to(m)
    m.save(MAPS / "overview_all_trips.html")

    # ---------- headline numbers ----------
    n_tr, n_st = (val.split == "train").sum(), (val.split == "student_test").sum()
    lines = [
        f"Trips: train={n_tr}, student_test={n_st}",
        f"GPS observations: total={val.n_obs.sum()}, per trip median={val.n_obs.median():.0f}, "
        f"min={val.n_obs.min()}, max={val.n_obs.max()}",
        f"Missing values: track NaN={val.track_nan.sum()}, node NaN={val.node_nan.sum()}",
        f"Timestamps non-increasing: {val.t_non_increasing.sum()}",
        f"Sampling: {(dts.dt == 1).mean() * 100:.1f}% of steps are exactly 1 s; "
        f"{(dts.dt > 5).sum()} gaps > 5 s in {(val.gaps_gt5s > 0).sum()} trips; "
        f"{(dts.dt > 30).sum()} gaps > 30 s in {(val.gaps_gt30s > 0).sum()} trips; largest gap {dts.dt.max() / 3600:.1f} h",
        f"Implied speed > 50 m/s (GPS jumps): {(speeds.speed > 50).sum()} steps in {(val.jumps_gt50ms > 0).sum()} trips",
        f"Road network: nodes median={net.n_nodes.median():.0f} (max {net.n_nodes.max()}), "
        f"arcs median={net.n_arcs.median():.0f} (max {net.n_arcs.max()})",
        f"Isolated nodes: {net.isolated_nodes.sum()}; self-loops: {net.self_loops.sum()}; duplicate arcs: {net.duplicate_arcs.sum()}; "
        f"zero-length arcs: {val.zero_len_arcs.sum()}",
        f"Weakly connected components per trip: median={net.weak_components.median():.0f}, max={net.weak_components.max()}",
        f"Largest strongly connected component: median {net.largest_scc_pct.median():.1f}% of nodes",
        f"Two-way arcs: median {net.two_way_arc_pct.median():.1f}% of arcs",
        f"Reference route: {val.n_route_arcs.sum()} arcs total; discontinuities (end node != next start node): "
        f"{val.route_breaks.sum()} in {(val.route_breaks > 0).sum()} trips",
        f"Labels: {(obs.true_arc_id == -1).mean() * 100:.2f}% of observations unlabelled (>50 m from route); "
        f"median distance to labelled arc {obs.dist_to_true_m.median():.1f} m, 95th pct {obs.dist_to_true_m.quantile(.95):.1f} m",
        "Unlabelled observations are concentrated in trips "
        + ", ".join(f"{r.trip_id} ({r.true_arc_id:.0f}%)" for r in lab.head(3).itertuples()),
    ]
    (OUT / "eda_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
