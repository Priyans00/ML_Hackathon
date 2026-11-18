"""Road-network HMM and Viterbi decoding for PathSnap.

The classifier scores a *candidate arc* independently for each observation.
This module keeps that candidate-to-arc identity intact and chooses one
candidate per observation jointly.  It deliberately consumes no reference
route or label columns: labels belong only in offline evaluation code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {
    "obs_idx", "arc_id", "projection_fraction", "score", "t", "x", "y",
}


@dataclass(frozen=True)
class DecoderConfig:
    """Explainable HMM controls, expressed in metres and seconds.

    ``reset_gap_s`` prevents a long GPS outage from forcing an implausible
    connection. ``max_speed_mps`` and ``max_network_search_m`` bound shortest
    path work and reject physically implausible transitions.
    """

    reset_gap_s: float = 60.0
    max_speed_mps: float = 50.0
    min_transition_m: float = 150.0
    max_network_search_m: float = 2_000.0
    distance_factor: float = 4.0
    min_sigma_m: float = 25.0
    emission_floor: float = 1e-6

    def __post_init__(self):
        if min(
            self.reset_gap_s, self.max_speed_mps, self.min_transition_m,
            self.max_network_search_m, self.distance_factor, self.min_sigma_m,
            self.emission_floor,
        ) <= 0:
            raise ValueError("decoder configuration values must be positive")


class RoadNetwork:
    """Directed road graph plus per-arc metadata for one trip.

    Arc IDs are only meaningful within a trip, so an instance must never be
    reused across trips. Distances between arc end/start nodes are cached after
    bounded Dijkstra searches.
    """

    def __init__(self, arcs: pd.DataFrame, max_search_m: float):
        required = {"arc_id", "u", "v", "length_m"}
        missing = required.difference(arcs.columns)
        if missing:
            raise ValueError("arcs are missing columns: " + ", ".join(sorted(missing)))
        self.arc_lookup = arcs.set_index("arc_id", drop=False)
        self.max_search_m = float(max_search_m)
        self.graph = nx.DiGraph()
        for row in arcs.itertuples(index=False):
            length = float(row.length_m)
            if not np.isfinite(length) or length < 0:
                continue
            previous = self.graph.get_edge_data(int(row.u), int(row.v), {}).get("weight")
            if previous is None or length < previous:
                self.graph.add_edge(int(row.u), int(row.v), weight=length)
        self._distances: dict[int, dict[int, float]] = {}

    def arc(self, arc_id: int):
        try:
            return self.arc_lookup.loc[int(arc_id)]
        except KeyError as exc:
            raise ValueError(f"unknown arc_id {arc_id}") from exc

    def distance(self, source: int, target: int) -> float | None:
        """Return bounded directed network distance, or ``None`` if unreachable."""
        source, target = int(source), int(target)
        if source == target:
            return 0.0
        if source not in self.graph or target not in self.graph:
            return None
        if source not in self._distances:
            self._distances[source] = nx.single_source_dijkstra_path_length(
                self.graph, source, cutoff=self.max_search_m, weight="weight"
            )
        value = self._distances[source].get(target)
        return None if value is None else float(value)


class ViterbiDecoder:
    """Decode model-scored candidates into a road-segment sequence.

    Emission log-probability is ``log(classifier_probability)``.  Transition
    log-probability is a Gaussian-like penalty on the discrepancy between GPS
    displacement and feasible directed road distance.  A transition with no
    bounded directed path, or an implausible speed, is impossible.  Rather than
    dropping later observations, the decoder starts a new segment after such a
    break; this is also how missing candidates and large GPS gaps are handled.
    """

    def __init__(self, trip, config: DecoderConfig | None = None):
        self.trip = trip
        self.config = config or DecoderConfig()
        self.network = RoadNetwork(trip.arcs, self.config.max_network_search_m)

    def _emission(self, row) -> float:
        score = float(row.score)
        if not np.isfinite(score):
            score = self.config.emission_floor
        return float(np.log(np.clip(score, self.config.emission_floor, 1.0)))

    def _road_distance(self, previous, current) -> float | None:
        before, after = self.network.arc(previous.arc_id), self.network.arc(current.arc_id)
        before_fraction = float(np.clip(previous.projection_fraction, 0.0, 1.0))
        after_fraction = float(np.clip(current.projection_fraction, 0.0, 1.0))
        before_length, after_length = float(before.length_m), float(after.length_m)

        # Moving forward on the same directed arc needs no graph search.
        if int(previous.arc_id) == int(current.arc_id) and after_fraction >= before_fraction:
            return (after_fraction - before_fraction) * before_length

        core = self.network.distance(int(before.v), int(after.u))
        if core is None:
            return None
        return (1.0 - before_fraction) * before_length + core + after_fraction * after_length

    def transition_log_probability(self, previous, current) -> float | None:
        """Score one directed candidate transition, returning ``None`` if invalid."""
        dt = float(current.t) - float(previous.t)
        if not np.isfinite(dt) or dt <= 0 or dt > self.config.reset_gap_s:
            return None
        gps_distance = float(np.hypot(float(current.x) - float(previous.x),
                                      float(current.y) - float(previous.y)))
        if not np.isfinite(gps_distance):
            return None
        road_distance = self._road_distance(previous, current)
        if road_distance is None:
            return None
        plausible_limit = max(
            self.config.min_transition_m,
            self.config.max_speed_mps * dt + 100.0,
            self.config.distance_factor * gps_distance + 100.0,
        )
        if road_distance > plausible_limit:
            return None
        sigma = max(self.config.min_sigma_m, 0.5 * gps_distance + 5.0 * dt)
        return float(-0.5 * ((road_distance - gps_distance) / sigma) ** 2)

    @staticmethod
    def _validate(table: pd.DataFrame) -> None:
        missing = REQUIRED_CANDIDATE_COLUMNS.difference(table.columns)
        if missing:
            raise ValueError("candidate table is missing columns: " + ", ".join(sorted(missing)))

    @staticmethod
    def _ordered_groups(table: pd.DataFrame) -> Iterable[pd.DataFrame]:
        order = ["obs_idx", "score", "arc_id"]
        ordered = table.sort_values(order, ascending=[True, False, True], kind="mergesort")
        for _, group in ordered.groupby("obs_idx", sort=True):
            yield group.reset_index(drop=True)

    def _decode_segment(self, groups: list[pd.DataFrame], segment_id: int) -> list[dict]:
        """Run ordinary Viterbi backtracking for one contiguous feasible run."""
        scores: list[np.ndarray] = []
        backpointers: list[np.ndarray] = []
        for index, group in enumerate(groups):
            emissions = np.asarray([self._emission(row) for row in group.itertuples(index=False)])
            if index == 0:
                scores.append(emissions)
                backpointers.append(np.full(len(group), -1, dtype=int))
                continue
            previous_group, previous_scores = groups[index - 1], scores[-1]
            current_scores = np.full(len(group), -np.inf)
            pointers = np.full(len(group), -1, dtype=int)
            for current_idx, current in enumerate(group.itertuples(index=False)):
                candidates = []
                for previous_idx, previous in enumerate(previous_group.itertuples(index=False)):
                    transition = self.transition_log_probability(previous, current)
                    if transition is not None and np.isfinite(previous_scores[previous_idx]):
                        candidates.append((previous_scores[previous_idx] + transition, previous_idx))
                if candidates:
                    best_score, best_idx = max(candidates, key=lambda item: item[0])
                    current_scores[current_idx] = emissions[current_idx] + best_score
                    pointers[current_idx] = best_idx
            if not np.isfinite(current_scores).any():
                raise RuntimeError("segment contains an impossible transition")
            scores.append(current_scores)
            backpointers.append(pointers)

        chosen = int(np.nanargmax(scores[-1]))
        rows: list[dict] = []
        for index in range(len(groups) - 1, -1, -1):
            row = groups[index].iloc[chosen]
            rows.append({
                "obs_idx": int(row.obs_idx),
                "arc_id": int(row.arc_id),
                "score": float(row.score),
                "segment_id": segment_id,
                "decoder_reset": index == 0,
            })
            if index:
                chosen = int(backpointers[index][chosen])
        return list(reversed(rows))

    def decode(self, scored_candidates: pd.DataFrame) -> pd.DataFrame:
        """Return one decoded arc for each observation having candidates.

        An empty candidate table produces a correctly shaped empty result.
        Missing observation indices are represented by a new decoded segment;
        they cannot safely receive an invented road arc.
        """
        self._validate(scored_candidates)
        columns = ["obs_idx", "arc_id", "score", "segment_id", "decoder_reset"]
        if scored_candidates.empty:
            return pd.DataFrame(columns=columns)

        output: list[dict] = []
        pending: list[pd.DataFrame] = []
        segment_id = 0
        previous_group: pd.DataFrame | None = None
        previous_obs: int | None = None

        def flush() -> None:
            nonlocal pending, segment_id
            if pending:
                output.extend(self._decode_segment(pending, segment_id))
                segment_id += 1
                pending = []

        for group in self._ordered_groups(scored_candidates):
            obs = int(group.obs_idx.iloc[0])
            split_for_gap = previous_obs is not None and obs != previous_obs + 1
            time_gap = False
            impossible = False
            if previous_group is not None and not split_for_gap:
                dt = float(group.t.iloc[0]) - float(previous_group.t.iloc[0])
                time_gap = not np.isfinite(dt) or dt <= 0 or dt > self.config.reset_gap_s
                if not time_gap:
                    possible = any(
                        self.transition_log_probability(left, right) is not None
                        for left in previous_group.itertuples(index=False)
                        for right in group.itertuples(index=False)
                    )
                    impossible = not possible
            if split_for_gap or time_gap or impossible:
                flush()
            pending.append(group)
            previous_group, previous_obs = group, obs
        flush()
        return pd.DataFrame(output, columns=columns).sort_values("obs_idx", kind="mergesort").reset_index(drop=True)


def continuity_metrics(decoded: pd.DataFrame, candidate_table: pd.DataFrame, trip,
                       config: DecoderConfig | None = None) -> dict:
    """Compute label-free route continuity diagnostics for one selected sequence."""
    if decoded.empty:
        return {"n_pairs": 0, "plausible_transition_rate": np.nan, "n_segments": 0}
    config = config or DecoderConfig()
    decoder = ViterbiDecoder(trip, config)
    fields = ["obs_idx", "arc_id", "projection_fraction", "t", "x", "y"]
    selected = decoded[["obs_idx", "arc_id"]].merge(
        candidate_table[fields], on=["obs_idx", "arc_id"], how="left", validate="one_to_one"
    ).sort_values("obs_idx")
    plausible, total = 0, 0
    rows = list(selected.itertuples(index=False))
    for previous, current in zip(rows, rows[1:]):
        if int(current.obs_idx) != int(previous.obs_idx) + 1:
            continue
        total += 1
        if decoder.transition_log_probability(previous, current) is not None:
            plausible += 1
    return {
        "n_pairs": int(total),
        "plausible_transition_rate": float(plausible / total) if total else np.nan,
        "n_segments": int(decoded.segment_id.nunique()),
    }
