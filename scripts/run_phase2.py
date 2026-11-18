"""Generate labelled candidate features and coverage metrics.

Run with the repository root as the working directory:
    E:\anaconda\envs\lab_env\python.exe scripts\run_phase2.py
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from data_processing.label_alignment import label_trip
from map_matching.candidates import CandidateConfig
from map_matching.features import build_feature_table
from utils.data_loader import list_trips, load_trip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=float, default=75.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="results/phase2/candidates.parquet")
    args = parser.parse_args()
    config = CandidateConfig(radius_m=args.radius, top_k=args.top_k)
    frames, metrics = [], []
    started = time.perf_counter()
    trips = list_trips((args.split,))
    total = len(trips)
    for number, (split, trip_id) in enumerate(trips, 1):
        trip_started = time.perf_counter()
        stage_started = time.perf_counter()
        trip = load_trip(trip_id, split)
        load_s = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        labels, _ = label_trip(trip)
        label_s = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        candidates = build_feature_table(trip, config=config, labels=labels)
        feature_s = time.perf_counter() - stage_started
        frames.append(candidates.assign(split=split))
        observed = set(trip.track.obs_idx)
        covered = set(candidates.obs_idx)
        labelled = labels[labels.true_arc_id >= 0][["obs_idx", "true_arc_id"]]
        candidate_pairs = candidates[["obs_idx", "arc_id"]].drop_duplicates()
        valid_pairs = labelled.merge(candidate_pairs, left_on=["obs_idx", "true_arc_id"],
                                     right_on=["obs_idx", "arc_id"], how="left", indicator=True)
        recall = (valid_pairs["_merge"] == "both").mean() if len(valid_pairs) else float("nan")
        metrics.append({
            "split": split, "trip_id": trip_id, "n_observations": len(observed),
            "n_candidate_rows": len(candidates), "coverage_pct": 100 * len(covered) / max(len(observed), 1),
            "labelled_candidate_recall_pct": 100 * recall,
            "mean_candidates": len(candidates) / max(len(covered), 1),
        })
        print(
            f"[{number}/{total}] {split} {trip_id}: load={load_s:.2f}s "
            f"labels={label_s:.2f}s candidates+features={feature_s:.2f}s "
            f"obs={len(observed):,} candidates={len(candidates):,} "
            f"trip_total={time.perf_counter() - trip_started:.2f}s",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
    pd.DataFrame(metrics).to_csv(output.with_name("coverage.csv"), index=False)
    print(f"wrote {output} in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
