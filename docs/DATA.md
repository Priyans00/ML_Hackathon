# Dataset, preprocessing and labels

Owner: Shreya (dataset exploration and preprocessing).

## How to run (from the repo root)

```bash
pip install -r requirements.txt
python -m data_processing.validate          # results/eda/validation.csv
python -m data_processing.label_alignment   # data/processed/obs.parquet, results/eda/label_summary.csv  (~35 s)
python -m eda.run_eda                       # results/eda/*.png, network_stats.csv, eda_summary.txt
python -m eda.plot_trip 00000048 train      # results/maps/trip_00000048.html
```

`data/processed/` is git-ignored. Regenerate it with the commands above.

## Shared code everyone should import

| Module | What it gives you |
|---|---|
| `utils.data_loader.list_trips()` | `[(split, trip_id), ...]` for train and student_test |
| `utils.data_loader.load_trip(trip_id, split)` | `Trip` with `track`, `nodes`, `arcs`, `route`, `proj` |
| `utils.geo.LocalProjection` | per-trip lon/lat <-> metres |
| `utils.geo.point_segment_distance` | vectorised point-to-arc distance and projection fraction |
| `utils.geo.bearing_deg` | heading of a vector in degrees |
| `eda.plot_trip.plot_trip(...)` | Folium map: GPS, nearby network, reference route, optional predicted route |

`Trip` tables:
- `track`: `obs_idx, lon, lat, t, x, y`
- `nodes`: `node_id, lon, lat, x, y`
- `arcs`: `arc_id, u, v, ax, ay, bx, by, length_m` (a = start node, b = end node, in metres)
- `route`: `route_idx, arc_id`

## Data contract: `data/processed/obs.parquet`

One row per GPS observation (227,577 rows).

| Column | Meaning |
|---|---|
| `split` | `train` or `student_test` |
| `trip_id` | 8-digit string, e.g. `00000048` |
| `obs_idx` | row index in the `.track` file |
| `t, lon, lat` | raw values |
| `x, y` | metres in the trip's local projection |
| `route_idx` | position in the `.route` list, -1 if unlabelled |
| `true_arc_id` | label: arc the vehicle was on, -1 if unlabelled |
| `dist_to_true_m` | distance from the point to its labelled arc |

Candidate generation should keep `(trip_id, obs_idx, arc_id)` as the key and set
`label = (arc_id == true_arc_id)`. Rows with `true_arc_id == -1` should be excluded
from training and from Top-1 accuracy, and reported separately.

## Key findings

1. **Trips are worldwide, not one city.** Trips cover Europe, Russia, Japan, the
   Americas, South Africa and more. A single national grid (e.g. EPSG:27700) gives
   wrong distances for most trips, so each trip uses its own local Azimuthal
   Equidistant projection centred on the GPS median. Checked against haversine:
   max error 0.56 %.
2. **Each trip has its own road network** (299 to 155,003 nodes). Arc and node IDs
   are only meaningful within a trip.
3. **No missing values**, no bad coordinates, timestamps strictly increasing.
4. **Sampling:** 96.7 % of steps are exactly 1 s. 1,260 gaps > 5 s in 47 trips,
   135 gaps > 30 s in 30 trips, largest gap 42.5 h (trip 00000090).
5. **GPS jumps:** 1,541 steps imply > 50 m/s, in 42 trips.
6. **Network:** 0 isolated nodes, 24 self-loops, 1,903 duplicate arcs, 106 zero-length
   arcs. Median 19 weakly connected components per trip; the largest strongly
   connected component holds a median 96.4 % of nodes. ~95 % of arcs exist in both
   directions, so direction matters when choosing between an arc and its reverse.
7. **Reference route limitations:** 14 discontinuities (an arc's end node is not the
   next arc's start node) in 11 trips.
8. **Stationary off-network periods:** trips 00000054 (89 %) and 00000033 (59 %) contain
   long parked periods where GPS jitters off the road network. These points are
   unlabelled (-1) rather than forced onto a road.

## Labelling method

`.route` has far fewer rows than `.track`, so per-observation labels are derived by a
**monotone alignment** solved with dynamic programming:
- the label's route position never moves backwards in time,
- the route distance skipped between consecutive points must be plausible:
  at most max(150 m, 50 m/s x dt); after a gap > 60 s any forward jump is allowed,
- cost = sum of min(point-to-arc distance, 100 m) + 0.02 per metre of route skipped,
- points further than 50 m from their chosen arc get label -1.

Results: 10.19 % unlabelled overall (mostly trips 54 and 33), 0.07 % on student_test.
Median distance to the labelled arc 5.0 m, 95th percentile 17.4 m. On average 92 % of
route arcs receive at least one GPS point; the rest are short arcs passed between
1 s samples.

Known limitation: at the edge of a parked cluster, drifting points within 50 m of the
route can still be labelled with the nearest route arc.
