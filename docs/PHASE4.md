# Phase 4: HMM/Viterbi route reconstruction and integration

## Confirmed handoff contract

The earlier pipeline supplies a per-trip `Trip` with a directed `arcs` table,
and a feature table keyed by `(trip_id, obs_idx, arc_id)`. `arc_id` is local to
one trip. The selected model is loaded through `CandidateScorer` and returns
`trip_id, obs_idx, arc_id, candidate_rank, score, prediction`. Phase 4 uses
the continuous `score`, not the thresholded binary `prediction`, and it does
not alter the upstream schema.

## Decoder design

For each GPS observation, each generated candidate arc is one hidden HMM state.
Its identity remains the upstream `arc_id`; the state also retains its projected
position along that directed arc. A Viterbi state therefore means, “the vehicle
was on this particular directed road segment at this GPS time.”

The emission is `log(p)`, where `p` is the tuned Extra Trees probability for
that candidate. Probabilities are clipped to a small positive floor only to
avoid `log(0)`.

For a transition from candidate A to B, the decoder calculates directed road
distance as: remaining distance on A + shortest-path distance from A's end node
to B's start node + entered distance on B. It compares that with the observed
GPS displacement using a Gaussian-like log penalty. No directed path within the
bounded road search, or a road distance above a generous speed/displacement
limit, makes the transition impossible. The Viterbi recurrence is:

```text
V[t, j] = log(emission[t, j]) + max_i(V[t-1, i] + log(transition[i, j]))
```

The maximum final state is backtracked to return one predicted arc for every
observation that has candidates.

This formulation is deliberately modest: the network is directed, so reverse
arcs and duplicate geometry remain distinct. It does not invent a route between
two road components and it does not use the reference `.route` during
inference.

## Edge cases

- An observation without candidates has no safe road-segment prediction. It
  ends the current Viterbi segment; decoding resumes at the next observation
  with candidates.
- A missing observation index, non-positive time delta, GPS gap over 60 seconds,
  or a fully impossible candidate-to-candidate step starts a new segment.
- Within a segment, alternatives with impossible transitions receive no Viterbi
  path; a feasible lower-scoring candidate can therefore win over an isolated
  independent classifier peak.
- Shortest-path searches are capped at 2 km. This is an explicit runtime and
  plausibility guard, not evidence that longer physical routes are impossible.

## Evaluation protocol

`python -m scripts.evaluate_phase4 --split student_test` uses only the
permitted `student_test` partition. It runs the normal prediction pipeline
without labels, then uses the existing alignment code *offline* to calculate
development-only arc accuracy. It reports:

- classifier-only arc accuracy and decoded arc accuracy over labelled points;
- plausible-transition rate: fraction of adjacent selected pairs that admit a
  bounded, speed-plausible directed road transition;
- decoded segment count and decoded observation coverage.

The script refuses evaluator data by design. Metrics in `results/phase4/` are
the only valid evidence for any claim that decoding helped. No metric is stated
here until the runner has actually produced it.

## API and UI boundary

Laptop 1 owns `backend/`: the model, road graph, candidate generation, decoder,
and logging. It exposes `GET /health`, `GET /api/trips`, `GET /api/trips/{id}`
and `POST /api/match`. Requests can only select a directory in the configured
evaluator partition. Responses contain raw GPS points, predicted segments, and
non-label diagnostics; they never contain `route`, `true_arc_id`, candidate
labels, or reference geometry.

Laptop 2 owns `ui/`: it lists the returned trip IDs, posts a selection to the
API, and draws blue GPS points and red predicted segments with built-in SVG.
It deliberately has no model artifact, label file, or fallback prediction.

## Viva explanation

The classifier answers “which roads look locally plausible?” The HMM answers
“which sequence of those locally plausible roads can be driven through the
directed road network at the observed pace?” Viterbi finds the globally
highest-scoring such sequence efficiently by retaining only the best partial
path ending at each candidate state.

## Actual permitted-development results

The student-test run produced 19,163 observations across 10 trips. Mean
classifier-only arc accuracy was 82.2159%. Mean HMM/Viterbi decoded arc accuracy
was 81.9702%, a mean change of -0.2457 percentage points. Mean decoded plausible
transition rate was 99.6444%, versus 92.9055% for independent classifier choices.
No trip lost observations after the decoder reset fix; every trip's `n_decoded`
equals `n_observations` in `results/phase4/student_test_per_trip.csv`.
