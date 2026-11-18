
# PathSnap: distributed GPS map matching

This repository reconstructs a plausible road-segment sequence from noisy GPS
observations. It keeps the four supplied data files per trip separate from
derived labels, candidate features, model scores, and decoded routes.

## Architecture

```text
Laptop 2: ui/index.html  -- HTTP -->  Laptop 1: FastAPI
                                           |
                           raw GPS + road graph + candidate features
                                           |
                              tuned Extra Trees candidate scorer
                                           |
                                HMM / Viterbi route decoder
                                           |
                           prediction-only route response to the UI
```

The UI never receives reference-route labels. During final evaluation, any
evaluator reference data remains on Laptop 1; no prediction code uses it.

## Repository layout

- `utils/`, `data_processing/`, `eda/`: raw-data loading, validation, labels,
  and geographical visualisation.
- `map_matching/`: candidate generation, features, candidate model, and HMM
  decoder.
- `scripts/`: reproducible Phase 2, Phase 3, and permitted-development Phase 4
  evaluation runners.
- `backend/`: FastAPI service for Laptop 1.
- `ui/`: dependency-free evaluator UI for Laptop 2.
- `tests/`: pipeline, model, decoder, and API tests.
- `docs/`: interfaces, Phase 4 design, deployment, and report sources.

## Setup

Use Python 3.10–3.12 (the original project was validated in its `lab_env`).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

The saved Phase 3 model is `results/phase3/final_model.joblib`. Recreate all
derived work from the repository root with:

```bash
python -m data_processing.validate
python -m data_processing.label_alignment
python -m scripts.run_phase2
python -m scripts.run_phase3
python -m scripts.finalize_phase3
```

`data/processed/` is intentionally ignored because it is reproducible from
the raw files. Do not add passwords, API keys, or other secrets.

## HMM evaluation on permitted development data

The comparison runner accepts only `train` and `student_test`; it refuses to
make an evaluator partition part of parameter selection or model development.

```bash
python -m scripts.evaluate_phase4 --split student_test
# faster smoke check
python -m scripts.evaluate_phase4 --split student_test --trip 00000000
```

It writes per-trip and aggregate candidate-only versus decoded-route metrics to
`results/phase4/`. See `docs/PHASE4.md` for metric definitions and limitations.

## Two-laptop run

On Laptop 1, place the model, raw road-network data, and this repository. Set
the evaluator split only to the server-side folder supplied by the organisers,
then expose the API on the LAN:

```bash
export PATHSNAP_EVALUATOR_SPLIT=student_test  # replace only for official server-side evaluator data
export PATHSNAP_CORS_ORIGINS=http://LAPTOP_2_LAN_IP:8080
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

On Laptop 2, edit the single non-secret URL in `ui/config.js` to Laptop 1's
LAN address, for example `http://192.168.1.42:8000`, then serve the UI:

```bash
python -m http.server 8080 --directory ui
```

Open `http://LAPTOP_2_LAN_IP:8080`. Details, firewall checks, and the honest
deployment-verification status are in `docs/DEPLOYMENT.md`.

## Tests

```bash
pytest -q
```

The API tests use a fake service; they do not access labels. Decoder tests
cover network continuity, missing candidate observations, and GPS-gap resets.
