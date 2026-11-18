
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

The commands below are for Windows PowerShell.

### Laptop 1: API, model and road network

```powershell
cd C:\path\to\ML_Hackathon

py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q

# Put official server-side data in C:\path\to\ML_Hackathon\evaluator_test.
# DATA_ROOT is the parent directory of that evaluator_test folder.
$env:PATHSNAP_DATA_ROOT = "C:\path\to\ML_Hackathon"
$env:PATHSNAP_EVALUATOR_SPLIT = "evaluator_test"

# Replace with Laptop 2's actual LAN IP and UI port.
$env:PATHSNAP_CORS_ORIGINS = "http://192.168.1.50:8080"

# Find this laptop's IPv4 address.
ipconfig

# Run once in an elevated PowerShell if Windows Firewall blocks TCP 8000.
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound `
  -Protocol TCP -LocalPort 8000 -Action Allow

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Verify from Laptop 1 or Laptop 2:

```powershell
Invoke-RestMethod "http://192.168.1.40:8000/health"
Invoke-RestMethod "http://192.168.1.40:8000/api/trips"
```

Use the actual IPv4 address printed by `ipconfig`; the example `192.168.1.40`
is only a placeholder.

Keep the Laptop 1 terminal visible during the evaluator demonstration. It
logs every matching request and shows the API endpoint activity.

### Laptop 2: evaluator UI

```powershell
cd C:\path\to\ML_Hackathon

py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn
```

Edit `ui/config.js`:

```javascript
// Replace the address with Laptop 1's actual LAN IPv4 address.
window.PATHSNAP_API_BASE = "http://192.168.1.40:8000";
```

Then start and open the UI:

```powershell
python -m http.server 8080 --directory ui
# Open http://localhost:8080 in a browser on Laptop 2.
```

If Laptop 2 and its browser are different machines, replace `localhost` with
Laptop 2's own LAN IPv4 address when opening the UI.

Laptop 2 must call Laptop 1 over the LAN. It does not receive evaluator
reference routes or model artifacts.

## Tests

```bash
pytest -q
```

The API tests use a fake service; they do not access labels. Decoder tests
cover network continuity, missing candidate observations, and GPS-gap resets.
