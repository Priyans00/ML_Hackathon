# Two-laptop deployment and verification record

## Required layout

Laptop 1 hosts the repository, `dataset/`, `results/phase3/final_model.joblib`,
the FastAPI process, road graph construction, and HMM/Viterbi decoding. Laptop
2 hosts only the static evaluator UI and sends the chosen trip ID to Laptop 1.

## Before the demonstration

1. Install the same project requirements on both laptops; Laptop 1 must have
   the complete data/model repository.
2. Put the organisers' evaluator partition only on Laptop 1. Set
   `PATHSNAP_EVALUATOR_SPLIT` to its server-side folder name. Do not copy its
   reference routes or labels to Laptop 2.
3. Find Laptop 1's LAN IPv4 address (`ipconfig` on Windows or `ip addr` on
   Linux). Allow inbound TCP 8000 on its private-network firewall.
4. On Laptop 1, start `uvicorn backend.main:app --host 0.0.0.0 --port 8000`.
   Check `http://LAPTOP_1_IP:8000/health` from Laptop 2.
5. Set `window.PATHSNAP_API_BASE` in `ui/config.js` on Laptop 2 to that exact
   address. Start `python -m http.server 8080 --directory ui` and open it in a
   browser.
6. Select each offered trip at least once. Preserve the Laptop 1 terminal logs
   and screenshots of a successful API response and the UI, as required by the
   assignment.

## Configuration safety

`PATHSNAP_CORS_ORIGINS` should be the Laptop 2 UI origin (for example,
`http://192.168.1.50:8080`) for the demonstration. The API URL is not a secret.
Never put API keys, passwords, or evaluator reference labels in this repository
or in the UI.

## Verification status

**Not yet verified on two physical laptops in this checkout.** Local API and
integration tests are recorded separately when run; they are not evidence of a
cross-laptop deployment. Record date, device IPs, all-trip outcome, screenshots,
and any firewall changes here after the real demonstration.
