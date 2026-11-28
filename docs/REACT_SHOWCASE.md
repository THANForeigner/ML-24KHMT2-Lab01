# React GPA showcase

Chạy toàn bộ ứng dụng bằng một lệnh Docker Compose: xem [hướng dẫn Docker](DOCKER.md).

## 1. Scope

All three approved screens (**Tổng quan**, **Thử dự đoán**, **So sánh**) are implemented in `frontend/`, with a read-only FastAPI service in `backend/`. The original Streamlit application remains available; its piecewise loader is shared with the API.

- Light/dark themes and collapsible sidebar, saved locally.
- Six models and all five Simple variants.
- All 12 fields, with unused fields identified rather than removed.
- Explicit prediction submission; edits never trigger inference.
- Submitted model/profile snapshots and stale-result notices.
- Real sensitivity curves from the selected saved estimator.
- Raw predictions, including values outside GPA 0–4.
- A six-model overview catalog with direct entry into the selected model.
- Explicit six-model comparison on the same profile, retained across tab navigation.
- Real validation MAE/RMSE/R² and actual-versus-predicted scatter data.

The default page is **Tổng quan**. Direct links use `#overview`, `#prediction` and `#comparison`; browser back/forward is supported. Profile, Simple variant and submitted results remain in memory while navigating. Reloading resets the profile/results; theme and sidebar preferences persist locally.

The comparison never treats a higher predicted GPA as evidence of greater accuracy. Simple artifacts fitted on train plus validation are explicitly excluded from validation scores. The other metrics use all 7,500 validation rows; scatter plots display a deterministic sample of at most 800 points. Neither screen uses the held-out test set.

## 2. Development

From the repository root, using Python 3.11+ and Node 22.14+:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-web.txt -e .
.venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to port 8000. On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## 3. Single-process serving

Build the frontend before starting the API:

```bash
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The API serves `frontend/dist/` on the same origin. This is a local showcase, not a hardened public service; add request limits, TLS and access controls before public exposure.

## 4. API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Process health |
| `GET /api/bootstrap` | Models, field options/domains and initial training profile |
| `GET /api/sample?index=1` | Complete training profile |
| `POST /api/predict` | Prediction and sensitivity curve |
| `POST /api/compare` | Six-model prediction snapshot from one profile and Simple variant |
| `GET /api/validation` | Eligible validation scores and exclusion/unavailability explanations |
| `GET /api/agreement?model=Ridge` | Deterministic validation scatter sample and uncut axis domain |

Prediction requests contain `model`, `variant` (default `M03`), `profile` (all 12 fields) and `variable` (a supported sensitivity field). Categorical values use original dataset identifiers; paid subscription is a JSON boolean. The response contains the submitted model/profile, raw `gpa`, `out_of_range`, and `curve` points (`x`, `gpa`).

Invalid requests return 422; artifact/inference failures return 503. Paths are server-owned, never supplied by callers. The service does not train, write artifacts or read the held-out test split. It checks the training dataset SHA-256 before using it.

Comparison requests contain `profile` and `variant` (default `M03`). Each returned row includes `model`, nullable `gpa`, `out_of_range` and nullable `error`; one unavailable artifact does not remove the other results. Validation datasets are also checksum-verified. Validation statistics and fitted estimators are cached in process; restart the API after intentionally replacing artifacts or datasets.

## 5. Verification

```bash
.venv/bin/python -m pip install -r requirements-showcase.txt
.venv/bin/python -m pytest tests --ignore=tests/test_notebook_smoke.py -q
cd frontend
npm test
npm run build
npx playwright install chromium
npm run test:e2e
```

Browser tests start the API and Vite automatically when not already running. They exercise real HTTP prediction/comparison, navigation, actual validation data and widths from 320 to 1440px in both themes.

Saved artifacts were produced with different scikit-learn versions (including 1.7.2 and 1.8.0); the current environment can emit `InconsistentVersionWarning`. Artifact parity is tested, but does not guarantee compatibility for all inputs. Do not reserialize or retrain artifacts just to suppress warnings.

Synthetic data and model estimates do not establish a causal effect of AI use on GPA.
