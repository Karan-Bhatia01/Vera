# Vera — AI-Powered Data Analysis & AutoML

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-API-lightgrey)](https://flask.palletsprojects.com/) [![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev/)

**Upload a CSV. Get a profiled dataset, AI insights, interactive EDA charts, and a trained ML model — no code required.**

Vera runs a 3-stage pipeline: **profile → explore → model**. pandas and scikit-learn do all the maths; a Groq-hosted LLM only interprets the numbers and makes choices (which charts to draw, how to fill missing values, which features and models to use). Every LLM step has a fallback, so a slow or failing LLM never breaks the pipeline.

---

## Architecture

```
React SPA (frontend/)  ──JSON/HTTP──▶  Flask API (app.py, routes/*)
  JWT in localStorage                    @require_auth on data routes
  polls job status every 3s                     │
                     ┌──────────────────┬───────┴─────────┬───────────────────┐
                     ▼                  ▼                 ▼                   ▼
              MongoDB + GridFS     Groq LLM          src/ml/ AutoML      Job store
              datasets, insights,  (src/agents,      (scikit-learn)      (.runtime/jobs.json)
              users                 src/components/llm.py)
```

Long-running work (Data Info, EDA, ML training, ML analysis) runs in **background threads**. The API returns a job id immediately and the UI polls `/api/pipeline_status/<step>/<id>`.

---

## Features

### 1. Upload (`/upload`)
- CSV only, **max 25,000 rows × 20 columns**, **3 datasets per user**.
- Encoding (UTF-8, Latin-1, CP1252) and delimiter (`,` `;` tab `|`) are detected automatically. The same parser is used for upload and analysis.
- Uploading a filename you already have is rejected — delete the old one first.
- Delete a dataset from **Your Library** (trash icon on each card).

### 2. Data Info (`/dashboard/info`)
Shape, dtypes, null %, unique counts, duplicates, memory, numeric stats, outliers, skew, top categories, strongest correlations and cardinality flags — all computed in pandas. One Groq call (`AnalysisExplainer`) then returns structured insights: summary, business evaluation, quality flags, preprocessing and feature-engineering ideas, a recommended target, and **10 recommended charts**. Results are stored in MongoDB.

### 3. EDA (`/dashboard/eda`)
After you pick a target column on the Dashboard, the EDA job fills missing values (strategy chosen per column by an LLM agent, with median/mode fallback) and turns the 10 AI-recommended charts into **Chart.js** data: bar, line, scatter, pie, doughnut, histogram, radar, polar area and bubble. Each chart comes with the AI's written takeaway.

> EDA reads the charts produced by Data Info, so let Data Info finish first.

### 4. ML Modelling (`/dashboard/ml`)
1. **Feature plan** — LLM agent assigns each column to drop / ordinal / one-hot / numeric (dtype heuristic fallback).
2. **Problem type** — classification vs regression, detected from the target.
3. **Preprocess** — impute, scale, encode, 80/20 split, with guards against mislabelled columns and one-hot explosion.
4. **Model shortlist** — LLM agent picks 3–5 models from the registry (defaults if unavailable).
5. **Train & evaluate** — accuracy / F1 / precision / recall (+ ROC-AUC for binary) or R² / RMSE / MAE, plus feature importance.
6. **Simple Analysis** — on request, a background job asks the LLM for a plain-English executive summary while the metrics stay on screen.

### 5. Authentication
- Email + password (bcrypt) or **Google sign-in**. Both end with the backend issuing its own 24-hour HS256 JWT.
- Google: the frontend gets an access token via `@react-oauth/google`; the backend verifies it with Google's `userinfo` endpoint and creates the account on first sign-in.
- Emails are case-insensitive.
- `ProtectedRoute` checks token expiry locally, then confirms it with `/api/me`. Any 401 sends the user back to `/login`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, React Router 7, Vite, Tailwind CSS v4, Axios, Chart.js (react-chartjs-2), Framer Motion |
| Backend | Flask (blueprints), flask-cors, gunicorn |
| Auth | PyJWT, bcrypt, Google OAuth (`@react-oauth/google`) |
| Database | MongoDB + GridFS (pymongo) |
| ML | pandas, NumPy, scikit-learn (XGBoost / LightGBM used automatically if installed) |
| LLM | Groq (`groq` SDK), responses validated with pydantic |

---

## Project Structure

```
ClarityAI2.0/
├── app.py                     # Flask app factory + blueprint registration
├── requirements.txt
├── Procfile                   # gunicorn app:app
├── routes/
│   ├── auth.py                # /auth/signup, /auth/login, /auth/google, /api/me
│   ├── upload.py              # /api/upload
│   ├── dataset.py             # /api/stored_datasets, /api/insights/<f>, DELETE /api/datasets/<f>
│   ├── analysis.py            # /api/info/<f>, /api/run_info  (Data Info job)
│   ├── eda.py                 # /api/run_eda                  (EDA job)
│   ├── ml.py                  # /api/run_ml, /api/analyze_ml/<id>  (training + analysis jobs)
│   └── health.py              # /api/health (pings MongoDB)
├── services/
│   ├── auth_service.py        # signup / login / Google auth, JWT issue + verify
│   ├── auth_decorator.py      # @require_auth
│   └── upload_service.py      # upload limits + validation
├── src/
│   ├── core/connections.py    # shared Mongo / GridFS / Groq clients
│   ├── agents/                # single-call LLM agents (each with a fallback)
│   │   ├── llm_provider.py    # chat_completion / call_llm with 429 retry + backoff
│   │   ├── feature_engineering_agent.py
│   │   ├── model_selection_agent.py
│   │   └── missing_value_agent.py
│   ├── prompts/*.md           # system prompts for the agents
│   ├── components/
│   │   ├── llm.py             # AnalysisExplainer (dataset insights) + ML results explainer
│   │   ├── data_info.py       # dataset statistics
│   │   ├── data_ingestion.py  # GridFS storage, listing, ownership, delete
│   │   ├── eda_processing.py  # preprocessing + Chart.js chart data
│   │   ├── mongo_storage.py   # insights + users collections
│   │   └── job_store.py       # background job tracking (disk-backed)
│   ├── ml/                    # AutoML: dtype_utils, problem_type, models, preprocessing, training, pipeline
│   ├── utils.py               # GridFS → DataFrame, robust CSV parser, LLM JSON parsing
│   ├── logger.py              # logs/<timestamp>.log
│   └── exception.py
└── frontend/
    ├── index.html             # fonts: Manrope + JetBrains Mono
    ├── vite.config.js         # envDir: '../' (reads the root .env)
    └── src/
        ├── App.jsx            # routes (public vs ProtectedRoute)
        ├── api/client.js      # axios instance, auth header, 401 handling, session helpers
        ├── context/PipelineContext.jsx   # shared job polling + result cache
        ├── hooks/useTheme.js
        ├── components/        # landing/, dashboard/, auth/, charts/AestheticChart.jsx, ProtectedRoute, …
        └── pages/             # Login, Signup, Upload, dashboard/*, policies/*
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas)
- A Groq API key
- A Google OAuth client ID (for Google sign-in)

### Environment
Create a `.env` in the **project root**. The frontend reads it too, through `envDir: '../'`.

```env
# Required
MONGODB_URI=mongodb://localhost:27017/
GROQ_API_KEY=your-groq-key
JWT_SECRET=a-long-random-string        # the API refuses to start without it
VITE_GOOGLE_CLIENT_ID=your-google-oauth-client-id

# Optional
DB_NAME=clarityAI_database             # default shown
VITE_API_URL=http://localhost:5000     # where the frontend calls the API
GROQ_MODEL=openai/gpt-oss-120b         # model for the agents
GROQ_TIMEOUT=45                        # seconds per LLM call
GROQ_RATE_LIMIT_RETRIES=4              # retries on Groq HTTP 429
```

### Backend

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                     # http://127.0.0.1:5000  (health: /api/health)
```

### Frontend

```bash
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

### Production

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --threads 8 --timeout 120
cd frontend && npm run build      # static files in frontend/dist
```

> Use a **single worker** (with threads). The job store lives in each process's memory, so jobs started in one worker are not visible to another.

---

## API Overview

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/auth/signup`, `/auth/login`, `/auth/google` | Get a JWT |
| GET | `/api/me` | Verify the current token |
| POST | `/api/upload` | Upload a CSV (multipart `file`) |
| GET | `/api/stored_datasets` | The caller's datasets |
| DELETE | `/api/datasets/<filename>` | Delete one of the caller's datasets |
| GET | `/api/info/<filename>` | Quick stats + column list |
| GET | `/api/insights/<filename>` | Stored stats + AI insights |
| POST | `/api/run_info` · `/api/run_eda` · `/api/run_ml` | Start a background job |
| POST | `/api/analyze_ml/<ml_job_id>` | Start the ML executive-analysis job |
| GET | `/api/pipeline_status/<info\|eda\|ml\|analysis>/<job_id>` | Poll a job |
| GET | `/api/health` | API + MongoDB status |

Every route except auth and health needs `Authorization: Bearer <token>`.

---

## Notes for Contributors

- Never block a request thread on heavy work: start a job with `job_store` and poll it.
- LLM calls go through `src/agents/llm_provider.py`, which handles retries and `max_tokens`. Give prompts that return large JSON a generous token budget.
- Every LLM step needs a deterministic fallback.
- The ML pipeline assumes numpy dtypes: keep `normalize_dtypes` in front of any new preprocessing.
- Read stored CSVs with `load_dataframe_from_mongo` (it uses the same robust parser as upload).
