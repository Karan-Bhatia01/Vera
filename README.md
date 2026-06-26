# Vera — AI-Powered Data Intelligence & AutoML

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-API-lightgrey)](https://flask.palletsprojects.com/) [![React](https://img.shields.io/badge/React-Frontend-61dafb)](https://react.dev/) [![License](https://img.shields.io/badge/License-MIT-green)](#)

---

## What Vera Does

**Upload a CSV. Get a profiled dataset, AI insights, automated EDA, and a trained ML model — no code required.**

Vera removes the gap between "I have data" and "I have a working model." It handles data profiling, cleaning, feature engineering, model selection, training, evaluation, and explainability — with LLM agents orchestrating the decisions at each step, and pure-Python doing all the maths.

---

## Architecture at a Glance

Vera is a **React single-page app** talking to a **Flask JSON API**. Long-running work (analysis, EDA, training) runs in **background threads** tracked by an async job store, so the UI polls for progress instead of blocking.

```
┌─────────────────────────┐     JSON over HTTP      ┌──────────────────────────────┐
│   React frontend         │  ───────────────────▶   │   Flask API (app.py)         │
│   (frontend/)            │  ◀───────────────────   │   routes/*.py blueprints     │
│   JWT in localStorage    │      poll job status    │   @require_auth on every API │
└─────────────────────────┘                          └───────────────┬──────────────┘
                                                                      │
                        ┌─────────────────────────┬───────────────────┼───────────────────────┐
                        ▼                         ▼                   ▼                       ▼
                 ┌──────────────┐         ┌──────────────┐    ┌──────────────┐        ┌──────────────┐
                 │  MongoDB +   │         │  LLM agents  │    │  src/ml/     │        │  Job store    │
                 │  GridFS      │         │ (Groq via    │    │  AutoML      │        │ (.runtime/    │
                 │  datasets,   │         │  src/agents) │    │  pipeline    │        │  jobs.json)   │
                 │  insights,   │         │              │    │              │        │               │
                 │  ml_results  │         │              │    │              │        │               │
                 └──────────────┘         └──────────────┘    └──────────────┘        └──────────────┘
```

---

## Features

### 1. Dataset Profiling & AI Insights (`/dashboard/info`)
Computes shape, dtypes, null %, unique counts, memory, per-column numeric stats, top categorical values, and the strongest correlations — all in pandas. That summary is sent to an LLM that returns a structured, **business-oriented** report: dataset summary, data-quality flags, per-column insights, recommended prediction target, feature-engineering ideas, and preprocessing steps. The AI only interprets; every number comes from Python.

### 2. Automated EDA with Business Insights (`/dashboard/eda`)
Generates distribution, boxplot, value-count, correlation, and scatter charts (matplotlib/seaborn — instant, no LLM). On demand, each chart image is sent to a vision LLM that frames findings as **business takeaways and drivers** plus concrete recommendations, not raw statistics.

### 3. Automatic ML Pipeline (`/dashboard/ml`)
Pick a target column and Vera:
1. **Feature plan** — an agent decides drop / ordinal / one-hot / numeric per column (with a dtype-based fallback if the LLM is unavailable).
2. **Problem type** — classification vs regression, detected safely from the target.
3. **Preprocess** — encode, scale, split (80/20), with guards against mislabelled columns and one-hot feature explosion.
4. **Model shortlist** — an agent picks 3–5 candidate models from the registry (defaults if the LLM is unavailable).
5. **Train & evaluate** — accuracy/F1/precision/recall/ROC-AUC (classification) or R²/RMSE/MAE (regression).
6. **Explain** — SHAP for the best model.
7. **Persist** — models to GridFS, metrics + best model to MongoDB.

The pipeline is **bulletproofed**: pandas extension/nullable dtypes (StringDtype, Int64, …) are normalized up front so a messy CSV can't crash training.

### 4. Authentication
Email/password auth issuing a JWT. The frontend stores the token and gates every protected page behind a `ProtectedRoute` that checks token presence **and expiry**; the backend enforces `@require_auth` on every API route.

### 5. Persistent Storage
Datasets (GridFS), AI insights, and ML results all live in MongoDB, scoped per user.

---

## LLM Agents

Each decision is a single, focused LLM call (no slow tool loops), wrapped to degrade gracefully — transient Groq rate limits (HTTP 429) are retried with backoff, and every agent has a deterministic fallback so the pipeline never hard-fails on the LLM.

| Agent (`src/agents/`) | Job | Fallback if LLM unavailable |
|------------------------|-----|------------------------------|
| `feature_engineering_agent` | Per-column transform plan | Dtype/cardinality heuristic plan |
| `model_selection_agent` | Shortlist 3–5 models | First 3–5 from the registry |
| `missing_value_agent` | Imputation strategy per column | — |
| `llm_provider` | Shared Groq client (retry/backoff, configurable model & `max_tokens`) | — |

Models and limits are configurable via env (`GROQ_MODEL`, `GROQ_INSIGHTS_MODEL`, `GROQ_INSIGHTS_MAX_TOKENS`, `GROQ_TIMEOUT`, …).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React, React Router, Vite, Tailwind, Axios |
| **Backend** | Flask (blueprints), Python 3.10+ |
| **Auth** | JWT (PyJWT), bcrypt |
| **Database** | MongoDB + GridFS |
| **Data Science** | Pandas, NumPy, scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn |
| **LLMs** | Groq (agents & insights); OpenAI-compatible vision endpoint for charts |

---

## Project Structure

```
ClarityAI2.0/
├── app.py                      # Flask app factory + blueprint registration
├── requirements.txt
│
├── routes/                     # API blueprints (all @require_auth)
│   ├── auth.py                 # login / signup → JWT
│   ├── upload.py               # CSV upload → GridFS
│   ├── dataset.py              # list datasets, fetch stored insights
│   ├── analysis.py             # Data Info job + AI insights
│   ├── eda.py                  # EDA job (charts + chart analysis)
│   ├── ml.py                   # ML training job
│   ├── chat.py                 # dataset Q&A
│   └── health.py
│
├── services/
│   ├── auth_service.py         # JWT issue/verify
│   ├── auth_decorator.py       # @require_auth
│   └── upload_service.py       # upload validation + storage
│
├── src/
│   ├── agents/                 # focused LLM agents (Groq)
│   │   ├── llm_provider.py     # shared client: retry/backoff, max_tokens
│   │   ├── feature_engineering_agent.py
│   │   ├── model_selection_agent.py
│   │   └── missing_value_agent.py
│   │
│   ├── ml/                     # the AutoML pipeline, split into pieces
│   │   ├── pipeline.py         # MLPipeline orchestrator
│   │   ├── dtype_utils.py      # normalize extension/nullable dtypes (bulletproofing)
│   │   ├── problem_type.py     # classification vs regression
│   │   ├── models.py           # model registry per problem type
│   │   ├── preprocessing.py    # feature-plan application, encode/scale/split
│   │   ├── training.py         # fit + evaluate + metrics
│   │   └── persistence.py      # save models (GridFS) + metrics (Mongo)
│   │
│   ├── components/
│   │   ├── data_ingestion.py   # GridFS storage + ownership
│   │   ├── data_info.py        # dataset stats (numeric stats, correlations, …)
│   │   ├── eda_processing.py   # EDA charts + on-demand chart analysis
│   │   ├── mongo_storage.py    # insights persistence (shared Mongo client)
│   │   ├── shap_explainer.py   # SHAP plots
│   │   └── job_store.py        # async job tracking (disk-backed)
│   │
│   ├── utils.py                # GridFS/df loading, chart rendering, chart-analysis prompt
│   ├── logger.py
│   └── exception.py
│
└── frontend/                   # React SPA
    └── src/
        ├── App.jsx             # routes (public vs ProtectedRoute)
        ├── api/client.js       # axios instance (auth header, timeout)
        ├── context/PipelineContext.jsx   # shared polling + result cache
        ├── components/
        │   ├── ProtectedRoute.jsx
        │   └── dashboard/      # Sidebar (with sign-out), layout
        └── pages/
            ├── Login.jsx, Signup.jsx, Upload.jsx
            └── dashboard/      # DashboardHome, DataInfo, DataEDA, MLModelling
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ (for the frontend)
- MongoDB (local or Atlas)
- A Groq API key (and an OpenAI-compatible vision endpoint key for chart analysis)

### Backend

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# .env
cat > .env << 'EOF'
MONGO_URI=mongodb://localhost:27017/
GROQ_API_KEY=your-groq-key
SECRET_KEY=change-me
EOF

python app.py                   # http://127.0.0.1:5000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173 (proxies to the API on :5000)
```

### Key Environment Variables

```env
# Required
MONGO_URI=mongodb://localhost:27017/
GROQ_API_KEY=...
SECRET_KEY=...                          # JWT signing

# Optional (sensible defaults)
GROQ_MODEL=groq/compound                # default agent model
GROQ_INSIGHTS_MODEL=llama-3.3-70b-versatile
GROQ_INSIGHTS_MAX_TOKENS=3000           # room for the full insights JSON
GROQ_TIMEOUT=45
```

### Production

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

---

## How the Pieces Fit (request → result)

1. **Login** → backend issues a JWT → stored in `localStorage`. `ProtectedRoute` guards `/upload` and `/dashboard/*`.
2. **Upload** → CSV stored in GridFS, tagged with the owner. The dataset list shows every owned CSV (max 3), analyzed or not.
3. **Data Info** → background job computes stats + AI insights, cached in Mongo.
4. **EDA** → background job builds charts; chart analysis is on-demand and business-framed.
5. **ML** → background job runs the `src/ml` pipeline; the UI polls progress and renders the model comparison, feature importance, and SHAP.

---

## Notes for Contributors

- All heavy work is async via `job_store`; never block a request thread on training.
- LLM calls go through `src/agents/llm_provider.call_llm` — it handles retries and `max_tokens`; give large-JSON prompts a generous token budget.
- The ML pipeline assumes **numpy-friendly dtypes**; `normalize_dtypes` enforces that at load — keep it in front of any new preprocessing.
- Every agent must have a deterministic fallback so a degraded LLM never breaks the pipeline.

---

Happy analyzing. 🚀
