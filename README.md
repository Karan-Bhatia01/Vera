# Vera — AI-Powered Data Intelligence & AutoML

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)](https://flask.palletsprojects.com/) [![License](https://img.shields.io/badge/License-MIT-green)](#) [![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-yellow)](https://vera-5wgh.onrender.com)

---

## What Vera Does

**Upload a CSV. Get a trained ML model, visual insights, and an AI that answers questions about your data — no code required.**

Most people with data can't build ML models without learning Python, pandas, and scikit-learn. Vera removes that barrier. It handles everything: data cleaning, feature engineering, model training, and gets performance metrics you can actually use. And it does it intelligently — with AI orchestrating decisions at every step.

---

## Why I Built This

I was stuck in that gap where I had data but didn't want to spend hours writing preprocessing pipelines and model experiments. The tools that exist are either "just for data nerds" (Jupyter) or "just for big enterprises" (fancy BI tools). Vera sits in the middle: powerful enough that it actually trains real models, accessible enough that anyone can use it, and smart enough that it learns from your data to make better decisions along the way.

The real breakthrough is using multiple AI models in a chain — each one doing what it's best at, then passing its output to the next. One model analyzes your schema and flags data quality issues. Another decides how to preprocess. A third explains what each feature means. Separately, they're helpful. Together, they're powerful.

---

## Live Demo

**[https://vera-5wgh.onrender.com](https://vera-5wgh.onrender.com)**

Upload a CSV file (any size up to 50MB, any schema). Vera will analyze it, show you charts with AI-powered insights, let you train models, compare them head-to-head, and download the best one as a `.pkl` file or export the entire workflow as a Jupyter notebook.

### Example Workflows:
- **Customer churn prediction** → Upload customer data, pick `churn_flag` as target, Vera trains 10 models and shows you which features matter most
- **House price regression** → Upload real estate data, select price column, get a ranking of models and SHAP explanations for every prediction
- **Data quality audit** → Just upload your data, skip ML if you want — Vera shows you exactly what's wrong and suggests fixes

---

## Project Video

https://github.com/user-attachments/assets/5de4be8b-4a96-4fea-af48-5e126a7f131b

---

## Features

### 1. Intelligent Dataset Analysis  
Vera loads your CSV and runs `AnalysisExplainer` — it computes descriptive statistics (mean, std, quantiles, nulls), detects duplicates, profiles dtypes, and sends everything to the LLM. The LLM (`llama-3.2-3b`) returns structured insights: summary, data quality flags per column, and actionable next steps. All computed stats come from Python — the AI just interprets them.

### 2. Exploratory Data Analysis (EDA) with AI Insights
Generates 12+ charts automatically: distributions, boxplots, value counts, correlation heatmaps, scatter plots. Charts render instantly. When you click "analyze," each chart image is sent to a vision model (`mistral-7b`) that describes what it sees, flags anomalies, and suggests what to investigate next. No batch processing — analysis happens on-demand, so the page loads fast.

### 3. AI-Guided Data Preprocessing
Missing values? Vera asks the LLM: "Given these null columns, what's the single best strategy?" The LLM returns suggestions (mean imputation, forward fill, drop, etc.). You can override, but the AI gives you a smart starting point. Same for detecting duplicates, dropping useless columns, and deciding which columns to encode as categorical vs numeric.

### 4. Automatic ML Pipeline
Here's where it gets real. You pick a target column. Vera:
1. **Decides feature engineering** — LLM gets cardinality of each column and returns a plan: drop IDs, ordinal-encode ordered data, one-hot-encode low-cardinality features, keep numeric as-is
2. **Auto-detects problem type** — If target is float/int with 20+ unique values, it's regression. Otherwise classification
3. **Preprocesses** — Scales numerics, encodes categoricals, handles missing values using your chosen strategy
4. **Trains 10+ models in parallel** — LogisticRegression, RandomForest, GradientBoosting, SVM, KNN, NaiveBayes, XGBoost (if installed), LightGBM (if installed), DecisionTree, Ridge, Lasso
5. **Evaluates** — For classification: accuracy, F1, precision, recall, ROC-AUC, confusion matrix. For regression: R², RMSE, MAE
6. **Generates SHAP explanations** — Shows which features matter most via mean |SHAP values|
7. **Saves everything** — Models stored as `.pkl` in MongoDB GridFS, metrics in the database

### 5. Model Comparison & Download
See all models ranked by their primary metric (accuracy for classification, R² for regression). Download any as a `.pkl`. Best model gets a ⭐ badge.

### 6. Feature Importance (SHAP)
Bar chart showing the average absolute SHAP value for each feature. Tree-based models use `TreeExplainer` (fast). Other models use `KernelExplainer` with a background sample. Tells you exactly which features your model relies on.

### 7. Jupyter Notebook Export
Click "Export Notebook" and get a fully executable `.ipynb` with sections: imports, data loading, EDA, preprocessing, feature engineering, model training, evaluation, and results. Uses `deepseek-v3` via Oxlo to generate realistic, runnable code. All imports and function calls work right out of the box.

### 8. Dataset Q&A Chatbot (RAG)
Ask questions like "What's the average salary?" or "Which city has the most customers?" Vera loads dataset context (shape, dtypes, descriptive stats, value distributions) and sends it with your question to `deepseek-r1-8b`. The LLM reasons over the data and returns answers grounded in actual values, not hallucinations.

### 9. Persistent Storage
All datasets, analyses, and ML results live in MongoDB. Upload the same file twice? Vera knows. Run the same target column on different preprocessings? Each result is stored. Everything queryable.

---

## How Oxlo.ai Models Are Used — The Multi-Step Pipeline

This is the architecture that makes Vera actually intelligent. Instead of calling one model for everything, different Oxlo.ai models handle different tasks. Each one gets context, outputs something specific, and the next model builds on it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VERA AI-POWERED PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

User uploads CSV (e.g., customer_data.csv)
        ↓
  [STAGE 1: ANALYSIS]
        ↓
  ✨ llama-3.2-3b analyzes schema
     Input:  shape, columns, dtypes, null counts, describe stats
     Output: {"summary": "...", "quality_flags": [...], "column_insights": [...]}
     Use:    Displayed to user on /info page; stored in MongoDB
        ↓
  [STAGE 2: CHART INSIGHTS]
        ↓
  ✨ Vera generates ~12 charts (matplotlib/seaborn, no LLM)
     (Distribution, Boxplot, Value Counts, Correlation, Scatter, etc.)
        ↓
  ✨ mistral-7b vision model (on-demand)
     Input:  Base64-encoded PNG of each chart
     Output: {"represents": "...", "key_findings": [...], "anomalies": [...]}
     Use:    Displayed below each chart in EDA results
        ↓
  [STAGE 3: PREPROCESSING]
        ↓
  ✨ llama-3.2-3b guesses missing value strategy
     Input:  {"null_values": {"col_A": 230, "col_B": 5}, ...}
     Output: {"col_A": "mean", "col_B": "drop", ...}
     Use:    Suggests imputation methods (user can override)
        ↓
  [STAGE 4: FEATURE ENGINEERING]
        ↓
  ✨ llama-3.2-3b (reused) decides column transformations
     Input:  Column names, cardinality, dtypes, target column
     Output: {"drop": [...], "ordinal": {...}, "onehot": [...], "numeric": [...]}
     Use:    Auto-configures sklearn ColumnTransformer pipeline
     Why:    Without this, you'd hand-tune every dataset. LLM does it in 1 second.
        ↓
  [STAGE 5: MODEL TRAINING]
        ↓
  ✨ Scikit-Learn (10+ models trained in parallel)
     No LLM here — pure Python for speed and accuracy
     Each model gets same preprocessed X_train, y_train
     Returns: Predictions, metrics (accuracy, F1, confusion matrices, etc.)
        ↓
  [STAGE 6: EXPLAINABILITY]
        ↓
  ✨ SHAP TreeExplainer or KernelExplainer
     Input:  Trained model, X_test, feature names
     Output: Feature importance bar chart (SHAP mean |values|)
     Use:    User sees: "Feature A drives 35% of model decisions"
        ↓
  [STAGE 7: RESULTS SUMMARY]
        ↓
  ✨ deepseek-r1-8b generates comprehensive summary
     Input:  {"filename": "...", "best_model": "RandomForest", ...}
     Output: Multi-section markdown: Performance metrics, key insights, recommendations
     Use:    Displayed on /ml_results page; user sees structured interpretation
        ↓
  [STAGE 8: NOTEBOOK GENERATION]
        ↓
  ✨ deepseek-v3 generates Jupyter notebook
     Input:  {"shape": [...], "columns": [...], "feature_plan": {...}, ...}
     Output: Complete .ipynb with 8 executable cells
     Use:    User downloads and runs locally for reproducibility/tweaking
        ↓
  [STAGE 9: Q&A CHATBOT]
        ↓
  ✨ deepseek-r1-8b answers dataset questions
     Input:  Dataset context (stats, distributions) + user query
     Output: Grounded answer ("The median salary is $65,400. [Based on 500 rows]")
     Use:    Live chat widget; users explore data conversationally

```

### Why Multiple Models Matter

- **`llama-3.2-3b`** (Fast, general-purpose) — Dataset analysis, preprocessing decisions, feature plans
- **`mistral-7b`** (Vision) — Chart interpretation, anomaly detection in images
- **`deepseek-r1-8b`** (Reasoning, 8B) — Comprehensive summaries, Q&A over data context
- **`deepseek-v3`** (Larger, more capable) — Notebook code generation (needs more capacity)

Each is chosen for speed + capability at that step. One model can't do everything well. Chaining them means each step is optimized, outputs are cached in MongoDB so you don't re-run if you ask the chatbot twice, and the pipeline is debuggable (you can see what each model returned).

### What Each Model Receives & Returns

| Model | Endpoint | Input | Output | Example Use |
|-------|----------|-------|--------|------------|
| `llama-3.2-3b` | Dataset Analysis | `{"shape": (1000, 50), "null_values": {...}}` | `{"summary": "...", "quality_flags": [...]}` | Insights page |
| `mistral-7b` | Chart Analysis | Base64 PNG + title | `{"key_findings": [...], "anomalies": [...]}` | Chart popup |
| `deepseek-r1-8b` | Chatbot | `"What's the average X?" + dataset stats` | `"The average X is 42.5..."` | Live chat |
| `deepseek-v3` | Notebook | Feature plan + target + problem type | Jupyter notebook JSON | Export .ipynb |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Flask, Python 3.8+ |
| **Database** | MongoDB + GridFS (file storage) |
| **Data Science** | Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **AI APIs** | Oxlo.ai (OpenAI-compatible endpoint) |
| **Deployment** | Gunicorn + Render |

---

## Architecture (Brief)

```
┌──────────────────────────────────────────────────────────────┐
│                    USER BROWSER                               │
│           (Vera Web App - vera_landing.html)                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ↓
┌──────────────────────────────────────────────────────────────┐
│                  FLASK WEB SERVER (app.py)                    │
│  Routes: /upload, /info, /preprocessing, /ml, /analyse_chart │
│  API endpoints: /api/chat, /api/generate_summary, /api/shap   │
└──────────────────┬─────────────────────────┬──────────────────┘
                   │                         │
        ┌──────────┴─────────────┐           │
        ↓                        ↓           ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   MONGODB        │  │  OXLO.AI API     │  │  PYTHON MODULES  │
│  clarityAI_db    │  │  (OpenAI SDK)    │  │  (Scikit-Learn)  │
│  ├── dataset     │  │  • llama-3.2-3b  │  │  • MLPipeline    │
│  ├── ml_results  │  │  • mistral-7b    │  │  • DataPrep      │
│  └── insights    │  │  • deepseek-r1   │  │  • SHAP          │
│     (GridFS)     │  │  • deepseek-v3   │  │  • Chatbot       │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Flow**: User submits CSV → Flask stores in MongoDB GridFS → Data modules load & analyze → AI models called via Oxlo SDK → Results stored → Frontend renders.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- MongoDB instance (local or cloud, e.g., MongoDB Atlas)
- Oxlo.ai API key ([register here](https://oxlo.ai))

### Local Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd ClarityAI2.0

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file (see .env.example below)
cat > .env << 'EOF'
MONGO_URI=mongodb://localhost:27017/
OXLO_API_KEY=your-key-here
OXLO_BASE_URL=https://api.oxlo.ai/v1
SECRET_KEY=your-secret-key-here
EOF

# 5. Run the app
python app.py

# 6. Open browser to http://localhost:5000
```

### Environment Variables

```env
# Required
OXLO_API_KEY=sk_...                                 # Oxlo API key
MONGO_URI=mongodb://localhost:27017/                # MongoDB connection string

# Optional
SECRET_KEY=dev-key-change-in-production             # Flask session key
OXLO_BASE_URL=https://api.oxlo.ai/v1               # Oxlo API endpoint (default shown)
```

### Running with Gunicorn (Production)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --timeout 60
```

---

## Project Structure

```
ClarityAI2.0/
├── app.py                           # Flask entry point, route definitions
├── requirements.txt                 # Python dependencies
├── Procfile                         # Gunicorn command for Render
│
├── src/
│   ├── agenticLayer/
│   │   └── llm.py                   # AnalysisExplainer (calls llama-3.2-3b)
│   │
│   ├── components/
│   │   ├── data_ingestion.py        # CSV upload & GridFS storage
│   │   ├── data_info.py             # Basic dataset metadata
│   │   ├── eda_processing.py        # EDA charts + analysis
│   │   ├── ml_pipeline.py           # AutoML: preprocessing, training, eval
│   │   ├── rag_pipeline.py          # Chatbot using deepseek-r1-8b
│   │   ├── mongo_storage.py         # MongoDB persistence layer
│   │   ├── notebook_exporter.py     # Jupyter notebook generation (deepseek-v3)
│   │   ├── shap_explainer.py        # SHAP feature importance
│   │   └── job_store.py             # Async job tracking
│   │
│   ├── utils.py                     # LLM calls, chart rendering, JSON parsing
│   ├── logger.py                    # Logging setup
│   └── exception.py                 # Custom exception handling
│
├── templates/
│   ├── vera_landing.html            # Home page
│   ├── index.html                   # Upload & file browser
│   ├── info.html                    # Dataset analysis results
│   ├── eda_processing.html          # Preprocessing controls
│   ├── preprocessing_result.html    # EDA charts & analysis
│   ├── ml_input.html                # ML target selection
│   ├── ml_training.html             # Live training progress
│   ├── ml_results.html              # Model comparison & results
│   └── partials/
│       └── navbar.html              # Navigation bar (reused)
│
├── static/
│   ├── css/
│   │   ├── design-system.css        # Colors, spacing, typography
│   │   ├── layout.css               # Grid, flex, responsive
│   │   ├── components.css           # Buttons, inputs, cards
│   │   └── (other CSS files...)
│   └── js/
│       ├── vera-chatbot.js          # Chat UI logic
│       └── modern-ui.js             # General UI interactions
│
└── logs/                            # Application logs (auto-created)
```

---

## Key Modules Explained

### `AnalysisExplainer` (`src/agenticLayer/llm.py`)
Runs on dataset upload. Computes stats (shape, nulls, duplicates, describe, unique values) using pandas. Sends compacted JSON to `llama-3.2-3b` with aggressive payload reduction (max 50KB). LLM returns structured insights: summary, quality_flags (high/medium/low), column_insights, next_steps. All maths done by Python; AI does interpretation.

### `DataPreprocessing` (`src/components/eda_processing.py`)
Orchestrates EDA → preprocessing → chart generation. Calls `get_ai_insights()` for missing value strategy (LLM decides). Builds charts (no AI here, instant). Exposes `analyse_single()` for on-demand chart analysis via `mistral-7b` vision model.

### `MLPipeline` (`src/components/ml_pipeline.py`)
The core AutoML engine:
1. Calls `llm_feature_plan()` — LLM decides drop/ordinal/onehot/numeric transformations
2. `detect_problem_type()` — Auto-classifies as classification or regression
3. `preprocess()` — Applies plan, scales, encodes, splits (80/20)
4. `train_and_evaluate()` — Trains all models, computes metrics, extracts feature importance
5. `generate_shap_plots()` — SHAP bar chart + force plots
6. `run()` — Main entry point; returns results dict for rendering

### `RAG Chatbot` (`src/components/rag_pipeline.py`)
Simple but effective. Loads dataset context (shape, stats, distributions) from MongoDB and formats as readable text. Appends user query. Sends to `deepseek-r1-8b` with conversation history (last 3 turns). Returns answer grounded in data.

### `Notebook Exporter` (`src/components/notebook_exporter.py`)
Builds a prompt with ML results (feature plan, best model, problem type, dataset shape). Sends to `deepseek-v3`. Expects JSON notebook schema back. Includes fallback template if LLM generation fails.

---

## Registered Oxlo.ai Email

The actual developer's Oxlo.ai registered email:

```
Registered Oxlo.ai Email: [bhatiakaran168@example.com]
```


---

## Built for OxBuild Hackathon

Vera was built for the **OxBuild hackathon by Oxlo.ai** — a sprint to build with Oxlo's AI models. The key insight: don't call one model for everything. Chain them. Each model gets specific context, outputs something precise, and the next model builds on it. The result is faster inference, lower cost, and more intelligent decisions.

This README was written to show judges exactly how the code works, not to hide it behind marketing speak. If you want to understand the pipeline, read the model orchestration section. If you want to hack on it, the code is organized to make changes easy.

Deploy with: `gunicorn app:app` and a `.env` file with API keys. That's it.

---

Happy analyzing. 🚀
