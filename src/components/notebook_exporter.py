"""
notebook_exporter.py
====================
Generates a downloadable Jupyter Notebook (.ipynb) tailored to the user's
dataset and pipeline, using DeepSeek V3.2 via Oxlo AI.
"""
from __future__ import annotations

import os
import sys
import json
from typing import Any

import openai
from pymongo import MongoClient
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo

load_dotenv()

_MONGO_URL    = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_DB_NAME      = "clarityAI_database"
_LLM_MODEL    = "deepseek-v3"   # DeepSeek V3 via Oxlo
_OXLO_BASE    = "https://api.oxlo.ai/v1"


def _get_client():
    return openai.OpenAI(base_url=_OXLO_BASE, api_key=os.environ.get("OXLO_API_KEY", ""))


def _get_ml_results(filename: str) -> dict | None:
    db  = MongoClient(_MONGO_URL)[_DB_NAME]
    col = db["ml_results"]
    return col.find_one({"filename": filename}, sort=[("_id", -1)])


def _get_dataset_insights(filename: str) -> dict | None:
    db  = MongoClient(_MONGO_URL)[_DB_NAME]
    col = db["dataset_insights"]
    return col.find_one({"filename": filename}, sort=[("stored_at", -1)])


def generate_notebook(filename: str, target_column: str = "") -> dict:
    """
    Generate a complete Jupyter Notebook for the user's dataset + pipeline.
    Returns the notebook as a dict (ready for json.dumps).
    """
    try:
        # Gather context
        ml_doc       = _get_ml_results(filename)
        insight_doc  = _get_dataset_insights(filename)

        # Try to get df shape
        try:
            df    = load_dataframe_from_mongo(filename)
            shape = df.shape
            cols  = df.columns.tolist()
            sample = df.head(3).to_dict(orient="records")
        except Exception:
            shape  = (0, 0)
            cols   = []
            sample = []

        problem_type  = ml_doc.get("problem_type", "classification") if ml_doc else "classification"
        feature_plan  = ml_doc.get("feature_plan", {}) if ml_doc else {}
        best_model    = ml_doc.get("best_model", "RandomForest") if ml_doc else "RandomForest"
        ai_summary    = insight_doc.get("ai_insights", {}).get("summary", "") if insight_doc else ""
        target_col    = target_column or (ml_doc.get("target_column", "") if ml_doc else "")

        context = {
            "filename":     filename,
            "shape":        list(shape),
            "columns":      cols[:20],   # limit for prompt size
            "target":       target_col,
            "problem_type": problem_type,
            "feature_plan": feature_plan,
            "best_model":   best_model,
            "ai_summary":   ai_summary,
            "sample":       sample[:2],
        }

        prompt = f"""
You are an expert data scientist. Generate a complete, executable Jupyter Notebook for this ML project.

Dataset context:
{json.dumps(context, default=str, indent=2)}

The notebook must include these sections as separate cells:
1. Imports and setup
2. Data loading (from MongoDB GridFS using pymongo + pandas)
3. Exploratory Data Analysis (descriptive stats, null checks, dtypes)
4. Data Preprocessing (handle nulls, encode categoricals, scale numerics)
5. Feature Engineering (based on feature_plan above)
6. Model Training (train {best_model} for {problem_type})
7. Model Evaluation (appropriate metrics for {problem_type})
8. Results Summary and Next Steps

Return ONLY a valid JSON object representing the notebook. Use this exact schema:
{{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {{"kernelspec": {{"display_name": "Python 3", "language": "python", "name": "python3"}}, "language_info": {{"name": "python", "version": "3.10.0"}}}},
  "cells": [
    {{"cell_type": "markdown", "metadata": {{}}, "source": ["# Title"], "id": "cell1"}},
    {{"cell_type": "code", "execution_count": null, "metadata": {{}}, "outputs": [], "source": ["# code here\\n", "import pandas as pd"], "id": "cell2"}}
  ]
}}

Make each code cell realistic and runnable. Use actual column names from the dataset.
Respond ONLY with the JSON — no preamble, no markdown fences.
"""
        client   = _get_client()
        response = client.chat.completions.create(
            model=_LLM_MODEL,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a Python/ML expert. Return ONLY valid JSON for a Jupyter notebook."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else raw
            if raw.strip().lower().startswith("json"):
                raw = raw.strip()[4:]
        raw = raw.strip()

        notebook = json.loads(raw)
        logging.info("Notebook generated for '%s'", filename)
        return notebook

    except Exception as e:
        # Fallback: return a basic notebook template
        logging.warning("DeepSeek notebook generation failed (%s). Using fallback template.", e)
        return _fallback_notebook(filename, target_column or "", cols if 'cols' in dir() else [])


def _fallback_notebook(filename: str, target: str, columns: list) -> dict:
    """Return a minimal but valid notebook when LLM generation fails."""
    col_str = str(columns[:10])
    cells = [
        _md_cell("# ML Pipeline Notebook", "cell_title"),
        _md_cell(f"**Dataset:** `{filename}`  \n**Target column:** `{target}`", "cell_meta"),
        _code_cell(
            "# 1. Imports\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
            "import seaborn as sns\nfrom pymongo import MongoClient\nimport gridfs, io\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.metrics import accuracy_score, classification_report",
            "cell_imports"
        ),
        _code_cell(
            f"# 2. Load data from MongoDB GridFS\nfrom pymongo import MongoClient\n"
            f"import os\nmongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')\n"
            f"client = MongoClient(mongo_uri)\n"
            f"db = client['clarityAI_database']\nfs = gridfs.GridFS(db)\n"
            f"grid_out = fs.find_one({{'filename': '{filename}'}}, sort=[('uploadDate', -1)])\n"
            f"df = pd.read_csv(io.BytesIO(grid_out.read()))\n"
            f"print(df.shape)\ndf.head()",
            "cell_load"
        ),
        _code_cell("# 3. EDA\nprint(df.dtypes)\nprint(df.isnull().sum())\ndf.describe()", "cell_eda"),
        _code_cell(
            f"# 4. Preprocessing\ndf = df.dropna()\n"
            f"X = df.drop(columns=['{target}'])\n"
            f"y = df['{target}']\n"
            f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "cell_prep"
        ),
        _code_cell(
            "# 5. Train Model\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)",
            "cell_train"
        ),
        _code_cell(
            "# 6. Evaluate\nprint('Accuracy:', accuracy_score(y_test, y_pred))\nprint(classification_report(y_test, y_pred))",
            "cell_eval"
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def _md_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source], "id": cell_id}


def _code_cell(source: str, cell_id: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines, "id": cell_id}
