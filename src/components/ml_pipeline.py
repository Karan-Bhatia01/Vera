from __future__ import annotations

import io
import os
import sys
import json
import pickle
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from pymongo import MongoClient
import gridfs
from dotenv import load_dotenv

# ── Sklearn ────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB = True
except ImportError:
    _XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LGBM = True
except ImportError:
    _LGBM = False

from src.logger import logging
from src.exception import CustomException
from src.utils import load_dataframe_from_mongo, llm_agent, parse_json_response

load_dotenv()

# ── MongoDB ────────────────────────────────────────────────────────────────────
_MONGO_URL = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
_DB_NAME   = "clarityAI_database"

# ── Constants ──────────────────────────────────────────────────────────────────
_TEST_SIZE        = 0.2
_RANDOM_STATE     = 42
_CLASS_UNIQUE_MAX = 20   # int columns with <= this many unique vals → classification


class MLPipeline:
    """
    Fully automatic ML pipeline guided by LLM.

    Workflow
    --------
    1. LLM decides feature engineering (drop / ordinal / onehot / numeric)
    2. Auto-detect problem type (classification vs regression)
    3. Preprocess: encode + scale + split
    4. Train all applicable model families
    5. Evaluate each model
    6. Save models (.pkl) + metrics to MongoDB
    7. Return full results dict for rendering
    """

    def __init__(self, filename: str, target_column: str) -> None:
        try:
            self.filename      = filename
            self.target_column = target_column

            client    = MongoClient(_MONGO_URL)
            self.db   = client[_DB_NAME]
            self.fs   = gridfs.GridFS(self.db)
            self.col  = self.db["ml_results"]

            self.df = load_dataframe_from_mongo(filename)
            logging.info("MLPipeline loaded '%s' — shape %s", filename, self.df.shape)

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 1. LLM feature decision ────────────────────────────────────────────────
    def llm_feature_plan(self) -> dict[str, Any]:
        """
        Ask the LLM which columns to drop, ordinal-encode, one-hot encode,
        or keep as numeric. Returns a structured JSON plan.
        
        AGGRESSIVE PAYLOAD REDUCTION:
        - Only include cardinality for first 10 columns (not all)
        - Use compact JSON (no indentation)
        - Exclude dtypes and null_counts (minimal summary only)
        """
        try:
            df = self.df
            columns = df.columns.tolist()
            nunique_dict = df.nunique().to_dict()
            
            # Only include cardinality for first 10 columns for brevity
            cols_sample = columns[:10] if len(columns) > 10 else columns
            
            summary = {
                "shape": df.shape,
                "target": self.target_column,
                "cols": len(columns),
                "card": {c: nunique_dict[c] for c in cols_sample},
            }

            prompt = textwrap.dedent(f"""
                Dataset: {df.shape[0]} rows, {len(columns)} columns. Target: '{self.target_column}'.
                
                Decide: drop, ordinal-encode, one-hot-encode, or keep numeric.
                Rules: Never include target. Drop IDs/names. Ordinal for ordered data.
                One-hot if cardinality <= 15. Drop if > 15. Keep numeric as-is.
                
                Return ONLY valid JSON, no explanation:
                {{"drop": [], "ordinal": {{}}, "onehot": [], "numeric": []}}
            """).strip()

            llm_result   = llm_agent(
                prompt=prompt,
                role="ML Engineer",
                context=json.dumps(summary, separators=(',', ':'), default=str),
            )
            raw = llm_result.get("response", "")
            plan = parse_json_response(raw) if isinstance(raw, str) else raw

            # Safety: remove target from all feature lists
            for key in ("drop", "onehot", "numeric"):
                if key in plan and isinstance(plan[key], list):
                    plan[key] = [c for c in plan[key] if c != self.target_column]
            if "ordinal" in plan and isinstance(plan["ordinal"], dict):
                plan["ordinal"].pop(self.target_column, None)

            logging.info("LLM feature plan: %s", plan)
            return plan

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 2. Detect problem type ─────────────────────────────────────────────────
    def detect_problem_type(self) -> str:
        """
        Returns 'classification' or 'regression'.

        Rules (in order):
        - object / bool dtype              → classification
        - float dtype                      → regression  (continuous by nature)
        - int dtype, unique <= 20          → classification
        - int dtype, unique >  20          → regression
        - fallback                         → classification
        """
        try:
            target = self.df[self.target_column]
            dtype  = target.dtype
            n_unique = target.nunique()

            if dtype == object or dtype == bool:
                problem = "classification"
            elif np.issubdtype(dtype, np.floating):
                problem = "regression"
            elif np.issubdtype(dtype, np.integer):
                problem = "classification" if n_unique <= _CLASS_UNIQUE_MAX else "regression"
            else:
                problem = "classification"

            logging.info(
                "Problem type: %s (target dtype=%s, nunique=%d)",
                problem, dtype, n_unique,
            )
            return problem

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 3. Preprocess ──────────────────────────────────────────────────────────
    def preprocess(
        self, plan: dict[str, Any], problem_type: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Any]:
        """
        Apply feature plan, encode, scale, split.

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names, label_encoder_or_None
        """
        try:
            df = self.df.copy()

            # Drop columns LLM flagged
            drop_cols = [c for c in plan.get("drop", []) if c in df.columns]
            df.drop(columns=drop_cols, inplace=True)
            logging.info("Dropped columns: %s", drop_cols)

            X = df.drop(columns=[self.target_column])
            y = df[self.target_column].copy()

            # Encode target for classification
            le = None
            if problem_type == "classification" and y.dtype == object:
                le = LabelEncoder()
                y  = le.fit_transform(y)

            # Determine column roles from plan (only keep cols present in X)
            ordinal_cols = [c for c in plan.get("ordinal", {}) if c in X.columns]
            onehot_cols  = [c for c in plan.get("onehot",  []) if c in X.columns]
            numeric_cols = [c for c in plan.get("numeric", []) if c in X.columns]

            # Any remaining columns not in plan → infer
            planned = set(ordinal_cols + onehot_cols + numeric_cols)
            for col in X.columns:
                if col in planned:
                    continue
                if X[col].dtype == object:
                    # Only onehot encode if cardinality is VERY low (<=10 unique values)
                    # This prevents memory issues from high-cardinality features
                    if X[col].nunique() <= 10:
                        onehot_cols.append(col)
                    else:
                        # Drop high-cardinality columns to save memory
                        X.drop(columns=[col], inplace=True)
                else:
                    numeric_cols.append(col)

            # Only keep cols still in X
            ordinal_cols = [c for c in ordinal_cols if c in X.columns]
            onehot_cols  = [c for c in onehot_cols  if c in X.columns]
            numeric_cols = [c for c in numeric_cols  if c in X.columns]

            # MEMORY PROTECTION: Limit onehot columns to prevent feature explosion
            # Estimate total onehot features: sum of unique values for each column
            onehot_feature_count = sum(X[c].nunique() for c in onehot_cols)
            if onehot_feature_count > 50:
                logging.warning(
                    "⚠️ OneHot encoding would create %d features (memory issues risk)",
                    onehot_feature_count
                )
                # Sort by cardinality and keep only lowest-cardinality columns
                onehot_cols_with_card = [(c, X[c].nunique()) for c in onehot_cols]
                onehot_cols_with_card.sort(key=lambda x: x[1])  # Sort by cardinality
                
                # Keep cumulative features under 50
                kept_cols = []
                cumulative = 0
                for col, card in onehot_cols_with_card:
                    if cumulative + card <= 50:
                        kept_cols.append(col)
                        cumulative += card
                    else:
                        logging.warning(f"    Dropping {col} ({card} unique values) to save memory")
                        X.drop(columns=[col], inplace=True)
                
                onehot_cols = kept_cols
                logging.info(f"    Keeping {len(onehot_cols)} onehot columns with ~{cumulative} features")

            # Build ordinal categories list - handle "auto" properly for sklearn
            ordinal_categories = []
            has_explicit_cats = False
            for c in ordinal_cols:
                cats = plan["ordinal"].get(c, "auto")
                if isinstance(cats, list):
                    ordinal_categories.append(cats)
                    has_explicit_cats = True
                else:
                    # For "auto", append None so sklearn infers from data
                    ordinal_categories.append(None)
            
            # If no explicit categories provided, let sklearn infer all from data
            ordinal_categories = ordinal_categories if has_explicit_cats else "auto"

            # ColumnTransformer with memory-efficient encoders
            transformers = []
            if numeric_cols:
                transformers.append(("num", StandardScaler(), numeric_cols))
            if ordinal_cols:
                transformers.append((
                    "ord",
                    OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1,
                                   dtype=float),
                    ordinal_cols,
                ))
            if onehot_cols:
                from sklearn.preprocessing import OneHotEncoder
                # Use sparse_output=False for compatibility, but memory is protected above
                transformers.append((
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    onehot_cols,
                ))

            preprocessor = ColumnTransformer(transformers, remainder="drop")

            # Split first, then fit on train only
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=_TEST_SIZE,
                random_state=_RANDOM_STATE,
                stratify=y if problem_type == "classification" else None,
            )

            # Log preprocessing info before transform
            logging.info(
                "Preprocessing info: numeric=%d, ordinal=%d, onehot=%d",
                len(numeric_cols), len(ordinal_cols), len(onehot_cols)
            )

            try:
                X_train = preprocessor.fit_transform(X_train)
                X_test  = preprocessor.transform(X_test)
            except MemoryError as e:
                logging.error("Memory error during preprocessing: %s", e)
                logging.error("Input shape: %s, Data size: ~%.2f GiB", 
                            X_train.shape, (X_train.shape[0] * X_train.shape[1] * 8 / 1e9))
                raise Exception(f"Insufficient memory for preprocessing. "
                              f"Try reducing dataset size or disabling one-hot encoding. Error: {e}")

            # Recover feature names
            try:
                feature_names = preprocessor.get_feature_names_out().tolist()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

            logging.info(
                "Preprocessed: X_train=%s, X_test=%s, features=%d",
                X_train.shape, X_test.shape, len(feature_names),
            )
            return X_train, X_test, y_train, y_test, feature_names, le

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 4. Get models ──────────────────────────────────────────────────────────
    def _get_models(self, problem_type: str) -> dict[str, Any]:
        if problem_type == "classification":
            models = {
                "LogisticRegression":    LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE),
                "DecisionTree":          DecisionTreeClassifier(random_state=_RANDOM_STATE),
                "RandomForest":          RandomForestClassifier(n_estimators=100, random_state=_RANDOM_STATE),
                "GradientBoosting":      GradientBoostingClassifier(random_state=_RANDOM_STATE),
                "SVM":                   SVC(probability=True, random_state=_RANDOM_STATE),
                "KNN":                   KNeighborsClassifier(),
                "NaiveBayes":            GaussianNB(),
            }
            if _XGB:
                models["XGBoost"] = XGBClassifier(
                    random_state=_RANDOM_STATE, eval_metric="logloss",
                    use_label_encoder=False, verbosity=0,
                )
            if _LGBM:
                models["LightGBM"] = LGBMClassifier(random_state=_RANDOM_STATE, verbose=-1)

        else:  # regression
            models = {
                "LinearRegression":      LinearRegression(),
                "Ridge":                 Ridge(random_state=_RANDOM_STATE),
                "Lasso":                 Lasso(random_state=_RANDOM_STATE),
                "DecisionTree":          DecisionTreeRegressor(random_state=_RANDOM_STATE),
                "RandomForest":          RandomForestRegressor(n_estimators=100, random_state=_RANDOM_STATE),
                "GradientBoosting":      GradientBoostingRegressor(random_state=_RANDOM_STATE),
                "SVR":                   SVR(),
                "KNN":                   KNeighborsRegressor(),
            }
            if _XGB:
                models["XGBoost"] = XGBRegressor(random_state=_RANDOM_STATE, verbosity=0)
            if _LGBM:
                models["LightGBM"] = LGBMRegressor(random_state=_RANDOM_STATE, verbose=-1)

        return models

    # ── 5. Train + evaluate ────────────────────────────────────────────────────
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
        feature_names: list[str],
        problem_type: str,
        label_encoder: Any,
    ) -> dict[str, Any]:
        """
        Train every model, evaluate, return full results dict.
        """
        try:
            models  = self._get_models(problem_type)
            results = {}

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if problem_type == "classification":
                        metrics = {
                            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
                            "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                            "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                        }
                        # ROC-AUC (binary only)
                        try:
                            y_prob = model.predict_proba(X_test)
                            if y_prob.shape[1] == 2:
                                metrics["roc_auc"] = round(
                                    roc_auc_score(y_test, y_prob[:, 1]), 4
                                )
                        except Exception:
                            pass

                        cm = confusion_matrix(y_test, y_pred).tolist()
                        metrics["confusion_matrix"] = cm

                    else:  # regression
                        metrics = {
                            "r2":   round(r2_score(y_test, y_pred), 4),
                            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                            "mae":  round(mean_absolute_error(y_test, y_pred), 4),
                        }

                    # Feature importance
                    fi = None
                    if hasattr(model, "feature_importances_"):
                        fi = dict(zip(
                            feature_names,
                            [round(float(v), 6) for v in model.feature_importances_],
                        ))
                        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
                    elif hasattr(model, "coef_"):
                        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
                        fi = dict(zip(
                            feature_names,
                            [round(float(v), 6) for v in coef],
                        ))
                        fi = dict(sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:15])

                    results[name] = {
                        "metrics":             metrics,
                        "feature_importance":  fi,
                        "model_object":        model,
                    }
                    logging.info("Trained %s: %s", name, metrics)

                except Exception as exc:
                    logging.warning("Model %s failed: %s", name, exc)

            return results

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── 6. Save to MongoDB ─────────────────────────────────────────────────────
    def save_to_mongo(
        self,
        results: dict[str, Any],
        problem_type: str,
        feature_plan: dict,
        shap_data: dict | None = None,
    ) -> str:
        """
        Save each model's .pkl to GridFS.
        Save all metrics + best model info to ml_results collection.
        Returns the inserted document _id as string.
        """
        try:
            # Determine primary metric for ranking
            rank_metric = "accuracy" if problem_type == "classification" else "r2"

            metrics_only = {}
            model_gridfs_ids = {}

            for name, data in results.items():
                # Save model pkl to GridFS
                pkl_bytes = pickle.dumps(data["model_object"])
                gridfs_id = self.fs.put(
                    pkl_bytes,
                    filename=f"{self.filename}__{name}.pkl",
                    metadata={
                        "source_file": self.filename,
                        "model_name":  name,
                        "problem_type": problem_type,
                    },
                )
                model_gridfs_ids[name] = str(gridfs_id)
                metrics_only[name] = {
                    k: v for k, v in data["metrics"].items()
                    if k != "confusion_matrix"
                }
                metrics_only[name]["confusion_matrix"] = data["metrics"].get("confusion_matrix")
                metrics_only[name]["feature_importance"] = data.get("feature_importance")

            # Find best model
            best_name = max(
                metrics_only,
                key=lambda n: metrics_only[n].get(rank_metric, float("-inf")),
            )

            doc = {
                "filename":        self.filename,
                "target_column":   self.target_column,
                "problem_type":    problem_type,
                "feature_plan":    feature_plan,
                "metrics":         metrics_only,
                "model_gridfs_ids": model_gridfs_ids,
                "best_model":      best_name,
                "rank_metric":     rank_metric,
                "shap":            shap_data or {},
            }

            result = self.col.insert_one(doc)
            logging.info("Saved ML results to MongoDB. doc_id=%s", result.inserted_id)
            return str(result.inserted_id)

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Full pipeline ──────────────────────────────────────────────────────────
    def run(self, progress_callback=None) -> dict[str, Any]:
        """
        Execute the full pipeline end-to-end.

        Args:
          progress_callback: Optional callable(pct, msg) for progress tracking

        Returns
        -------
        {
          "filename":        str,
          "target_column":   str,
          "problem_type":    str,
          "feature_plan":    dict,
          "results":         {model_name: {metrics, feature_importance}},
          "best_model":      str,
          "rank_metric":     str,
          "mongo_doc_id":    str,
          "shap":            dict with SHAP data,
        }
        """
        def progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)
            logging.info("[%d%%] %s", pct, msg)

        try:
            logging.info("=== ML Pipeline starting for '%s' ===", self.filename)

            # Step 1 — LLM feature plan
            progress(5, "Asking LLM for feature engineering plan...")
            feature_plan = self.llm_feature_plan()

            # Step 2 — detect problem type
            progress(15, "Detecting problem type...")
            problem_type = self.detect_problem_type()

            # Step 3 — preprocess
            progress(25, "Preprocessing data...")
            X_train, X_test, y_train, y_test, feature_names, le = self.preprocess(
                feature_plan, problem_type
            )

            # Step 4 + 5 — train + evaluate
            models = self._get_models(problem_type)
            n_models = len(models)
            results = {}

            for i, (name, model) in enumerate(models.items()):
                pct = 30 + int((i / n_models) * 50)
                progress(pct, f"Training {name}... ({i+1}/{n_models} models complete)")
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if problem_type == "classification":
                        metrics = {
                            "accuracy": round(accuracy_score(y_test, y_pred), 4),
                            "f1": round(
                                f1_score(
                                    y_test, y_pred, average="weighted", zero_division=0
                                ),
                                4,
                            ),
                            "precision": round(
                                precision_score(
                                    y_test, y_pred, average="weighted", zero_division=0
                                ),
                                4,
                            ),
                            "recall": round(
                                recall_score(
                                    y_test, y_pred, average="weighted", zero_division=0
                                ),
                                4,
                            ),
                        }
                        # ROC-AUC (binary only)
                        try:
                            y_prob = model.predict_proba(X_test)
                            if y_prob.shape[1] == 2:
                                metrics["roc_auc"] = round(
                                    roc_auc_score(y_test, y_prob[:, 1]), 4
                                )
                        except Exception:
                            pass

                        cm = confusion_matrix(y_test, y_pred).tolist()
                        metrics["confusion_matrix"] = cm

                    else:  # regression
                        metrics = {
                            "r2": round(r2_score(y_test, y_pred), 4),
                            "rmse": round(
                                np.sqrt(mean_squared_error(y_test, y_pred)), 4
                            ),
                            "mae": round(mean_absolute_error(y_test, y_pred), 4),
                        }

                    # Feature importance
                    fi = None
                    if hasattr(model, "feature_importances_"):
                        fi = dict(
                            zip(
                                feature_names,
                                [round(float(v), 6) for v in model.feature_importances_],
                            )
                        )
                        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
                    elif hasattr(model, "coef_"):
                        coef = (
                            model.coef_.flatten()
                            if model.coef_.ndim > 1
                            else model.coef_
                        )
                        fi = dict(
                            zip(
                                feature_names,
                                [round(float(v), 6) for v in coef],
                            )
                        )
                        fi = dict(sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[: 15])

                    results[name] = {
                        "metrics": metrics,
                        "feature_importance": fi,
                        "model_object": model,
                    }
                    logging.info("Trained %s: %s", name, metrics)

                except Exception as exc:
                    logging.warning("Model %s failed: %s", name, exc)

            # Determine best model
            rank_metric = "accuracy" if problem_type == "classification" else "r2"
            best_model = max(
                results,
                key=lambda n: results[n]["metrics"].get(rank_metric, float("-inf")),
            )

            # Step 7 — SHAP for best model only
            progress(82, f"Running SHAP for {best_model}...")
            shap_data = {}
            try:
                from src.components.shap_explainer import SHAPExplainer

                best_obj = results[best_model]["model_object"]
                explainer = SHAPExplainer(
                    model=best_obj,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=feature_names,
                    problem_type=problem_type,
                )
                shap_data = explainer.run_all()
                shap_data["model_name"] = best_model
                logging.info("SHAP completed for %s", best_model)
            except Exception as shap_err:
                logging.warning("SHAP skipped: %s", shap_err)

            # Step 8 — save to MongoDB
            progress(92, "Saving to MongoDB...")
            mongo_id = self.save_to_mongo(results, problem_type, feature_plan, shap_data)

            progress(100, "Pipeline complete!")

            # Strip model objects before returning (not JSON-serialisable)
            clean_results = {
                name: {
                    "metrics": data["metrics"],
                    "feature_importance": data.get("feature_importance"),
                }
                for name, data in results.items()
            }

            # Sort by primary metric descending
            clean_results = dict(
                sorted(
                    clean_results.items(),
                    key=lambda x: x[1]["metrics"].get(rank_metric, float("-inf")),
                    reverse=True,
                )
            )

            logging.info("=== ML Pipeline complete. Best: %s ===", best_model)

            return {
                "filename": self.filename,
                "target_column": self.target_column,
                "problem_type": problem_type,
                "feature_plan": feature_plan,
                "results": clean_results,
                "best_model": best_model,
                "rank_metric": rank_metric,
                "mongo_doc_id": mongo_id,
                "shap": shap_data,
            }

        except Exception as e:
            raise CustomException(e, sys) from e