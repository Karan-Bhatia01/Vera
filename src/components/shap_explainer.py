"""
SHAP model explainability module.

Provides SHAP-based explanations for trained models:
  - Feature importance via mean |SHAP values|
  - Force plots for individual predictions
  - Rendered as base64-encoded PNG for web display
"""

import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from src.logger import logging

matplotlib.use("Agg")


class SHAPExplainer:
    """
    SHAP explainability for sklearn models.
    
    Automatically selects TreeExplainer for tree-based models,
    KernelExplainer for others.
    """

    TREE_MODELS = {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
    }

    def __init__(self, model, X_train, X_test, feature_names, problem_type):
        """
        Args:
          model: Fitted sklearn model
          X_train: Training data (np.ndarray)
          X_test: Test data (np.ndarray)
          feature_names: List of feature names
          problem_type: "classification" or "regression"
        """
        self.model = model
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.feature_names = list(feature_names)
        self.problem_type = problem_type
        self._explainer = None
        self._shap_values = None

    def _get_explainer(self):
        """Get or create SHAP explainer."""
        if self._explainer is not None:
            return self._explainer

        model_type = type(self.model).__name__

        if model_type in self.TREE_MODELS:
            self._explainer = shap.TreeExplainer(self.model)
        else:
            # KernelExplainer — use small background sample for speed
            n_bg = min(50, len(self.X_train))
            background = shap.sample(self.X_train, n_bg)
            predict_fn = (
                self.model.predict_proba
                if self.problem_type == "classification"
                and hasattr(self.model, "predict_proba")
                else self.model.predict
            )
            self._explainer = shap.KernelExplainer(predict_fn, background)

        return self._explainer

    def _get_shap_values(self):
        """Compute SHAP values for test set."""
        if self._shap_values is not None:
            return self._shap_values

        explainer = self._get_explainer()
        # Use at most 200 test rows for speed
        X_sample = self.X_test[:200]
        shap_values = explainer.shap_values(X_sample)

        # For multi-class output (list of arrays), use class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        self._shap_values = shap_values
        return shap_values

    def _fig_to_b64(self, fig) -> str:
        """Convert matplotlib figure to base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="#111118",
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def summary_plot_b64(self) -> str:
        """Bar chart of mean absolute SHAP values per feature."""
        shap_values = self._get_shap_values()

        fig, ax = plt.subplots(
            figsize=(10, max(4, len(self.feature_names) * 0.35))
        )
        ax.set_facecolor("#111118")
        fig.patch.set_facecolor("#111118")

        shap.summary_plot(
            shap_values,
            self.X_test[:200],
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            color="#4f46e5",
        )
        plt.tick_params(colors="#a0a0b8")
        plt.xlabel("Mean |SHAP value|", color="#a0a0b8")
        plt.title(f"Feature Impact (SHAP)", color="#f0f0fa", pad=12)

        return self._fig_to_b64(fig)

    def force_plot_b64(self, row_index: int = 0) -> str:
        """Force plot for a single prediction."""
        explainer = self._get_explainer()
        shap_values = self._get_shap_values()

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        fig = plt.figure(figsize=(16, 3))
        fig.patch.set_facecolor("#111118")

        shap.force_plot(
            float(expected_value),
            shap_values[row_index],
            self.X_test[row_index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
            text_rotation=15,
        )
        return self._fig_to_b64(fig)

    def top_features(self, n: int = 10) -> list[dict]:
        """Top N features ranked by mean absolute SHAP value."""
        shap_values = self._get_shap_values()
        mean_abs = np.abs(shap_values).mean(axis=0)
        pairs = sorted(
            zip(self.feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": f, "importance": round(v, 6)} for f, v in pairs[:n]]

    def run_all(self) -> dict:
        """
        Run all SHAP analyses and return a dict ready for MongoDB storage
        and Jinja2 template rendering.
        Returns: {summary_plot, force_plot, top_features, model_name}
        Gracefully returns {} on any failure.
        """
        try:
            return {
                "summary_plot": self.summary_plot_b64(),
                "force_plot": self.force_plot_b64(row_index=0),
                "top_features": self.top_features(n=10),
            }
        except Exception as e:
            logging.warning("SHAP run_all failed: %s", e)
            return {}
