"""
SHAP value computation and feature importance analysis.

Automatically selects the appropriate SHAP explainer based on model type
(TreeExplainer for tree-based models, LinearExplainer for linear models).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from shared.config import SEED

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Compute SHAP values and derive feature importance rankings.

    Usage::

        analyzer = SHAPAnalyzer()
        shap_values = analyzer.compute(model, X_train, X_test, model_type="xgb")
        importance = analyzer.get_feature_importance()
        df = analyzer.get_shap_dataframe(feature_names)
        analyzer.save_shap_values(df, "output/shap_values.csv")
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.shap_values_: Optional[np.ndarray] = None
        self.explainer_: Any = None
        self.feature_importance_: Optional[pd.DataFrame] = None

    def compute(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        model_type: str = "xgb",
        max_samples: int = 200,
    ) -> np.ndarray:
        """Compute SHAP values for test-set predictions.

        Parameters
        ----------
        model : trained model
        X_train : np.ndarray
            Training features (used as background for TreeExplainer).
        X_test : np.ndarray
            Test features to explain.
        model_type : str
            'rf', 'xgb' → TreeExplainer; 'glm' → LinearExplainer.
        max_samples : int
            Max background samples for KernelExplainer / TreeExplainer.

        Returns
        -------
        np.ndarray
            SHAP values with shape (n_test_samples, n_features).
        """
        # Subsample background for efficiency
        bg = shap.sample(X_train, min(max_samples, X_train.shape[0]),
                         random_state=self.random_state).astype(np.float64)
        X_test_f = X_test.astype(np.float64)

        if model_type in ("rf", "xgb"):
            explainer = shap.TreeExplainer(model, bg)
        elif model_type == "glm":
            explainer = shap.LinearExplainer(model, bg)
        else:
            # Fallback: KernelExplainer
            explainer = shap.KernelExplainer(
                model.predict, bg, random_state=self.random_state
            )
        self.explainer_ = explainer

        logger.info(
            f"Computing SHAP: explainer={type(explainer).__name__}, "
            f"n_test={X_test_f.shape[0]}, n_features={X_test_f.shape[1]}"
        )
        self.shap_values_ = explainer.shap_values(X_test_f)
        # Ensure 2D
        if isinstance(self.shap_values_, list):
            self.shap_values_ = self.shap_values_[0]
        logger.info(f"SHAP values shape: {self.shap_values_.shape}")
        return self.shap_values_

    def get_feature_importance(
        self,
        feature_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return feature importance ranked by mean(|SHAP|).

        Parameters
        ----------
        feature_names : list[str], optional

        Returns
        -------
        pd.DataFrame with columns: feature, mean_abs_shap, rank
        """
        if self.shap_values_ is None:
            raise RuntimeError("No SHAP values computed. Call .compute() first.")

        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(len(mean_abs))]
        order = np.argsort(mean_abs)[::-1]

        self.feature_importance_ = pd.DataFrame({
            "feature": [feature_names[i] for i in order],
            "mean_abs_shap": mean_abs[order].round(6),
            "rank": range(1, len(order) + 1),
        }).reset_index(drop=True)
        return self.feature_importance_

    def get_shap_dataframe(
        self,
        feature_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return SHAP values as a wide-format DataFrame.

        Rows = test samples, columns = features.
        """
        if self.shap_values_ is None:
            raise RuntimeError("No SHAP values computed. Call .compute() first.")
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(self.shap_values_.shape[1])]
        return pd.DataFrame(self.shap_values_, columns=feature_names)

    def save_shap_values(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Save SHAP value DataFrame to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"SHAP values saved → {path}")
