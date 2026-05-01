"""
SHAP value computation and feature importance analysis.

Automatically selects the appropriate SHAP explainer based on model type
(TreeExplainer for tree-based models, LinearExplainer for linear models,
SurvSHAP for survival models).
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

    For survival models::

        shap_values = analyzer.compute_survival(
            model, X_train_df, X_test_df, y_train, feature_names,
            model_type="rsf",
        )
        # analyzer.shap_values_ is 2D (time-aggregated)
        # analyzer.survshap_values_ is 3D (n_samples, n_features, n_times)
        # analyzer.survshap_times_ has the time grid
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.shap_values_: Optional[np.ndarray] = None
        self.explainer_: Any = None
        self.feature_importance_: Optional[pd.DataFrame] = None
        # Survival-specific
        self.survshap_values_: Optional[np.ndarray] = None  # (n_samples, n_features, n_times)
        self.survshap_times_: Optional[np.ndarray] = None
        self._model_survshap_: Any = None

    # ------------------------------------------------------------------
    # Standard SHAP (continuous / binary)
    # ------------------------------------------------------------------
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
        bg = shap.sample(X_train, min(max_samples, X_train.shape[0]),
                         random_state=self.random_state).astype(np.float64)
        X_test_f = X_test.astype(np.float64)

        if model_type in ("rf", "xgb"):
            explainer = shap.TreeExplainer(model, bg, feature_perturbation="interventional")
        elif model_type == "glm":
            explainer = shap.LinearExplainer(model, bg)
        else:
            explainer = shap.KernelExplainer(
                model.predict, bg, random_state=self.random_state
            )
        self.explainer_ = explainer

        logger.info(
            f"Computing SHAP: explainer={type(explainer).__name__}, "
            f"n_test={X_test_f.shape[0]}, n_features={X_test_f.shape[1]}"
        )
        if isinstance(explainer, shap.TreeExplainer):
            self.shap_values_ = explainer.shap_values(X_test_f, check_additivity=False)
        else:
            self.shap_values_ = explainer.shap_values(X_test_f)
        if isinstance(self.shap_values_, list):
            # Multi-class: take positive class
            self.shap_values_ = self.shap_values_[1]
        elif self.shap_values_.ndim == 3:
            # Binary classification: (n_samples, n_features, 2) → positive class
            self.shap_values_ = self.shap_values_[:, :, 1]
        logger.info(f"SHAP values shape: {self.shap_values_.shape}")
        return self.shap_values_

    # ------------------------------------------------------------------
    # SurvSHAP (survival)
    # ------------------------------------------------------------------
    def compute_survival(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: list[str],
        model_type: str = "rsf",
        calculation_method: str = "sampling",
        function_type: str = "sf",
        max_samples: int = 50,
    ) -> np.ndarray:
        """Compute SurvSHAP(t) time-dependent SHAP for survival models.

        Parameters
        ----------
        model : trained survival model (RSF, CoxPH, etc.)
        X_train : pd.DataFrame
            Training features (used as background).
        X_test : pd.DataFrame
            Test features to explain.
        y_train : np.ndarray
            Structured array with 'event' and 'time' fields.
        feature_names : list[str]
        model_type : str
            'rsf' → TreeSHAP; 'cox'/other → KernelSHAP.
        calculation_method : str
            'kernel', 'sampling', 'shap_kernel', or 'treeshap'.
        function_type : str
            'sf' (survival function) or 'chf' (cumulative hazard).
        max_samples : int
            Max background / explained samples.

        Returns
        -------
        np.ndarray
            Time-aggregated SHAP values (n_test, n_features) for
            compatibility with standard plots.
        Also populates:
            self.survshap_values_  — (n_samples, n_features, n_times)
            self.survshap_times_   — (n_times,)
        """
        from survshap import SurvivalModelExplainer, ModelSurvSHAP

        # Subsample to control runtime
        n_test = min(X_test.shape[0], max_samples)
        X_test_sub = X_test.iloc[:n_test]
        n_train_bg = min(X_train.shape[0], 80)
        X_train_bg = X_train.iloc[:n_train_bg]

        # Use sampling for all survival models (treeshap has additivity
        # issues with scikit-survival ensembles)
        calc_method = calculation_method

        logger.info(
            f"Computing SurvSHAP: model={model_type}, method={calc_method}, "
            f"n_bg={n_train_bg}, n_explain={n_test}"
        )

        explainer = SurvivalModelExplainer(
            model=model,
            data=X_train_bg,
            y=y_train[:n_train_bg],
        )
        self.explainer_ = explainer

        survshap = ModelSurvSHAP(
            function_type=function_type,
            calculation_method=calc_method,
            aggregation_method="integral",
            random_state=self.random_state,
        )
        survshap.fit(explainer, new_observations=X_test_sub)
        self._model_survshap_ = survshap
        self.survshap_times_ = survshap.timestamps

        # Extract 3D array from full_result DataFrame
        sv_3d = self._extract_survshap_3d(
            survshap.full_result, n_test, feature_names
        )
        self.survshap_values_ = sv_3d

        # Aggregate over time → 2D for standard plots
        self.shap_values_ = np.abs(sv_3d).mean(axis=2)

        logger.info(
            f"SurvSHAP shape: {sv_3d.shape} (samples × features × times)"
        )
        return self.shap_values_

    @staticmethod
    def _extract_survshap_3d(
        full_result: pd.DataFrame,
        n_samples: int,
        feature_names: list[str],
    ) -> np.ndarray:
        """Extract (n_samples, n_features, n_times) from ModelSurvSHAP result.

        The full_result DataFrame has columns:
        variable_str, variable_name, variable_value, B,
        aggregated_change, index, [t = <val>, ...]

        We filter to B==0 (average path) and reshape.
        """
        # Filter to average path only
        df = full_result[full_result["B"] == 0]
        # Identify timestamp columns (start with "t = ")
        meta_cols = {"variable_str", "variable_name", "variable_value",
                      "B", "aggregated_change", "index"}
        time_cols = [c for c in df.columns if c not in meta_cols]
        n_times = len(time_cols)

        sv_3d = np.zeros((n_samples, len(feature_names), n_times))
        for i in range(n_samples):
            sample_rows = df[df["index"] == i]
            for j, name in enumerate(feature_names):
                row = sample_rows[sample_rows["variable_name"] == name]
                if len(row) == 1:
                    sv_3d[i, j, :] = row[time_cols].values.flatten()

        return sv_3d

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
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
