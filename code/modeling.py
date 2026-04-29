"""
Model training with automatic algorithm selection for clinical endpoints.

Supports: Random Forest, XGBoost, GLM (Linear/Logistic/Cox).
Automatically triggers hyperparameter tuning for small samples (N < 200).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

from shared.config import (
    SEED,
    SMALL_SAMPLE_THRESHOLD,
    CV_FOLDS,
    EndpointType,
)

logger = logging.getLogger(__name__)

# Default hyperparameter grids for small-sample tuning
_RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
}

_XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}


class ModelTrainer:
    """Train and evaluate models for clinical outcome prediction.

    Automatically selects the appropriate model type based on endpoint
    and triggers GridSearchCV when the training set is small (N < threshold).
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.model_: Any = None
        self.model_type_: str = ""
        self.is_tuned_: bool = False
        self.cv_scores_: Optional[list[float]] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "auto",
        endpoint: EndpointType = EndpointType.CONTINUOUS,
    ) -> Any:
        """Train the specified (or auto-selected) model.

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        model_type : str
            One of 'rf', 'xgb', 'glm', or 'auto'.
        endpoint : EndpointType

        Returns
        -------
        Trained model object.
        """
        n_samples = X_train.shape[0]
        if model_type == "auto":
            model_type = self._auto_select(n_samples, endpoint)

        self.model_type_ = model_type
        logger.info(
            f"Training {model_type} | N={n_samples} | endpoint={endpoint.value}"
        )

        if model_type == "rf":
            self.model_ = self._train_rf(X_train, y_train, n_samples)
        elif model_type == "xgb":
            self.model_ = self._train_xgb(X_train, y_train, n_samples)
        elif model_type == "glm":
            self.model_ = self._train_glm(X_train, y_train)
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Use rf/xgb/glm/auto.")

        # Cross-validation estimate for small samples
        if n_samples < SMALL_SAMPLE_THRESHOLD:
            self.cv_scores_ = cross_val_score(
                self.model_, X_train, y_train, cv=CV_FOLDS,
                scoring="r2", n_jobs=-1,
            )
            logger.info(f"CV R² (k={CV_FOLDS}): {self.cv_scores_.mean():.4f} ± {self.cv_scores_.std():.4f}")

        return self.model_

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate the trained model on test data.

        Returns
        -------
        dict with keys: r2, rmse, mae
        """
        if self.model_ is None:
            raise RuntimeError("No trained model. Call .train() first.")
        y_pred = self.model_.predict(X_test)
        return {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_select(n_samples: int, endpoint: EndpointType) -> str:
        if endpoint == EndpointType.CONTINUOUS:
            return "xgb" if n_samples >= SMALL_SAMPLE_THRESHOLD else "rf"
        return "xgb"

    def _should_tune(self, n_samples: int) -> bool:
        return n_samples < SMALL_SAMPLE_THRESHOLD

    def _train_rf(self, X: np.ndarray, y: np.ndarray, n: int) -> RandomForestRegressor:
        if self._should_tune(n):
            gs = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                _RF_PARAM_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"RF best params: {gs.best_params_}")
            return gs.best_estimator_
        model = RandomForestRegressor(
            n_estimators=200, random_state=self.random_state, n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_xgb(self, X: np.ndarray, y: np.ndarray, n: int) -> xgb.XGBRegressor:
        if self._should_tune(n):
            gs = GridSearchCV(
                xgb.XGBRegressor(random_state=self.random_state, verbosity=0),
                _XGB_PARAM_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"XGB best params: {gs.best_params_}")
            return gs.best_estimator_
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbosity=0,
        )
        model.fit(X, y)
        return model

    def _train_glm(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        model = LinearRegression()
        model.fit(X, y)
        return model
