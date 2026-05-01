"""
Model training with automatic algorithm selection for clinical endpoints.

Supports: Random Forest, XGBoost, GLM / Logistic Regression.
Automatically triggers hyperparameter tuning for small samples (N < 200).

Endpoint routing:
- continuous → regressors (R² scoring)
- binary     → classifiers (ROC AUC scoring)
- survival   → Cox PH / RSF
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, accuracy_score,
)
import xgboost as xgb
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from shared.config import (
    SEED,
    SMALL_SAMPLE_THRESHOLD,
    CV_FOLDS,
    EndpointType,
)

logger = logging.getLogger(__name__)

# Hyperparameter grids for small-sample tuning
_RF_REG_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
}

_XGB_REG_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

_RF_CLF_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced", "balanced_subsample", None],
}

_XGB_CLF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
}

_LOGREG_GRID = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "class_weight": ["balanced", None],
}

_POISSON_GRID = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
}

_RSF_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
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
        self.cv_scores_: Optional[np.ndarray] = None
        self._is_classifier_: bool = False
        self._is_survival_: bool = False
        self._is_count_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        self._is_classifier_ = endpoint == EndpointType.BINARY
        self._is_survival_ = endpoint == EndpointType.SURVIVAL
        self._is_count_ = endpoint == EndpointType.COUNT
        logger.info(
            f"Training {model_type} | N={n_samples} | endpoint={endpoint.value}"
        )

        if self._is_survival_:
            self.model_ = self._train_survival(X_train, y_train, model_type, n_samples)
        elif self._is_classifier_:
            self.model_ = self._train_classifier(X_train, y_train, model_type, n_samples)
        elif self._is_count_:
            self.model_ = self._train_count(X_train, y_train, model_type, n_samples)
        else:
            self.model_ = self._train_regressor(X_train, y_train, model_type, n_samples)

        # Cross-validation for small samples (skip for survival — expensive)
        if n_samples < SMALL_SAMPLE_THRESHOLD and not self._is_survival_:
            scoring = "roc_auc" if self._is_classifier_ else "r2"
            self.cv_scores_ = cross_val_score(
                self.model_, X_train, y_train, cv=CV_FOLDS,
                scoring=scoring, n_jobs=-1,
            )
            metric_name = "ROC AUC" if self._is_classifier_ else "R²"
            logger.info(
                f"CV {metric_name} (k={CV_FOLDS}): "
                f"{self.cv_scores_.mean():.4f} ± {self.cv_scores_.std():.4f}"
            )

        return self.model_

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate the trained model on test data.

        Returns
        -------
        dict:
            For continuous: r2, rmse, mae
            For binary: roc_auc, accuracy
        """
        if self.model_ is None:
            raise RuntimeError("No trained model. Call .train() first.")

        if self._is_survival_:
            risk_scores = self.model_.predict(X_test)
            c_index = concordance_index_censored(
                y_test["event"].astype(bool), y_test["time"], risk_scores
            )
            return {
                "c_index": float(c_index[0]),
            }
        elif self._is_count_:
            y_pred = np.clip(self.model_.predict(X_test), 0, None)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            # D² pseudo-R² (deviance-based)
            y_mean = np.mean(y_test)
            deviance_null = 2 * np.sum(y_test * np.log((y_test + 1e-10) / (y_mean + 1e-10)) - (y_test - y_mean))
            deviance_model = 2 * np.sum(y_test * np.log((y_test + 1e-10) / (y_pred + 1e-10)) - (y_test - y_pred))
            d2 = 1.0 - deviance_model / deviance_null if deviance_null != 0 else 0.0
            return {
                "rmse": rmse,
                "mae": mae,
                "d2_pseudo_r2": float(max(0.0, d2)),
            }
        elif self._is_classifier_:
            y_prob = self.model_.predict_proba(X_test)[:, 1]
            y_pred = self.model_.predict(X_test)
            return {
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
            }
        else:
            y_pred = self.model_.predict(X_test)
            return {
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
            }

    # ------------------------------------------------------------------
    # Internal — routing
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_select(n_samples: int, endpoint: EndpointType) -> str:
        if endpoint == EndpointType.SURVIVAL:
            return "rsf" if n_samples >= SMALL_SAMPLE_THRESHOLD else "cox"
        if endpoint == EndpointType.BINARY:
            return "xgb" if n_samples >= SMALL_SAMPLE_THRESHOLD else "rf"
        if endpoint == EndpointType.CONTINUOUS:
            return "xgb" if n_samples >= SMALL_SAMPLE_THRESHOLD else "rf"
        if endpoint == EndpointType.COUNT:
            return "xgb" if n_samples >= SMALL_SAMPLE_THRESHOLD else "glm"
        return "xgb"

    def _should_tune(self, n_samples: int) -> bool:
        return n_samples < SMALL_SAMPLE_THRESHOLD

    # ------------------------------------------------------------------
    # Regressors (continuous)
    # ------------------------------------------------------------------
    def _train_regressor(self, X: np.ndarray, y: np.ndarray, model_type: str, n: int) -> Any:
        if model_type == "rf":
            return self._train_rf_reg(X, y, n)
        elif model_type == "xgb":
            return self._train_xgb_reg(X, y, n)
        elif model_type == "glm":
            return self._train_glm_reg(X, y)
        raise ValueError(f"Unknown regressor: '{model_type}'")

    def _train_rf_reg(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                _RF_REG_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"RF regressor best params: {gs.best_params_}")
            return gs.best_estimator_
        model = RandomForestRegressor(
            n_estimators=200, random_state=self.random_state, n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_xgb_reg(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                xgb.XGBRegressor(random_state=self.random_state, verbosity=0),
                _XGB_REG_GRID, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"XGB regressor best params: {gs.best_params_}")
            return gs.best_estimator_
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbosity=0,
        )
        model.fit(X, y)
        return model

    def _train_glm_reg(self, X: np.ndarray, y: np.ndarray) -> Any:
        model = LinearRegression()
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Count models (Poisson / NB)
    # ------------------------------------------------------------------
    def _train_count(self, X: np.ndarray, y: np.ndarray, model_type: str, n: int) -> Any:
        if model_type == "rf":
            return self._train_count_rf(X, y, n)
        elif model_type == "xgb":
            return self._train_count_xgb(X, y, n)
        elif model_type == "glm":
            return self._train_poisson(X, y, n)
        raise ValueError(f"Unknown count model: '{model_type}'")

    def _train_count_rf(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                _RF_REG_GRID, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"RF count best params: {gs.best_params_}")
            return gs.best_estimator_
        model = RandomForestRegressor(
            n_estimators=200, random_state=self.random_state, n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_count_xgb(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                xgb.XGBRegressor(
                    objective="count:poisson", random_state=self.random_state,
                    verbosity=0,
                ),
                _XGB_REG_GRID, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"XGB count best params: {gs.best_params_}")
            return gs.best_estimator_
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            objective="count:poisson", random_state=self.random_state,
            verbosity=0,
        )
        model.fit(X, y)
        return model

    def _train_poisson(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                PoissonRegressor(max_iter=500),
                _POISSON_GRID, cv=CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"Poisson best params: {gs.best_params_}")
            return gs.best_estimator_
        model = PoissonRegressor(alpha=0.001, max_iter=500)
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Classifiers (binary)
    # ------------------------------------------------------------------
    def _train_classifier(self, X: np.ndarray, y: np.ndarray, model_type: str, n: int) -> Any:
        if model_type == "rf":
            return self._train_rf_clf(X, y, n)
        elif model_type == "xgb":
            return self._train_xgb_clf(X, y, n)
        elif model_type == "glm":
            return self._train_logreg(X, y, n)
        raise ValueError(f"Unknown classifier: '{model_type}'")

    def _train_rf_clf(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                _RF_CLF_GRID, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"RF classifier best params: {gs.best_params_}")
            return gs.best_estimator_
        model = RandomForestClassifier(
            n_estimators=200, random_state=self.random_state,
            class_weight="balanced", n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_xgb_clf(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                xgb.XGBClassifier(random_state=self.random_state, verbosity=0),
                _XGB_CLF_GRID, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"XGB classifier best params: {gs.best_params_}")
            return gs.best_estimator_
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbosity=0,
        )
        model.fit(X, y)
        return model

    def _train_logreg(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                LogisticRegression(
                    random_state=self.random_state, max_iter=2000,
                    solver="lbfgs",
                ),
                _LOGREG_GRID, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"LogReg best params: {gs.best_params_}")
            return gs.best_estimator_
        model = LogisticRegression(
            random_state=self.random_state, max_iter=2000, solver="lbfgs",
        )
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Survival models
    # ------------------------------------------------------------------
    def _train_survival(self, X: np.ndarray, y: np.ndarray, model_type: str, n: int) -> Any:
        if model_type == "cox":
            return self._train_cox(X, y, n)
        elif model_type == "rsf":
            return self._train_rsf(X, y, n)
        # Default for survival: Cox PH
        return self._train_cox(X, y, n)

    def _train_cox(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        model = CoxPHSurvivalAnalysis(alpha=1.0)
        model.fit(X, y)
        return model

    def _train_rsf(self, X: np.ndarray, y: np.ndarray, n: int) -> Any:
        if self._should_tune(n):
            gs = GridSearchCV(
                RandomSurvivalForest(random_state=self.random_state, n_jobs=-1),
                _RSF_GRID, cv=CV_FOLDS, n_jobs=-1,
            )
            gs.fit(X, y)
            self.is_tuned_ = True
            logger.info(f"RSF best params: {gs.best_params_}")
            return gs.best_estimator_
        model = RandomSurvivalForest(
            n_estimators=200, random_state=self.random_state, n_jobs=-1,
        )
        model.fit(X, y)
        return model
