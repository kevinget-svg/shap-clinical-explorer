"""
Tests for code.modeling — model training and evaluation for all endpoints.
"""

import numpy as np
import pandas as pd
import pytest

from core.modeling import ModelTrainer
from core.preprocessing import Preprocessor, split_train_test
from core.synthetic_data import (
    get_demo_data,
    get_demo_data_binary,
    get_demo_data_survival,
    get_demo_data_count,
)
from shared.config import EndpointType, SMALL_SAMPLE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prep_continuous():
    df = get_demo_data(n_subjects=120)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")
    return Xtr, Xte, ytr, yte


def _prep_binary():
    df = get_demo_data_binary(n_subjects=150)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values.astype(int)
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")
    return Xtr, Xte, ytr, yte


def _prep_survival():
    from sksurv.util import Surv
    df = get_demo_data_survival(n_subjects=150)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    cnsr = df["CNSR"].values.astype(bool)
    time = df["AVAL"].values.astype(float)
    y = Surv.from_arrays(event=~cnsr, time=time)
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")
    return Xtr, Xte, ytr, yte


def _prep_count():
    df = get_demo_data_count(n_subjects=120)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values.astype(float)
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")
    return Xtr, Xte, ytr, yte


# ---------------------------------------------------------------------------
# Train tests — by endpoint
# ---------------------------------------------------------------------------
class TestTrainContinuous:
    def test_train_auto(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="auto", endpoint=EndpointType.CONTINUOUS)
        assert model is not None
        assert trainer.model_type_ in ("xgb", "rf", "glm")

    def test_train_rf(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.CONTINUOUS)
        assert trainer.model_type_ == "rf"

    def test_train_xgb(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="xgb", endpoint=EndpointType.CONTINUOUS)
        assert trainer.model_type_ == "xgb"

    def test_train_glm(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="glm", endpoint=EndpointType.CONTINUOUS)
        assert trainer.model_type_ == "glm"

    def test_invalid_model_type(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        with pytest.raises(ValueError):
            trainer.train(Xtr, ytr, model_type="svm", endpoint=EndpointType.CONTINUOUS)


class TestTrainBinary:
    def test_train_auto(self):
        Xtr, Xte, ytr, yte = _prep_binary()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="auto", endpoint=EndpointType.BINARY)
        assert model is not None
        assert trainer.model_type_ in ("xgb", "rf")

    def test_train_glm(self):
        Xtr, Xte, ytr, yte = _prep_binary()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="glm", endpoint=EndpointType.BINARY)
        assert trainer.model_type_ == "glm"


class TestTrainSurvival:
    def test_train_auto(self):
        Xtr, Xte, ytr, yte = _prep_survival()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="auto", endpoint=EndpointType.SURVIVAL)
        assert model is not None
        assert trainer.model_type_ in ("rsf", "cox")

    def test_train_cox(self):
        Xtr, Xte, ytr, yte = _prep_survival()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="cox", endpoint=EndpointType.SURVIVAL)
        assert trainer.model_type_ == "cox"


class TestTrainCount:
    def test_train_auto(self):
        Xtr, Xte, ytr, yte = _prep_count()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="auto", endpoint=EndpointType.COUNT)
        assert model is not None
        assert trainer.model_type_ in ("xgb", "glm")

    def test_train_glm(self):
        Xtr, Xte, ytr, yte = _prep_count()
        trainer = ModelTrainer()
        model = trainer.train(Xtr, ytr, model_type="glm", endpoint=EndpointType.COUNT)
        assert trainer.model_type_ == "glm"


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_continuous_metrics(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.CONTINUOUS)
        metrics = trainer.evaluate(Xte, yte)
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["rmse"] >= 0

    def test_binary_metrics(self):
        Xtr, Xte, ytr, yte = _prep_binary()
        trainer = ModelTrainer()
        trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.BINARY)
        metrics = trainer.evaluate(Xte, yte)
        assert "roc_auc" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_survival_metrics(self):
        Xtr, Xte, ytr, yte = _prep_survival()
        trainer = ModelTrainer()
        trainer.train(Xtr, ytr, model_type="cox", endpoint=EndpointType.SURVIVAL)
        metrics = trainer.evaluate(Xte, yte)
        assert "c_index" in metrics
        assert 0 <= metrics["c_index"] <= 1

    def test_count_metrics(self):
        Xtr, Xte, ytr, yte = _prep_count()
        trainer = ModelTrainer()
        trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.COUNT)
        metrics = trainer.evaluate(Xte, yte)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "d2_pseudo_r2" in metrics
        assert metrics["rmse"] >= 0

    def test_errors_without_train(self):
        Xtr, Xte, ytr, yte = _prep_continuous()
        trainer = ModelTrainer()
        with pytest.raises(RuntimeError, match="No trained model"):
            trainer.evaluate(Xte, yte)


# ---------------------------------------------------------------------------
# Auto-select
# ---------------------------------------------------------------------------
class TestAutoSelect:
    def test_continuous_large_n(self):
        result = ModelTrainer._auto_select(300, EndpointType.CONTINUOUS)
        assert result == "xgb"

    def test_continuous_small_n(self):
        result = ModelTrainer._auto_select(50, EndpointType.CONTINUOUS)
        assert result == "rf"

    def test_binary_large_n(self):
        result = ModelTrainer._auto_select(300, EndpointType.BINARY)
        assert result == "xgb"

    def test_survival_large_n(self):
        result = ModelTrainer._auto_select(300, EndpointType.SURVIVAL)
        assert result == "rsf"

    def test_survival_small_n(self):
        result = ModelTrainer._auto_select(50, EndpointType.SURVIVAL)
        assert result == "cox"

    def test_count_large_n(self):
        result = ModelTrainer._auto_select(300, EndpointType.COUNT)
        assert result == "xgb"

    def test_count_small_n(self):
        result = ModelTrainer._auto_select(50, EndpointType.COUNT)
        assert result == "glm"


# ---------------------------------------------------------------------------
# Small-sample GridSearchCV
# ---------------------------------------------------------------------------
class TestSmallSampleTuning:
    def test_grid_search_triggered_for_small_n(self):
        df = get_demo_data(n_subjects=60)
        pp = Preprocessor()
        X = pp.fit_transform(df, "AVAL")
        y = df["AVAL"].values
        Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")

        trainer = ModelTrainer()
        trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.CONTINUOUS)
        # N=48 in training (<200) should trigger tuning
        assert trainer.is_tuned_
