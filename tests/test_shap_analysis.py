"""
Tests for code.shap_analysis — SHAP computation and feature importance.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from code.modeling import ModelTrainer
from code.preprocessing import Preprocessor, split_train_test
from code.shap_analysis import SHAPAnalyzer
from code.synthetic_data import (
    get_demo_data,
    get_demo_data_survival,
)
from shared.config import EndpointType


@pytest.fixture
def continuous_data():
    """Train a small RF model and return everything needed for SHAP."""
    df = get_demo_data(n_subjects=100)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values
    feature_names = pp.feature_names
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")

    trainer = ModelTrainer()
    model = trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.CONTINUOUS)
    return Xtr, Xte, ytr, yte, feature_names, model


@pytest.fixture
def glm_continuous_data():
    """Train a small GLM model for LinearExplainer testing."""
    df = get_demo_data(n_subjects=80)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values
    feature_names = pp.feature_names
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")

    trainer = ModelTrainer()
    model = trainer.train(Xtr, ytr, model_type="glm", endpoint=EndpointType.CONTINUOUS)
    return Xtr, Xte, ytr, yte, feature_names, model


@pytest.fixture
def survival_data():
    """Train a Cox PH model and return data for survshap testing."""
    from sksurv.util import Surv
    df = get_demo_data_survival(n_subjects=120)
    pp = Preprocessor()
    X_df = pp.fit_transform(df, "AVAL")
    feature_names = pp.feature_names
    cnsr = df["CNSR"].values.astype(bool)
    time = df["AVAL"].values.astype(float)
    y = Surv.from_arrays(event=~cnsr, time=time)

    Xtr_arr, Xte_arr, ytr, yte = split_train_test(X_df, y, treatment_col="ARM")

    trainer = ModelTrainer()
    model = trainer.train(Xtr_arr, ytr, model_type="cox", endpoint=EndpointType.SURVIVAL)
    # For survshap we need DataFrames
    X_train_df = pd.DataFrame(Xtr_arr, columns=feature_names)
    X_test_df = pd.DataFrame(Xte_arr, columns=feature_names)
    return Xtr_arr, Xte_arr, ytr, yte, feature_names, model, X_train_df, X_test_df


class TestCompute:
    def test_tree_explainer(self, continuous_data):
        Xtr, Xte, ytr, yte, feature_names, model = continuous_data
        analyzer = SHAPAnalyzer()
        shap_vals = analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)
        assert shap_vals.shape == (Xte.shape[0], Xte.shape[1])
        assert analyzer.shap_values_ is not None

    def test_linear_explainer(self, glm_continuous_data):
        Xtr, Xte, ytr, yte, feature_names, model = glm_continuous_data
        analyzer = SHAPAnalyzer()
        shap_vals = analyzer.compute(model, Xtr, Xte, model_type="glm", max_samples=30)
        assert shap_vals.shape == (Xte.shape[0], Xte.shape[1])

    def test_survival_shap(self, survival_data):
        Xtr, Xte, ytr, yte, fnames, model, Xtr_df, Xte_df = survival_data
        analyzer = SHAPAnalyzer()
        shap_vals = analyzer.compute_survival(
            model, Xtr_df, Xte_df, ytr, fnames,
            model_type="cox", max_samples=20,
        )
        # 2D time-aggregated result
        assert shap_vals.shape == (min(20, Xte_df.shape[0]), len(fnames))
        # 3D survshap values
        assert analyzer.survshap_values_ is not None
        assert analyzer.survshap_values_.ndim == 3


class TestFeatureImportance:
    def test_returns_ranked_df(self, continuous_data):
        Xtr, Xte, ytr, yte, feature_names, model = continuous_data
        analyzer = SHAPAnalyzer()
        analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)
        imp = analyzer.get_feature_importance(feature_names)
        assert list(imp.columns) == ["feature", "mean_abs_shap", "rank"]
        assert imp["rank"].iloc[0] == 1
        assert imp["mean_abs_shap"].is_monotonic_decreasing

    def test_errors_without_compute(self):
        analyzer = SHAPAnalyzer()
        with pytest.raises(RuntimeError, match="No SHAP values"):
            analyzer.get_feature_importance()

    def test_default_feature_names(self, continuous_data):
        Xtr, Xte, ytr, yte, feature_names, model = continuous_data
        analyzer = SHAPAnalyzer()
        analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)
        imp = analyzer.get_feature_importance()  # auto-generate names
        # Default names are F0..Fn sorted by mean_abs_shap descending
        assert all(c.startswith("F") for c in imp["feature"].tolist())
        assert imp["rank"].iloc[0] == 1


class TestShapDataframe:
    def test_wide_format(self, continuous_data):
        Xtr, Xte, ytr, yte, feature_names, model = continuous_data
        analyzer = SHAPAnalyzer()
        analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)
        df = analyzer.get_shap_dataframe(feature_names)
        assert df.shape == (Xte.shape[0], len(feature_names))
        assert list(df.columns) == feature_names

    def test_errors_without_compute(self):
        analyzer = SHAPAnalyzer()
        with pytest.raises(RuntimeError, match="No SHAP values"):
            analyzer.get_shap_dataframe()


class TestSaveShapValues:
    def test_saves_csv(self, continuous_data):
        Xtr, Xte, ytr, yte, feature_names, model = continuous_data
        analyzer = SHAPAnalyzer()
        analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)
        df = analyzer.get_shap_dataframe(feature_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shap_values.csv"
            analyzer.save_shap_values(df, path)
            assert path.exists()
            loaded = pd.read_csv(path)
            assert loaded.shape == df.shape
