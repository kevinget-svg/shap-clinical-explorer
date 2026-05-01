"""
Tests for code.visualization — publication-ready SHAP plotting functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.modeling import ModelTrainer
from core.preprocessing import Preprocessor, split_train_test
from core.shap_analysis import SHAPAnalyzer
from core.synthetic_data import (
    get_demo_data,
    get_demo_data_binary,
    get_demo_data_survival,
    get_demo_data_count,
)
from core.visualization import (
    plot_beeswarm,
    plot_summary_bar,
    plot_dependence,
    plot_waterfall,
    plot_rct_comparison,
    plot_summary_panel,
    plot_roc_curve,
    plot_survshap_panel,
    plot_survshap_aggregated,
    plot_count_obs_vs_pred,
    plot_count_panel,
    make_prefix,
)
from shared.config import TrialDesign, EndpointType


@pytest.fixture
def continuous_setup():
    """Generate SHAP values for continuous endpoint."""
    df = get_demo_data(n_subjects=80)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values
    feature_names = pp.feature_names
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")

    trainer = ModelTrainer()
    model = trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.CONTINUOUS)

    analyzer = SHAPAnalyzer()
    shap_vals = analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        yield shap_vals, Xte, yte, feature_names, model, Path(tmpdir)


@pytest.fixture
def binary_setup():
    """Generate SHAP values for binary endpoint."""
    df = get_demo_data_binary(n_subjects=80)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values.astype(int)
    feature_names = pp.feature_names
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")

    trainer = ModelTrainer()
    model = trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.BINARY)

    analyzer = SHAPAnalyzer()
    shap_vals = analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        yield shap_vals, Xte, yte, feature_names, model, Path(tmpdir)


@pytest.fixture
def survival_setup():
    """Generate SurvSHAP data for survival endpoint."""
    from sksurv.util import Surv
    df = get_demo_data_survival(n_subjects=100)
    pp = Preprocessor()
    X_df = pp.fit_transform(df, "AVAL")
    feature_names = pp.feature_names
    cnsr = df["CNSR"].values.astype(bool)
    time = df["AVAL"].values.astype(float)
    y = Surv.from_arrays(event=~cnsr, time=time)

    Xtr_arr, Xte_arr, ytr, yte = split_train_test(X_df, y, treatment_col="ARM")
    X_train_df = pd.DataFrame(Xtr_arr, columns=feature_names)
    X_test_df = pd.DataFrame(Xte_arr, columns=feature_names)

    trainer = ModelTrainer()
    model = trainer.train(Xtr_arr, ytr, model_type="cox", endpoint=EndpointType.SURVIVAL)

    analyzer = SHAPAnalyzer()
    shap_vals = analyzer.compute_survival(
        model, X_train_df, X_test_df, ytr, feature_names,
        model_type="cox", max_samples=20,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yield shap_vals, Xte_arr, yte, feature_names, model, analyzer, Path(tmpdir)


@pytest.fixture
def count_setup():
    """Generate SHAP values for count endpoint."""
    df = get_demo_data_count(n_subjects=80)
    pp = Preprocessor()
    X = pp.fit_transform(df, "AVAL")
    y = df["AVAL"].values.astype(float)
    feature_names = pp.feature_names
    Xtr, Xte, ytr, yte = split_train_test(X, y, treatment_col="ARM")

    trainer = ModelTrainer()
    model = trainer.train(Xtr, ytr, model_type="rf", endpoint=EndpointType.COUNT)

    analyzer = SHAPAnalyzer()
    shap_vals = analyzer.compute(model, Xtr, Xte, model_type="rf", max_samples=30)

    with tempfile.TemporaryDirectory() as tmpdir:
        yield shap_vals, Xte, yte, feature_names, model, Path(tmpdir)


# ---------------------------------------------------------------------------
# Make Prefix
# ---------------------------------------------------------------------------
class TestMakePrefix:
    def test_rct_continuous(self):
        assert make_prefix(TrialDesign.RCT_TWO_ARM, EndpointType.CONTINUOUS) == "RCT_CONT"

    def test_single_binary(self):
        assert make_prefix(TrialDesign.SINGLE_ARM, EndpointType.BINARY) == "SIG_BINARY"

    def test_rct_count(self):
        assert make_prefix(TrialDesign.RCT_TWO_ARM, EndpointType.COUNT) == "RCT_COUNT"


# ---------------------------------------------------------------------------
# Standard SHAP Plots
# ---------------------------------------------------------------------------
class TestBeeswarm:
    def test_creates_figure_and_files(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        fig = plot_beeswarm(shap_vals, Xte, fnames, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_beeswarm.png").exists()
        assert (tmpdir / "TEST_beeswarm.svg").exists()


class TestSummaryBar:
    def test_creates_figure_and_files(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        fig = plot_summary_bar(shap_vals, fnames, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_summary_bar.png").exists()


class TestDependence:
    def test_creates_figure_for_arm(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        fig = plot_dependence(shap_vals, Xte, fnames, tmpdir, "TEST", target_feature="ARM")
        assert fig is not None
        assert (tmpdir / "TEST_dependence_ARM.png").exists()

    def test_invalid_feature_raises(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        with pytest.raises(ValueError):
            plot_dependence(shap_vals, Xte, fnames, tmpdir, "TEST", target_feature="NOT_A_FEATURE")


class TestWaterfall:
    def test_creates_figure(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        fig = plot_waterfall(shap_vals, fnames, sample_idx=0, output_dir=tmpdir, prefix="TEST")
        assert fig is not None
        assert (tmpdir / "TEST_waterfall_sample0.png").exists()


class TestRCTComparison:
    def test_creates_figure(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        arm_idx = fnames.index("ARM") if "ARM" in fnames else 0
        fig = plot_rct_comparison(shap_vals, Xte, fnames, arm_idx, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_rct_comparison.png").exists()


class TestSummaryPanel:
    def test_creates_16_9_panel(self, continuous_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = continuous_setup
        fig = plot_summary_panel(shap_vals, Xte, fnames, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_summary_panel.png").exists()


# ---------------------------------------------------------------------------
# Binary-specific
# ---------------------------------------------------------------------------
class TestROC:
    def test_roc_curve(self, binary_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = binary_setup
        fig = plot_roc_curve(model, Xte, yte, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_roc_curve.png").exists()


# ---------------------------------------------------------------------------
# Survival-specific
# ---------------------------------------------------------------------------
class TestSurvSHAP:
    def test_survshap_panel(self, survival_setup):
        shap_vals, Xte, yte, fnames, model, analyzer, tmpdir = survival_setup
        if analyzer.survshap_values_ is not None:
            fig = plot_survshap_panel(
                analyzer.survshap_values_, fnames, analyzer.survshap_times_,
                tmpdir, "TEST", max_features=5,
            )
            assert fig is not None
            assert (tmpdir / "TEST_survshap_panel.png").exists()

    def test_survshap_aggregated(self, survival_setup):
        shap_vals, Xte, yte, fnames, model, analyzer, tmpdir = survival_setup
        if analyzer.survshap_values_ is not None:
            fig = plot_survshap_aggregated(
                analyzer.survshap_values_, Xte, fnames,
                tmpdir, "TEST", sample_idx=0,
                times=analyzer.survshap_times_,
            )
            assert fig is not None
            assert (tmpdir / "TEST_survshap_decomp_sample0.png").exists()


# ---------------------------------------------------------------------------
# Count-specific
# ---------------------------------------------------------------------------
class TestCountPlots:
    def test_count_obs_vs_pred(self, count_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = count_setup
        fig = plot_count_obs_vs_pred(model, Xte, yte, fnames, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_count_obs_vs_pred.png").exists()

    def test_count_panel(self, count_setup):
        shap_vals, Xte, yte, fnames, model, tmpdir = count_setup
        fig = plot_count_panel(shap_vals, Xte, yte, fnames, model, tmpdir, "TEST")
        assert fig is not None
        assert (tmpdir / "TEST_count_panel.png").exists()
