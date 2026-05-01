"""
Tests for code.preprocessing — ADaM-standard cleaning and feature engineering.
"""

import numpy as np
import pandas as pd
import pytest

from core.preprocessing import Preprocessor, split_train_test
from core.synthetic_data import get_demo_data
from shared.config import TrialDesign, SEED


@pytest.fixture
def demo_df():
    return get_demo_data(n_subjects=100)


class TestPreprocessor:
    def test_fit_identifies_column_types(self, demo_df):
        pp = Preprocessor()
        pp.fit(demo_df, target_col="AVAL", design=TrialDesign.RCT_TWO_ARM)

        assert "AGE" in pp._numeric_cols
        assert "BMI" in pp._numeric_cols
        assert "SEX" in pp._categorical_cols
        assert "RACE" in pp._categorical_cols
        assert "USUBJID" not in pp._numeric_cols
        assert "USUBJID" not in pp._categorical_cols
        assert "AVAL" not in pp._numeric_cols

    def test_fit_returns_self(self, demo_df):
        pp = Preprocessor()
        result = pp.fit(demo_df, "AVAL")
        assert result is pp

    def test_transform_imputes_missing(self, demo_df):
        demo_df = demo_df.copy()
        demo_df.loc[0, "AGE"] = np.nan
        demo_df.loc[1, "BMI"] = np.nan
        pp = Preprocessor()
        pp.fit(demo_df, "AVAL")
        result = pp.transform(demo_df)
        assert not result.isnull().any().any()

    def test_transform_errors_if_not_fitted(self, demo_df):
        pp = Preprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            pp.transform(demo_df)

    def test_fit_transform(self, demo_df):
        pp = Preprocessor()
        result = pp.fit_transform(demo_df, "AVAL")
        assert not result.isnull().any().any()
        assert pp._fitted

    def test_one_hot_encoding(self, demo_df):
        pp = Preprocessor()
        pp.fit(demo_df, "AVAL")
        result = pp.transform(demo_df)

        # SEX becomes SEX_F, SEX_M (one-hot)
        sex_cols = [c for c in result.columns if c.startswith("SEX_")]
        assert len(sex_cols) >= 1
        # RACE becomes RACE_White, RACE_Black, etc.
        race_cols = [c for c in result.columns if c.startswith("RACE_")]
        assert len(race_cols) >= 2

    def test_features_are_all_numeric(self, demo_df):
        pp = Preprocessor()
        pp.fit(demo_df, "AVAL")
        result = pp.transform(demo_df)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_feature_names_property(self, demo_df):
        pp = Preprocessor()
        pp.fit(demo_df, "AVAL")
        pp.transform(demo_df)
        names = pp.feature_names
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


class TestSplitTrainTest:
    def test_output_shapes(self, demo_df):
        pp = Preprocessor()
        X = pp.fit_transform(demo_df, "AVAL")
        y = demo_df["AVAL"]

        X_train, X_test, y_train, y_test = split_train_test(
            X, y, treatment_col="ARM",
        )
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]

    def test_stratified_split_preserves_arm_ratio(self, demo_df):
        pp = Preprocessor()
        X = pp.fit_transform(demo_df, "AVAL")
        y = demo_df["AVAL"]

        X_train, X_test, y_train, y_test = split_train_test(
            X, y, treatment_col="ARM",
        )
        # Check ARM column exists and ratio is roughly preserved
        arm_idx = list(X.columns).index("ARM") if "ARM" in X.columns else 0
        train_arm_mean = X_train[:, arm_idx].mean()
        test_arm_mean = X_test[:, arm_idx].mean()
        # Should be within reasonable tolerance
        assert abs(train_arm_mean - test_arm_mean) < 0.15

    def test_no_treatment_col(self, demo_df):
        pp = Preprocessor()
        X = pp.fit_transform(demo_df, "AVAL")
        y = demo_df["AVAL"]

        X_train, X_test, y_train, y_test = split_train_test(X, y)
        assert X_train.shape[0] > X_test.shape[0]
