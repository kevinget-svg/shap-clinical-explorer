"""
Tests for code.data_loader — clinical data loading and validation.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.data_loader import (
    load_clinical_data,
    validate_clinical_data,
    get_data_summary,
)
from core.synthetic_data import get_demo_data


class TestLoadClinicalData:
    def test_load_csv(self):
        df_in = get_demo_data(n_subjects=50)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df_in.to_csv(f, index=False)
            tmp_path = Path(f.name)

        try:
            df = load_clinical_data(tmp_path)
            assert len(df) == 50
            assert list(df.columns) == list(df_in.columns)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_clinical_data("data.txt")

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_clinical_data("nonexistent_file.csv")


class TestValidateClinicalData:
    def test_all_columns_present(self):
        df = get_demo_data(n_subjects=10)
        missing = validate_clinical_data(df, required_cols=["USUBJID", "ARM", "AGE"])
        assert missing == []

    def test_missing_columns(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        missing = validate_clinical_data(df, required_cols=["A", "C"])
        assert "C" in missing
        assert "A" not in missing

    def test_default_required_cols(self):
        df = get_demo_data(n_subjects=10)
        missing = validate_clinical_data(df)
        # ADSL_REQUIRED_COLS includes SAFFL, which synthetic data doesn't have
        assert "SAFFL" in missing
        assert "USUBJID" not in missing


class TestGetDataSummary:
    def test_returns_string_with_key_info(self):
        df = get_demo_data(n_subjects=10)
        summary = get_data_summary(df)
        assert "10 rows" in summary
        assert "10 cols" in summary
        assert "Missing" in summary
        assert "AVAL" in summary
