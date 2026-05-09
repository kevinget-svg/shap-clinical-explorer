"""
Clinical data loader supporting multiple input formats.

Supported formats: .csv, .xlsx, .sas7bdat, .RData
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from shared.constants import ADSL_REQUIRED_COLS

logger = logging.getLogger(__name__)

# Map of supported extensions to their pandas-compatible reader
_READERS = {
    ".csv": lambda p: pd.read_csv(p),
    ".xlsx": lambda p: pd.read_excel(p),
    ".sas7bdat": lambda p: _read_sas(p),
    ".rdata": lambda p: _read_rdata(p),
    ".rda": lambda p: _read_rdata(p),
}


def _read_sas(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Reading .sas7bdat files requires pyreadstat. "
            "Install it with: pip install pyreadstat"
        )
    df, meta = pyreadstat.read_sas7bdat(str(path))
    logger.info(f"SAS metadata: {meta.column_names_to_labels}")
    return df


def _read_rdata(path: Path) -> pd.DataFrame:
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "Reading .RData files requires pyreadstat. "
            "Install it with: pip install pyreadstat"
        )
    df, meta = pyreadstat.read_rdata(str(path))
    # RData may contain multiple dataframes; return the first
    if isinstance(df, dict):
        name = list(df.keys())[0]
        logger.info(f"RData: using table '{name}'")
        return df[name]
    return df


def load_clinical_data(path: str | Path) -> pd.DataFrame:
    """Load clinical data from a supported file format.

    Parameters
    ----------
    path : str or Path
        Path to .csv, .xlsx, .sas7bdat, or .RData/.rda file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()

    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: {list(_READERS.keys())}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = reader(path)
    logger.info(f"Loaded {path.name}: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def validate_clinical_data(
    df: pd.DataFrame,
    required_cols: Optional[list[str]] = None,
) -> list[str]:
    """Validate that required columns exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded clinical dataset.
    required_cols : list[str], optional
        Columns that must be present. Defaults to ADSL_REQUIRED_COLS.

    Returns
    -------
    list[str]
        List of missing columns (empty if valid).
    """
    if required_cols is None:
        required_cols = ADSL_REQUIRED_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing ADaM columns: {missing}")
    else:
        logger.info("All required columns present.")
    return missing


def get_data_summary(df: pd.DataFrame) -> str:
    """Return a basic statistical summary suitable for logging.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
    """
    lines = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} cols",
        f"Dtypes:\n{df.dtypes.to_string()}",
        f"Missing (%):\n{(df.isnull().mean() * 100).round(2).to_string()}",
    ]
    # For numeric columns, add describe
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        lines.append(f"Numeric summary:\n{df[num_cols].describe().to_string()}")
    return "\n".join(lines)
