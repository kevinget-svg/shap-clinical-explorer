"""
Streamlit Community Cloud entry point for SHAP Clinical Explorer.

The real app lives at code/streamlit_app.py.  Python's stdlib contains a
``code`` module, so we force-register our local ``code/`` package into
``sys.modules`` before any import can pick up the wrong one.
"""

import importlib.util
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent

# Ensure project root is on sys.path (for shared.*, etc.)
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Force-register our local code/ package so that the stdlib 'code'
# module never gets loaded in its place.
_code_path = _project_root / "code"
_init_path = _code_path / "__init__.py"
_spec = importlib.util.spec_from_file_location(
    "code",
    _init_path,
    submodule_search_locations=[str(_code_path)],
)
_code_pkg = importlib.util.module_from_spec(_spec)
sys.modules["code"] = _code_pkg
_spec.loader.exec_module(_code_pkg)

# Now all internal 'from code.xxx import ...' statements will resolve
# to our local package instead of the stdlib.
import code.streamlit_app  # noqa: F401 – side-effect import is intentional
