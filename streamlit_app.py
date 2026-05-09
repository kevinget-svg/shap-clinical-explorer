"""
Streamlit Community Cloud entry point for SHAP Clinical Explorer.

The real app lives at core/streamlit_app.py.  We use ``exec()`` rather
than ``import`` so that Streamlit re-executes the UI code on every
widget-change rerun — a regular ``import`` would be cached in
``sys.modules`` and never run again after the first page load.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent

# Ensure project root is on sys.path so that internal imports like
# ``from core.xxx import ...`` resolve correctly.
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Read and execute the real app inline.
_app_path = _project_root / "core" / "streamlit_app.py"
_source = _app_path.read_text()
_code = compile(_source, str(_app_path), "exec")
# __file__ and __name__ are not auto-populated in exec(); supply them so
# that Path(__file__).resolve() inside the app resolves correctly.
exec(_code, {"__name__": "__main__", "__file__": str(_app_path)})
