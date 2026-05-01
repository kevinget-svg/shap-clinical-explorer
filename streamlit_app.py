"""
Streamlit Community Cloud entry point for SHAP Clinical Explorer.

The real app lives at core/streamlit_app.py (the package was named 'core'
instead of 'code' to avoid shadowing Python's stdlib code module).
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for local development
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Executing the import runs all module-level Streamlit UI code
# (set_page_config, sidebar, tabs, …) inside the real app.
import core.streamlit_app  # noqa: F401 – side-effect import is intentional
