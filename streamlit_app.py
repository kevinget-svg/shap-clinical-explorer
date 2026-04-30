"""
Streamlit Community Cloud entry point for SHAP Clinical Explorer.

The real app lives at code/streamlit_app.py.  This root-level wrapper
ensures the local ``code/`` package takes priority over Python's stdlib
``code`` module (Python ≥ 3.12) and then delegates to the real app.
"""

import sys
from pathlib import Path

# Put the project root first on sys.path so that `import code` finds our
# local code/ directory, not the standard-library code module.
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Executing the import runs all module-level Streamlit UI code
# (set_page_config, sidebar, tabs, …) inside the real app.
import code.streamlit_app  # noqa: F401 – side-effect import is intentional
