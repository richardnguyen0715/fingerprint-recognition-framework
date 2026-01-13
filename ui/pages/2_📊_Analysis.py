"""
Analysis page for threshold and score analysis.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ui.state.session_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Analysis - Fingerprint Recognition",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
initialize_session_state()

# Import view module
from ui.views.analysis import render_analysis_page

# Render the page
render_analysis_page()
