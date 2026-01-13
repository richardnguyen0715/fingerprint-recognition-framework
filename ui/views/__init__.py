"""
Views module for the Streamlit UI.
"""

from ui.views.verification import render_verification_page
from ui.views.analysis import render_analysis_page
from ui.views.comparison import render_comparison_page

__all__ = [
    "render_verification_page",
    "render_analysis_page",
    "render_comparison_page",
]

