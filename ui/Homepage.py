"""
Streamlit UI for Fingerprint Recognition.

This is the main entry point (Home page) for the Streamlit application.

Usage:
    streamlit run ui/🏠_Homepage.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ui.state.session_state import initialize_session_state

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Home - Fingerprint Recognition",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for better sidebar styling
    st.markdown(
        """
        <style>
        /* Sidebar styling */
        [data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }
        [data-testid="stSidebarNav"]::before {
            content: "🔍 Navigation";
            display: block;
            margin-left: 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
            color: #31333F;
        }
        /* Page link styling */
        [data-testid="stSidebarNav"] a {
            padding: 0.5rem 1rem;
        }
        [data-testid="stSidebarNav"] a:hover {
            background-color: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Application title and description
    st.title("🔍 Fingerprint Recognition System")
    st.markdown(
        """
        Welcome to the Fingerprint Recognition System. This application allows you to:
        
        - **Verify** fingerprints (1:1 matching)
        - **Analyze** matching scores and thresholds
        - **Compare** different recognition methods
        
        **👈 Use the sidebar to navigate between different features.**
        """
    )
    
    st.markdown("---")
    
    # Render home page content
    render_home_page()


def render_home_page() -> None:
    """Render the home page with system overview."""
    st.header("System Overview")
    
    # Display available models
    st.subheader("Available Recognition Methods")
    
    try:
        from src.registry import get_registry
        
        registry = get_registry()
        matchers = registry.list_matchers()
        
        if matchers:
            # Group by category
            categories = registry.get_categories()
            
            for category in sorted(categories):
                category_matchers = registry.list_by_category(category)
                
                with st.expander(
                    f"**{category.title()}** ({len(category_matchers)} methods)",
                    expanded=True,
                ):
                    for matcher_info in category_matchers:
                        st.markdown(f"**{matcher_info.name}**")
                        st.caption(matcher_info.description)
                        
                        if matcher_info.parameters:
                            params_text = ", ".join(
                                p.display_name for p in matcher_info.parameters
                            )
                            st.text(f"Parameters: {params_text}")
                        st.markdown("---")
        else:
            st.warning("No recognition methods found in the registry.")
            
    except ImportError as e:
        st.error(f"Failed to load model registry: {e}")
    
    # Quick start guide
    st.subheader("Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(
            """
            **1️⃣ Verification**
            
            Compare two fingerprints to check if they match.
            
            👉 Go to **Verification** page in sidebar
            """
        )
    
    with col2:
        st.info(
            """
            **2️⃣ Analysis**
            
            Analyze scores and understand thresholds.
            
            👉 Go to **Analysis** page in sidebar
            """
        )
    
    with col3:
        st.info(
            """
            **3️⃣ Comparison**
            
            Compare multiple methods on same images.
            
            👉 Go to **Comparison** page in sidebar
            """
        )


if __name__ == "__main__":
    main()
