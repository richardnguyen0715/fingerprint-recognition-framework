"""
Comparison page for multi-method fingerprint matching.

This page allows users to run multiple recognition methods on
the same fingerprint pair and compare their results.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from ui.components.uploader import render_dual_uploader
from ui.components.model_selector import render_model_selector_multi
from ui.components.config_panel import render_config_panel, get_default_parameters
from ui.components.comparison_table import (
    render_comparison_table,
    render_comparison_chart,
    render_comparison_summary,
    render_detailed_comparison,
)
from ui.state.session_state import (
    get_image_a,
    get_image_b,
    set_image_a,
    set_image_b,
    get_comparison_models,
    set_comparison_models,
    get_comparison_results,
    set_comparison_result,
    clear_comparison_results,
    has_images,
    get_model_parameters,
)
from ui.utils.validation import validate_image_pair


def render_comparison_page() -> None:
    """Render the method comparison page."""
    st.header("Method Comparison")
    st.markdown(
        """
        Compare multiple recognition methods on the same fingerprint pair.
        This helps you understand how different algorithms perform and
        choose the best method for your use case.
        """
    )
    
    # Main layout sections
    _render_image_section()
    
    st.markdown("---")
    
    _render_method_selection()
    
    st.markdown("---")
    
    _render_comparison_execution()
    
    st.markdown("---")
    
    _render_comparison_results()


def _render_image_section() -> None:
    """Render the image upload section."""
    st.subheader("Step 1: Fingerprint Images")
    
    # Check if images are already in session
    image_a = get_image_a()
    image_b = get_image_b()
    
    if image_a is not None and image_b is not None:
        st.success("Using images from verification page")
        
        # Show previews
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fingerprint A**")
            from ui.utils.adapters import prepare_image_for_display
            st.image(
                prepare_image_for_display(image_a, (200, 200)),
                width='content',
            )
        with col2:
            st.markdown("**Fingerprint B**")
            st.image(
                prepare_image_for_display(image_b, (200, 200)),
                width='content',
            )
        
        # Option to upload new images
        with st.expander("Upload different images"):
            _render_upload_widget()
    else:
        st.info("Please upload fingerprint images")
        _render_upload_widget()


def _render_upload_widget() -> None:
    """Render the upload widget."""
    image_a, name_a, image_b, name_b = render_dual_uploader(
        show_preview=True,
        preview_size=(180, 180),
    )
    
    # Update session state
    if image_a is not None:
        current = get_image_a()
        if current is None or not _arrays_equal(current, image_a):
            set_image_a(image_a, name_a)
            clear_comparison_results()  # Clear old results
    
    if image_b is not None:
        current = get_image_b()
        if current is None or not _arrays_equal(current, image_b):
            set_image_b(image_b, name_b)
            clear_comparison_results()


def _render_method_selection() -> None:
    """Render the method selection section."""
    st.subheader("Step 2: Select Methods to Compare")
    
    # Get currently selected models
    current_selection = get_comparison_models()
    
    # Multi-select
    selected = render_model_selector_multi(
        key="comparison_model_selector",
        default_selection=current_selection,
    )
    
    # Update state
    if selected != current_selection:
        set_comparison_models(selected)
        clear_comparison_results()  # Clear old results when selection changes
    
    # Show selection summary
    if selected:
        st.caption(f"Selected {len(selected)} method(s) for comparison")
    else:
        st.warning("Please select at least one method to compare")


def _render_comparison_execution() -> None:
    """Render the comparison execution section."""
    st.subheader("Step 3: Run Comparison")
    
    # Check readiness
    image_a = get_image_a()
    image_b = get_image_b()
    selected_models = get_comparison_models()
    
    ready = (
        image_a is not None
        and image_b is not None
        and len(selected_models) > 0
    )
    
    if not ready:
        missing = []
        if image_a is None or image_b is None:
            missing.append("fingerprint images")
        if len(selected_models) == 0:
            missing.append("recognition methods")
        st.warning(f"Missing: {', '.join(missing)}")
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
            run_clicked = st.button(
                "Run All Methods",
                type="primary",
                width='stretch',
                disabled=not ready,
            )
        except TypeError:
            run_clicked = st.button(
                "Run All Methods",
                type="primary",
                use_container_width=True,
                disabled=not ready,
            )
    
    if run_clicked and ready:
        _run_all_comparisons()


def _run_all_comparisons() -> None:
    """Execute all selected matching methods."""
    from src.registry import get_registry
    
    image_a = get_image_a()
    image_b = get_image_b()
    selected_models = get_comparison_models()
    
    registry = get_registry()
    
    # Clear previous results
    clear_comparison_results()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(selected_models)
    
    for idx, model_id in enumerate(selected_models):
        # Update progress
        progress = (idx) / total
        progress_bar.progress(progress)
        
        matcher_info = registry.get_matcher_info(model_id)
        model_name = matcher_info.name if matcher_info else model_id
        status_text.text(f"Running {model_name}...")
        
        try:
            # Get parameters (use defaults)
            params = get_model_parameters(model_id)
            if not params:
                params = get_default_parameters(model_id)
            
            # Create and run matcher
            matcher = registry.create_matcher(model_id, **params)
            
            if matcher is None:
                st.warning(f"Failed to create matcher: {model_id}")
                continue
            
            # Execute matching
            result = matcher.match(image_a, image_b)
            
            # Store result
            result_dict = {
                "score": result.score,
                "details": result.details,
                "visualization_data": result.visualization_data,
                "model_id": model_id,
                "model_name": matcher.name,
                "parameters": params,
            }
            
            set_comparison_result(model_id, result_dict)
            
        except Exception as e:
            st.error(f"Error running {model_name}: {str(e)}")
    
    # Complete
    progress_bar.progress(1.0)
    status_text.text("Comparison complete!")
    st.success(f"Completed {total} method(s)")


def _render_comparison_results() -> None:
    """Render comparison results."""
    st.subheader("Step 4: Comparison Results")
    
    results = get_comparison_results()
    
    if not results:
        st.info("No comparison results yet. Select methods and click 'Run All Methods'.")
        return
    
    # Tabs for different views
    tabs = st.tabs([
        "Summary",
        "Table",
        "Chart",
        "Details",
    ])
    
    with tabs[0]:
        render_comparison_summary(results)
    
    with tabs[1]:
        render_comparison_table(results, highlight_best=True)
    
    with tabs[2]:
        render_comparison_chart(results)
    
    with tabs[3]:
        render_detailed_comparison(results)
    
    # Export option
    st.markdown("---")
    _render_export_option(results)


def _render_export_option(results: Dict[str, Dict[str, Any]]) -> None:
    """Render option to export results."""
    st.markdown("### Export Results")
    
    # Prepare export data
    export_data = []
    for model_id, result in results.items():
        row = {
            "method": result.get("model_name", model_id),
            "score": result.get("score", 0),
        }
        # Add key details
        details = result.get("details", {})
        for key, value in details.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
        export_data.append(row)
    
    # Convert to CSV
    import pandas as pd
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="comparison_results.csv",
        mime="text/csv",
    )


def _arrays_equal(a, b) -> bool:
    """Check if two numpy arrays are equal."""
    import numpy as np
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return np.allclose(a, b)
