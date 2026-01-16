"""
Verification page for 1:1 fingerprint matching.

This page provides the main workflow for comparing two fingerprint
images and displaying the matching result.
"""

from typing import Any, Dict, Optional

import streamlit as st

from ui.components.uploader import render_dual_uploader
from ui.components.model_selector import render_model_selector, render_model_info
from ui.components.config_panel import render_config_panel, get_default_parameters
from ui.components.result_viewer import render_result_viewer, render_score_display
from ui.state.session_state import (
    get_image_a,
    get_image_b,
    set_image_a,
    set_image_b,
    get_selected_model,
    set_selected_model,
    get_model_parameters,
    set_model_parameters,
    get_last_result,
    set_last_result,
    has_images,
)
from ui.utils.validation import validate_image_pair


def render_verification_page() -> None:
    """Render the main verification page."""
    st.header("Fingerprint Verification")
    st.markdown(
        """
        Compare two fingerprint images to determine if they belong to the same person.
        Upload both fingerprints, select a recognition method, and run the matching.
        """
    )
    
    # Main layout
    _render_upload_section()
    
    st.markdown("---")
    
    _render_model_selection()
    
    st.markdown("---")
    
    _render_matching_section()


def _render_upload_section() -> None:
    """Render the image upload section."""
    st.subheader("Step 1: Upload Fingerprints")
    
    # Use dual uploader
    image_a, name_a, image_b, name_b = render_dual_uploader(
        show_preview=True,
        preview_size=(220, 220),
    )
    
    # Update session state
    if image_a is not None:
        current_a = get_image_a()
        if current_a is None or not _arrays_equal(current_a, image_a):
            set_image_a(image_a, name_a)
    
    if image_b is not None:
        current_b = get_image_b()
        if current_b is None or not _arrays_equal(current_b, image_b):
            set_image_b(image_b, name_b)
    
    # Status indicator
    if has_images():
        st.success("Both fingerprints uploaded")
    else:
        if get_image_a() is None and get_image_b() is None:
            st.info("Please upload both fingerprint images")
        elif get_image_a() is None:
            st.warning("Please upload Fingerprint A")
        else:
            st.warning("Please upload Fingerprint B")


def _render_model_selection() -> None:
    """Render the model selection section."""
    st.subheader("Step 2: Select Recognition Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("**Select Recognition Method**")
            selected_id = render_model_selector(
                key="verification_model_selector",
                show_description=True,
            )
            
            if selected_id:
                set_selected_model(selected_id)
    
    with col2:
        with st.container(border=True):
            st.markdown("**⚙️ Configure Parameters**")
            current_model = get_selected_model()
            if current_model:
                # Get current parameters
                current_params = get_model_parameters(current_model)
                
                # Render config panel
                params = render_config_panel(
                    model_id=current_model,
                    current_params=current_params,
                    key_prefix="verification_config",
                    use_expander=False,
                    expanded=True,
                )
                
                # Update stored parameters
                if params:
                    set_model_parameters(current_model, params)
            else:
                st.info("Select a method to configure parameters")


def _render_matching_section() -> None:
    """Render the matching execution and results section."""
    st.subheader("Step 3: Run Matching")
    
    # Check readiness
    ready = _check_matching_readiness()
    
    # Match button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
            match_clicked = st.button(
                "Match Fingerprints",
                type="primary",
                width='stretch',
                disabled=not ready,
            )
        except TypeError:
            match_clicked = st.button(
                "Match Fingerprints",
                type="primary",
                use_container_width=True,
                disabled=not ready,
            )    
    # Run matching if clicked
    if match_clicked and ready:
        _run_matching()
    
    # Display results
    st.markdown("---")
    st.subheader("Step 4: Results")
    
    last_result = get_last_result()
    
    if last_result is not None:
        render_result_viewer(
            result=last_result,
            show_score=True,
            show_details=True,
            show_visualization=True,
        )
    else:
        st.info("No results yet. Upload images, select a method, and click 'Match Fingerprints'.")


def _check_matching_readiness() -> bool:
    """
    Check if all requirements are met for matching.
    
    Returns:
        True if ready to match
    """
    image_a = get_image_a()
    image_b = get_image_b()
    model_id = get_selected_model()
    
    if image_a is None or image_b is None:
        return False
    
    if model_id is None:
        return False
    
    # Validate image pair
    valid, _ = validate_image_pair(image_a, image_b)
    
    return valid


def _run_matching() -> None:
    """Execute fingerprint matching and store result."""
    image_a = get_image_a()
    image_b = get_image_b()
    model_id = get_selected_model()
    
    if image_a is None or image_b is None or model_id is None:
        st.error("Missing required inputs.")
        return
    
    # Get parameters
    params = get_model_parameters(model_id)
    if not params:
        params = get_default_parameters(model_id)
    
    # Run matching with progress
    with st.spinner("Matching fingerprints..."):
        try:
            from src.registry import get_registry
            
            registry = get_registry()
            matcher = registry.create_matcher(model_id, **params)
            
            if matcher is None:
                st.error(f"Failed to create matcher '{model_id}'")
                return
            
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
            
            set_last_result(result_dict)
            st.success("✅ Matching complete!")
            
        except Exception as e:
            st.error(f"Matching failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def _arrays_equal(a, b) -> bool:
    """Check if two numpy arrays are equal."""
    import numpy as np
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return np.allclose(a, b)
