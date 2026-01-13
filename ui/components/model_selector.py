"""
Model selection component.

This module provides components for selecting fingerprint matching
models from the registry.
"""

from typing import List, Optional, Tuple

import streamlit as st

from src.registry import MatcherRegistry, MatcherInfo, get_registry


def render_model_selector(
    key: str = "model_selector",
    show_description: bool = True,
    category_filter: Optional[str] = None,
) -> Optional[str]:
    """
    Render a model selector dropdown.
    
    Args:
        key: Unique key for Streamlit widget state
        show_description: Whether to show model description
        category_filter: Optional category to filter models
        
    Returns:
        Selected model ID or None if no selection
    """
    registry = get_registry()
    
    # Get available matchers
    if category_filter:
        matchers = registry.list_by_category(category_filter)
    else:
        matchers = registry.list_matchers()
    
    if not matchers:
        st.warning("No recognition methods available.")
        return None
    
    # Build options
    options = {m.id: m.name for m in matchers}
    
    # Create selectbox
    selected_name = st.selectbox(
        label="Select Recognition Method",
        options=list(options.values()),
        key=key,
        help="Choose a fingerprint matching algorithm",
        label_visibility="collapsed",
    )
    
    # Get selected ID
    selected_id = None
    for matcher_id, name in options.items():
        if name == selected_name:
            selected_id = matcher_id
            break
    
    # Show description
    if show_description and selected_id:
        matcher_info = registry.get_matcher_info(selected_id)
        if matcher_info:
            st.caption(matcher_info.description)
    
    return selected_id


def render_model_selector_multi(
    key: str = "model_selector_multi",
    default_selection: Optional[List[str]] = None,
) -> List[str]:
    """
    Render a multi-select model selector.
    
    Args:
        key: Unique key for Streamlit widget state
        default_selection: Optional list of default selected IDs
        
    Returns:
        List of selected model IDs
    """
    registry = get_registry()
    matchers = registry.list_matchers()
    
    if not matchers:
        st.warning("No recognition methods available.")
        return []
    
    # Build options
    options = {m.name: m.id for m in matchers}
    
    # Determine default values
    default_names = []
    if default_selection:
        for m in matchers:
            if m.id in default_selection:
                default_names.append(m.name)
    
    # Create multiselect
    selected_names = st.multiselect(
        label="Select Methods to Compare",
        options=list(options.keys()),
        default=default_names,
        key=key,
        help="Choose one or more algorithms to compare",
    )
    
    # Convert names back to IDs
    selected_ids = [options[name] for name in selected_names]
    
    return selected_ids


def render_model_info(
    model_id: str,
    show_parameters: bool = True,
) -> None:
    """
    Render detailed information about a model.
    
    Args:
        model_id: ID of the model to display
        show_parameters: Whether to show parameter information
    """
    registry = get_registry()
    matcher_info = registry.get_matcher_info(model_id)
    
    if matcher_info is None:
        st.error(f"Model '{model_id}' not found in registry.")
        return
    
    # Model name and category
    st.markdown(f"### {matcher_info.name}")
    st.caption(f"Category: {matcher_info.category.title()}")
    
    # Description
    st.markdown(matcher_info.description)
    
    # Parameters
    if show_parameters and matcher_info.parameters:
        st.markdown("**Configurable Parameters:**")
        for param in matcher_info.parameters:
            with st.expander(param.display_name):
                st.write(param.description)
                st.write(f"- **Type:** {param.param_type.value}")
                st.write(f"- **Default:** {param.default}")
                
                if param.min_value is not None:
                    st.write(f"- **Min:** {param.min_value}")
                if param.max_value is not None:
                    st.write(f"- **Max:** {param.max_value}")
                if param.options:
                    st.write(f"- **Options:** {', '.join(param.options)}")


def render_category_tabs() -> Tuple[str, Optional[str]]:
    """
    Render category tabs for model selection.
    
    Returns:
        Tuple of (selected_tab, selected_model_id)
    """
    registry = get_registry()
    categories = registry.get_categories()
    
    if not categories:
        st.warning("No model categories available.")
        return "All", None
    
    # Add "All" category
    tabs = ["All"] + sorted(categories)
    
    selected_tab = st.radio(
        label="Category",
        options=tabs,
        horizontal=True,
        label_visibility="collapsed",
    )
    
    # Get matchers for selected category
    if selected_tab == "All":
        selected_id = render_model_selector(
            key="model_selector_all",
            show_description=True,
        )
    else:
        selected_id = render_model_selector(
            key=f"model_selector_{selected_tab}",
            show_description=True,
            category_filter=selected_tab,
        )
    
    return selected_tab, selected_id
