"""
Result viewer component.

This module provides components for displaying matching results,
including scores, details, and visualizations.
"""

from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

from ui.utils.adapters import (
    convert_result_for_display,
    format_score,
    format_percentage,
    get_score_color,
    prepare_image_for_display,
)


def render_score_display(
    score: float,
    title: str = "Similarity Score",
    show_interpretation: bool = True,
) -> None:
    """
    Render a prominently displayed similarity score.
    
    Args:
        score: Similarity score [0, 1]
        title: Title to display
        show_interpretation: Whether to show score interpretation
    """
    color = get_score_color(score)
    
    # Large score display
    st.markdown(f"### {title}")
    
    # Score with color
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin: 10px 0;
        ">
            <span style="
                font-size: 48px;
                font-weight: bold;
                color: {color};
            ">
                {format_score(score)}
            </span>
            <br>
            <span style="color: gray;">
                ({format_percentage(score)})
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Interpretation
    if show_interpretation:
        if score >= 0.7:
            st.success("High similarity - likely same finger")
        elif score >= 0.4:
            st.warning("Moderate similarity - inconclusive")
        else:
            st.error("Low similarity - likely different fingers")


def render_score_gauge(
    score: float,
    width: int = 300,
    height: int = 40,
) -> None:
    """
    Render a gauge-style score visualization.
    
    Args:
        score: Similarity score [0, 1]
        width: Width of gauge in pixels
        height: Height of gauge in pixels
    """
    # Progress bar style visualization
    st.progress(score, text=f"Score: {format_score(score)}")


def render_details_panel(
    details: Dict[str, Any],
    title: str = "Detailed Results",
    exclude_keys: Optional[set] = None,
) -> None:
    """
    Render detailed matching results.
    
    Args:
        details: Dictionary of detailed results
        title: Title for the panel
        exclude_keys: Keys to exclude from display
    """
    exclude_keys = exclude_keys or set()
    
    st.markdown(f"### {title}")
    
    # Convert for display
    display_details = convert_result_for_display(details)
    
    # Display error/warning messages first if present
    if "error" in display_details:
        st.error(f"⚠️ **Error:** {display_details['error']}")
        exclude_keys.add("error")
    
    if "warning" in display_details:
        st.warning(f"⚠️ **Warning:** {display_details['warning']}")
        exclude_keys.add("warning")
    
    # Group by type
    metrics = {}
    dicts = {}
    others = {}
    
    for key, value in display_details.items():
        if key in exclude_keys:
            continue
        
        if isinstance(value, (int, float)):
            metrics[key] = value
        elif isinstance(value, dict):
            dicts[key] = value
        else:
            others[key] = value
    
    # Display metrics in columns
    if metrics:
        cols = st.columns(min(len(metrics), 4))
        for idx, (key, value) in enumerate(metrics.items()):
            with cols[idx % len(cols)]:
                label = key.replace("_", " ").title()
                if isinstance(value, float):
                    st.metric(label, f"{value:.4f}")
                else:
                    st.metric(label, str(value))
    
    # Display nested dictionaries
    for key, value in dicts.items():
        label = key.replace("_", " ").title()
        with st.expander(label):
            _render_nested_dict(value)
    
    # Display other values
    if others:
        with st.expander("Additional Information"):
            for key, value in others.items():
                label = key.replace("_", " ").title()
                st.write(f"**{label}:** {value}")


def _render_nested_dict(d: Dict[str, Any], level: int = 0) -> None:
    """
    Recursively render a nested dictionary.
    
    Args:
        d: Dictionary to render
        level: Current nesting level
    """
    for key, value in d.items():
        label = key.replace("_", " ").title()
        
        if isinstance(value, dict):
            st.markdown(f"{'  ' * level}**{label}:**")
            _render_nested_dict(value, level + 1)
        elif isinstance(value, float):
            st.markdown(f"{'  ' * level}- {label}: {value:.4f}")
        else:
            st.markdown(f"{'  ' * level}- {label}: {value}")


def render_result_viewer(
    result: Dict[str, Any],
    show_score: bool = True,
    show_details: bool = True,
    show_visualization: bool = True,
) -> None:
    """
    Render a complete result viewer.
    
    Args:
        result: Full result dictionary from matching
        show_score: Whether to show score display
        show_details: Whether to show details panel
        show_visualization: Whether to show visualizations
    """
    if result is None:
        st.info("No results to display. Run matching first.")
        return
    
    # Score display
    if show_score and "score" in result:
        render_score_display(result["score"])
    
    # Model info
    if "model_name" in result:
        st.caption(f"Method: {result['model_name']}")
    
    st.markdown("---")
    
    # Check if visualization data actually exists and has content
    has_viz_data = (
        show_visualization 
        and "visualization_data" in result 
        and result["visualization_data"] is not None
        and len(result["visualization_data"]) > 0
    )
    has_details = show_details and "details" in result
    
    # Tabs for organization
    if has_details or has_viz_data:
        tab_names = []
        
        if has_details:
            tab_names.append("Details")
        if has_viz_data:
            tab_names.append("Visualization")
        
        if tab_names:
            tabs = st.tabs(tab_names)
            tab_idx = 0
            
            if has_details:
                with tabs[tab_idx]:
                    render_details_panel(result["details"])
                tab_idx += 1
            
            if has_viz_data:
                with tabs[tab_idx]:
                    render_visualization_panel(result["visualization_data"])


def render_visualization_panel(
    visualization_data: Dict[str, Any],
) -> None:
    """
    Render visualizations from matching results.
    
    Args:
        visualization_data: Dictionary of visualization data
    """
    if visualization_data is None:
        st.info("No visualization data available.")
        return
    
    for key, value in visualization_data.items():
        label = key.replace("_", " ").title()
        
        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                # It's an image/map
                st.markdown(f"**{label}**")
                
                # Normalize for display
                display = prepare_image_for_display(
                    (value - value.min()) / (value.max() - value.min() + 1e-10)
                )
                st.image(display, width='stretch')
            else:
                st.markdown(f"**{label}:** Array with shape {value.shape}")
        
        elif isinstance(value, list):
            # Could be minutiae points, matches, etc.
            if len(value) > 0 and isinstance(value[0], dict):
                st.markdown(f"**{label}** ({len(value)} items)")
                
                # Show first few items
                with st.expander("View data"):
                    for idx, item in enumerate(value[:10]):
                        st.json(item)
                    if len(value) > 10:
                        st.caption(f"... and {len(value) - 10} more items")
            else:
                st.markdown(f"**{label}:** {len(value)} items")


def render_score_history(
    history: list,
    max_items: int = 10,
) -> None:
    """
    Render a history of matching scores.
    
    Args:
        history: List of result dictionaries
        max_items: Maximum items to display
    """
    if not history:
        st.info("No history available.")
        return
    
    st.markdown("### Recent Results")
    
    # Show recent results
    recent = history[-max_items:][::-1]
    
    for idx, result in enumerate(recent):
        score = result.get("score", 0)
        model = result.get("model_name", "Unknown")
        color = get_score_color(score)
        
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: space-between;
                padding: 5px 10px;
                border-left: 4px solid {color};
                margin: 5px 0;
                background-color: #f9f9f9;
            ">
                <span>{model}</span>
                <span style="font-weight: bold; color: {color};">
                    {format_score(score)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
