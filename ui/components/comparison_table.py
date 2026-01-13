"""
Comparison table component.

This module provides components for comparing results from
multiple fingerprint matching methods.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from ui.utils.adapters import format_score, get_score_color


def render_comparison_table(
    results: Dict[str, Dict[str, Any]],
    highlight_best: bool = True,
) -> None:
    """
    Render a comparison table of multiple matching results.
    
    Args:
        results: Dictionary mapping model_id to result dictionary
        highlight_best: Whether to highlight the best score
    """
    if not results:
        st.info("No comparison results available.")
        return
    
    # Find best score
    best_score = max(r.get("score", 0) for r in results.values())
    
    # Build table data
    st.markdown("### Comparison Results")
    
    # Create columns for the table
    cols = st.columns([2, 1, 1, 2])
    
    # Header
    with cols[0]:
        st.markdown("**Method**")
    with cols[1]:
        st.markdown("**Score**")
    with cols[2]:
        st.markdown("**Rank**")
    with cols[3]:
        st.markdown("**Key Metrics**")
    
    st.markdown("---")
    
    # Sort by score
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("score", 0),
        reverse=True,
    )
    
    # Render rows
    for rank, (model_id, result) in enumerate(sorted_results, start=1):
        score = result.get("score", 0)
        model_name = result.get("model_name", model_id)
        details = result.get("details", {})
        
        # Highlight best
        is_best = highlight_best and score == best_score
        color = get_score_color(score)
        
        cols = st.columns([2, 1, 1, 2])
        
        with cols[0]:
            if is_best:
                st.markdown(f"**{model_name}** (Best)")
            else:
                st.markdown(model_name)
        
        with cols[1]:
            st.markdown(
                f'<span style="color: {color}; font-weight: bold;">'
                f'{format_score(score)}</span>',
                unsafe_allow_html=True,
            )
        
        with cols[2]:
            st.markdown(f"#{rank}")
        
        with cols[3]:
            # Show key metrics
            key_metrics = _extract_key_metrics(details)
            if key_metrics:
                st.caption(key_metrics)
            else:
                st.caption("-")


def _extract_key_metrics(details: Dict[str, Any]) -> str:
    """
    Extract key metrics for display in comparison.
    
    Args:
        details: Details dictionary from result
        
    Returns:
        Formatted string of key metrics
    """
    metrics = []
    
    # Common metrics to look for
    if "matched_pairs" in details:
        metrics.append(f"Matches: {details['matched_pairs']}")
    if "minutiae_count_a" in details:
        metrics.append(f"Minutiae: {details['minutiae_count_a']}/{details.get('minutiae_count_b', '?')}")
    if "valid_descriptors_a" in details:
        metrics.append(f"Descriptors: {details['valid_descriptors_a']}/{details.get('valid_descriptors_b', '?')}")
    if "mean_ssim" in details:
        metrics.append(f"Mean: {details['mean_ssim']:.3f}")
    if "raw_mse" in details:
        metrics.append(f"MSE: {details['raw_mse']:.4f}")
    if "raw_ncc" in details:
        metrics.append(f"NCC: {details['raw_ncc']:.3f}")
    
    return " | ".join(metrics[:3])  # Limit to 3 metrics


def render_comparison_chart(
    results: Dict[str, Dict[str, Any]],
    chart_type: str = "bar",
) -> None:
    """
    Render a chart comparing matching scores.
    
    Args:
        results: Dictionary mapping model_id to result dictionary
        chart_type: Type of chart ("bar" or "horizontal_bar")
    """
    if not results:
        st.info("No comparison results available.")
        return
    
    # Prepare data
    data = {
        "Method": [],
        "Score": [],
    }
    
    for model_id, result in results.items():
        data["Method"].append(result.get("model_name", model_id))
        data["Score"].append(result.get("score", 0))
    
    # Sort by score
    sorted_indices = sorted(
        range(len(data["Score"])),
        key=lambda i: data["Score"][i],
        reverse=True,
    )
    
    data["Method"] = [data["Method"][i] for i in sorted_indices]
    data["Score"] = [data["Score"][i] for i in sorted_indices]
    
    st.markdown("### Score Comparison")
    
    # Use Streamlit's built-in bar chart
    import pandas as pd
    df = pd.DataFrame(data)
    df = df.set_index("Method")
    
    st.bar_chart(df)


def render_comparison_summary(
    results: Dict[str, Dict[str, Any]],
) -> None:
    """
    Render a summary of the comparison.
    
    Args:
        results: Dictionary mapping model_id to result dictionary
    """
    if not results:
        return
    
    scores = [r.get("score", 0) for r in results.values()]
    
    # Statistics
    best_score = max(scores)
    worst_score = min(scores)
    avg_score = sum(scores) / len(scores)
    score_range = best_score - worst_score
    
    # Find best method
    best_method = None
    for model_id, result in results.items():
        if result.get("score", 0) == best_score:
            best_method = result.get("model_name", model_id)
            break
    
    st.markdown("### Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Score", format_score(best_score))
    with col2:
        st.metric("Average", format_score(avg_score))
    with col3:
        st.metric("Range", format_score(score_range))
    with col4:
        st.metric("Methods Compared", len(results))
    
    if best_method:
        st.success(f"Best performing method: **{best_method}**")


def render_detailed_comparison(
    results: Dict[str, Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> None:
    """
    Render a detailed comparison with expandable sections.
    
    Args:
        results: Dictionary mapping model_id to result dictionary
        metric_keys: Specific metric keys to compare
    """
    if not results:
        return
    
    st.markdown("### Detailed Comparison")
    
    for model_id, result in results.items():
        model_name = result.get("model_name", model_id)
        score = result.get("score", 0)
        details = result.get("details", {})
        
        color = get_score_color(score)
        
        with st.expander(f"{model_name} - Score: {format_score(score)}"):
            # Score display
            st.markdown(
                f'<h3 style="color: {color};">{format_score(score)}</h3>',
                unsafe_allow_html=True,
            )
            
            # Details
            if details:
                if metric_keys:
                    # Show only specified metrics
                    filtered = {k: v for k, v in details.items() if k in metric_keys}
                    _render_details_grid(filtered)
                else:
                    _render_details_grid(details)


def _render_details_grid(details: Dict[str, Any]) -> None:
    """
    Render details in a grid layout.
    
    Args:
        details: Details dictionary
    """
    # Filter to numeric values
    numeric = {
        k: v for k, v in details.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    
    if not numeric:
        st.caption("No numeric metrics available.")
        return
    
    # Display in columns
    cols = st.columns(min(len(numeric), 3))
    
    for idx, (key, value) in enumerate(numeric.items()):
        with cols[idx % len(cols)]:
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                st.metric(label, f"{value:.4f}")
            else:
                st.metric(label, str(value))
