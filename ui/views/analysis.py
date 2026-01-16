"""
Analysis page for threshold and score analysis.

This page provides tools for analyzing matching scores and
determining optimal thresholds for verification decisions.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

from ui.state.session_state import (
    get_result_history,
    clear_result_history,
)
from ui.utils.adapters import format_score, get_score_color


def render_analysis_page() -> None:
    """Render the analysis page."""
    st.header("Score Analysis")
    st.markdown(
        """
        Analyze matching scores and explore threshold selection for verification decisions.
        This page helps you understand the distribution of scores and find optimal thresholds.
        """
    )
    
    # Tabs for different analysis views
    tabs = st.tabs([
        "Threshold Analysis",
        "Result History",
        "Score Interpretation",
    ])
    
    with tabs[0]:
        _render_threshold_analysis()
    
    with tabs[1]:
        _render_history_analysis()
    
    with tabs[2]:
        _render_score_interpretation()


def _render_threshold_analysis() -> None:
    """Render threshold analysis section."""
    st.subheader("Threshold Selection")
    
    st.markdown(
        """
        Use this tool to understand how different threshold values affect
        verification decisions. A **threshold** determines the minimum
        similarity score required to accept a match.
        """
    )
    
    # Threshold slider
    threshold = st.slider(
        label="Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Scores above this threshold are considered matches",
    )
    
    # Interactive demonstration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Test a Score")
        test_score = st.number_input(
            label="Enter a similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.01,
        )
        
        # Decision
        if test_score >= threshold:
            st.success(f"✅ **MATCH** (score {format_score(test_score)} ≥ threshold {format_score(threshold)})")
        else:
            st.error(f"❌ **NO MATCH** (score {format_score(test_score)} < threshold {format_score(threshold)})")
    
    with col2:
        st.markdown("### Threshold Guidelines")
        st.markdown(
            f"""
            | Threshold Range | Security Level | Use Case |
            |-----------------|----------------|----------|
            | 0.7 - 1.0 | High | Banking, Law Enforcement |
            | 0.5 - 0.7 | Medium | Access Control |
            | 0.3 - 0.5 | Low | Consumer Applications |
            
            **Current threshold:** {format_score(threshold)}
            """
        )
    
    # Visualize threshold regions
    st.markdown("### Score Regions")
    _render_threshold_visualization(threshold)


def _render_threshold_visualization(threshold: float) -> None:
    """Render visual representation of threshold regions."""
    # Create simple visualization using columns
    cols = st.columns(10)
    
    for i, col in enumerate(cols):
        value = (i + 0.5) / 10  # Center of each segment
        
        if value >= threshold:
            color = "#28a745"  # Green - accept
        else:
            color = "#dc3545"  # Red - reject
        
        with col:
            st.markdown(
                f"""
                <div style="
                    height: 40px;
                    background-color: {color};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 10px;
                ">
                    {i/10:.1f}
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.caption(
        "🟢 Green = Accept (Match) | 🔴 Red = Reject (No Match)"
    )


def _render_history_analysis() -> None:
    """Render analysis of result history."""
    st.subheader("Result History")
    
    history = get_result_history()
    
    if not history:
        st.info(
            "No matching results in history. Run some matches in the "
            "Verification page to see them here."
        )
        return
    
    # Summary statistics
    scores = [r.get("score", 0) for r in history]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(history))
    with col2:
        st.metric("Average Score", format_score(np.mean(scores)))
    with col3:
        st.metric("Highest Score", format_score(max(scores)))
    with col4:
        st.metric("Lowest Score", format_score(min(scores)))
    
    # Score distribution
    st.markdown("### Score Distribution")
    
    # Create histogram using numpy for bin calculation
    import pandas as pd
    hist_counts, bin_edges = np.histogram(scores, bins=10, range=(0, 1))
    
    # Create labels for each bin (as strings to avoid interval type issues)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    
    df = pd.DataFrame({
        "Range": bin_labels,
        "Count": hist_counts
    })
    df = df.set_index("Range")
    st.bar_chart(df)
    
    # Recent results table
    st.markdown("### Recent Results")
    
    results_data = []
    for idx, result in enumerate(reversed(history[-20:])):
        results_data.append({
            "#": len(history) - idx,
            "Method": result.get("model_name", "Unknown"),
            "Score": format_score(result.get("score", 0)),
        })
    
    try:
        st.dataframe(
            results_data,
            width='stretch',
            hide_index=True,
        )
    except TypeError:
        st.dataframe(
            results_data,
            use_container_width=True,
            hide_index=True,
        )
    
    # Clear history button
    st.markdown("---")
    if st.button("Clear History", type="secondary"):
        clear_result_history()
        st.rerun()


def _render_score_interpretation() -> None:
    """Render guide for interpreting scores."""
    st.subheader("Understanding Similarity Scores")
    
    st.markdown(
        """
        Similarity scores in fingerprint recognition represent how closely
        two fingerprints match. Here's how to interpret them:
        """
    )
    
    # Score ranges
    st.markdown("### Score Ranges")
    
    ranges = [
        ("0.8 - 1.0", "Very High", "#28a745", 
         "Strong evidence of same finger. Very confident match."),
        ("0.6 - 0.8", "High", "#5cb85c",
         "Good evidence of same finger. Likely match."),
        ("0.4 - 0.6", "Moderate", "#ffc107",
         "Inconclusive. May need additional verification."),
        ("0.2 - 0.4", "Low", "#f0ad4e",
         "Weak evidence. Likely different fingers."),
        ("0.0 - 0.2", "Very Low", "#dc3545",
         "Strong evidence of different fingers."),
    ]
    
    for range_str, level, color, description in ranges:
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                padding: 10px;
                margin: 5px 0;
                border-left: 5px solid {color};
                background-color: #f9f9f9;
            ">
                <div style="width: 100px; font-weight: bold;">{range_str}</div>
                <div style="width: 100px; color: {color}; font-weight: bold;">{level}</div>
                <div>{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Factors affecting scores
    st.markdown("### Factors Affecting Scores")
    
    st.markdown(
        """
        Several factors can affect the similarity score:
        
        **Image Quality**
        - Resolution and clarity of fingerprint images
        - Presence of noise, scars, or dirt
        - Adequate coverage of the fingerprint area
        
        **Capture Conditions**
        - Pressure applied during capture
        - Skin moisture level
        - Sensor type and quality
        
        **Method-Specific Factors**
        - *Image-based methods (SSIM, MSE)*: Sensitive to rotation and translation
        - *Minutiae-based methods*: Depend on accurate minutiae extraction
        - *Descriptor methods (MCC)*: Require sufficient minutiae count
        
        **Best Practices**
        - Use high-quality fingerprint images (≥500 DPI recommended)
        - Ensure consistent capture conditions
        - Compare results from multiple methods for critical decisions
        """
    )
    
    # Method comparison
    st.markdown("### Method Characteristics")
    
    methods = [
        {
            "Method": "SSIM",
            "Type": "Image-based",
            "Strengths": "Fast, simple, perceptual",
            "Weaknesses": "Alignment sensitive",
        },
        {
            "Method": "MSE",
            "Type": "Image-based",
            "Strengths": "Very fast, simple",
            "Weaknesses": "Very alignment sensitive",
        },
        {
            "Method": "NCC",
            "Type": "Image-based",
            "Strengths": "Intensity invariant",
            "Weaknesses": "Alignment sensitive",
        },
        {
            "Method": "Minutiae",
            "Type": "Feature-based",
            "Strengths": "Rotation invariant, interpretable",
            "Weaknesses": "Depends on extraction quality",
        },
        {
            "Method": "MCC",
            "Type": "Descriptor-based",
            "Strengths": "Robust, discriminative",
            "Weaknesses": "Requires many minutiae",
        },
        {
            "Method": "CNN Embedding",
            "Type": "Deep Learning",
            "Strengths": "Learns discriminative features",
            "Weaknesses": "Requires training, pretrained weights",
        },
        {
            "Method": "Patch CNN",
            "Type": "Deep Learning",
            "Strengths": "Combines minutiae + DL",
            "Weaknesses": "Requires training, pretrained weights",
        },
    ]

    try:
        st.dataframe(
            methods,
            width='stretch',
            hide_index=True,
        )
    except TypeError:
        st.dataframe(
            methods,
            use_container_width=True,
            hide_index=True,
        )