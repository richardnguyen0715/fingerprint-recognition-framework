"""
Session state management for the Streamlit UI.

This module handles initialization and management of Streamlit's
session state, ensuring consistent state across page navigations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st


# =============================================================================
# STATE KEYS
# =============================================================================


class StateKeys:
    """Constants for session state keys."""
    
    # Images
    IMAGE_A = "image_a"
    IMAGE_B = "image_b"
    IMAGE_A_NAME = "image_a_name"
    IMAGE_B_NAME = "image_b_name"
    
    # Model selection
    SELECTED_MODEL = "selected_model"
    MODEL_PARAMETERS = "model_parameters"
    
    # Results
    LAST_RESULT = "last_result"
    RESULT_HISTORY = "result_history"
    
    # Comparison mode
    COMPARISON_MODELS = "comparison_models"
    COMPARISON_RESULTS = "comparison_results"
    
    # UI state
    INITIALIZED = "initialized"


# =============================================================================
# STATE INITIALIZATION
# =============================================================================


def initialize_session_state() -> None:
    """
    Initialize all session state variables.
    
    This function should be called at the start of the application
    to ensure all state variables have default values.
    """
    if StateKeys.INITIALIZED in st.session_state:
        return
    
    # Initialize image storage
    if StateKeys.IMAGE_A not in st.session_state:
        st.session_state[StateKeys.IMAGE_A] = None
    if StateKeys.IMAGE_B not in st.session_state:
        st.session_state[StateKeys.IMAGE_B] = None
    if StateKeys.IMAGE_A_NAME not in st.session_state:
        st.session_state[StateKeys.IMAGE_A_NAME] = None
    if StateKeys.IMAGE_B_NAME not in st.session_state:
        st.session_state[StateKeys.IMAGE_B_NAME] = None
    
    # Initialize model selection
    if StateKeys.SELECTED_MODEL not in st.session_state:
        st.session_state[StateKeys.SELECTED_MODEL] = None
    if StateKeys.MODEL_PARAMETERS not in st.session_state:
        st.session_state[StateKeys.MODEL_PARAMETERS] = {}
    
    # Initialize results
    if StateKeys.LAST_RESULT not in st.session_state:
        st.session_state[StateKeys.LAST_RESULT] = None
    if StateKeys.RESULT_HISTORY not in st.session_state:
        st.session_state[StateKeys.RESULT_HISTORY] = []
    
    # Initialize comparison mode
    if StateKeys.COMPARISON_MODELS not in st.session_state:
        st.session_state[StateKeys.COMPARISON_MODELS] = []
    if StateKeys.COMPARISON_RESULTS not in st.session_state:
        st.session_state[StateKeys.COMPARISON_RESULTS] = {}
    
    # Mark as initialized
    st.session_state[StateKeys.INITIALIZED] = True


# =============================================================================
# STATE ACCESSORS
# =============================================================================


def get_image_a() -> Optional[np.ndarray]:
    """Get the first uploaded image."""
    return st.session_state.get(StateKeys.IMAGE_A)


def get_image_b() -> Optional[np.ndarray]:
    """Get the second uploaded image."""
    return st.session_state.get(StateKeys.IMAGE_B)


def get_image_a_name() -> Optional[str]:
    """Get the filename of the first image."""
    return st.session_state.get(StateKeys.IMAGE_A_NAME)


def get_image_b_name() -> Optional[str]:
    """Get the filename of the second image."""
    return st.session_state.get(StateKeys.IMAGE_B_NAME)


def set_image_a(image: Optional[np.ndarray], name: Optional[str] = None) -> None:
    """
    Set the first image.
    
    Args:
        image: Image array or None to clear
        name: Optional filename
    """
    st.session_state[StateKeys.IMAGE_A] = image
    st.session_state[StateKeys.IMAGE_A_NAME] = name
    # Clear previous results when images change
    clear_results()


def set_image_b(image: Optional[np.ndarray], name: Optional[str] = None) -> None:
    """
    Set the second image.
    
    Args:
        image: Image array or None to clear
        name: Optional filename
    """
    st.session_state[StateKeys.IMAGE_B] = image
    st.session_state[StateKeys.IMAGE_B_NAME] = name
    # Clear previous results when images change
    clear_results()


def get_selected_model() -> Optional[str]:
    """Get the currently selected model ID."""
    return st.session_state.get(StateKeys.SELECTED_MODEL)


def set_selected_model(model_id: Optional[str]) -> None:
    """
    Set the selected model.
    
    Args:
        model_id: Model identifier or None to clear
    """
    st.session_state[StateKeys.SELECTED_MODEL] = model_id


def get_model_parameters(model_id: str) -> Dict[str, Any]:
    """
    Get parameters for a specific model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Dictionary of parameter values
    """
    params = st.session_state.get(StateKeys.MODEL_PARAMETERS, {})
    return params.get(model_id, {})


def set_model_parameters(model_id: str, parameters: Dict[str, Any]) -> None:
    """
    Set parameters for a specific model.
    
    Args:
        model_id: Model identifier
        parameters: Dictionary of parameter values
    """
    if StateKeys.MODEL_PARAMETERS not in st.session_state:
        st.session_state[StateKeys.MODEL_PARAMETERS] = {}
    st.session_state[StateKeys.MODEL_PARAMETERS][model_id] = parameters


def get_last_result() -> Optional[Dict[str, Any]]:
    """Get the most recent matching result."""
    return st.session_state.get(StateKeys.LAST_RESULT)


def set_last_result(result: Optional[Dict[str, Any]]) -> None:
    """
    Set the most recent matching result.
    
    Args:
        result: Result dictionary or None to clear
    """
    st.session_state[StateKeys.LAST_RESULT] = result
    
    # Add to history if not None
    if result is not None:
        add_to_result_history(result)


def get_result_history() -> List[Dict[str, Any]]:
    """Get the history of matching results."""
    return st.session_state.get(StateKeys.RESULT_HISTORY, [])


def add_to_result_history(result: Dict[str, Any]) -> None:
    """
    Add a result to the history.
    
    Args:
        result: Result dictionary to add
    """
    history = st.session_state.get(StateKeys.RESULT_HISTORY, [])
    history.append(result)
    
    # Keep only last 50 results
    if len(history) > 50:
        history = history[-50:]
    
    st.session_state[StateKeys.RESULT_HISTORY] = history


def clear_results() -> None:
    """Clear all matching results."""
    st.session_state[StateKeys.LAST_RESULT] = None
    st.session_state[StateKeys.COMPARISON_RESULTS] = {}


def clear_result_history() -> None:
    """Clear the result history."""
    st.session_state[StateKeys.RESULT_HISTORY] = []


# =============================================================================
# COMPARISON MODE
# =============================================================================


def get_comparison_models() -> List[str]:
    """Get the list of models selected for comparison."""
    return st.session_state.get(StateKeys.COMPARISON_MODELS, [])


def set_comparison_models(model_ids: List[str]) -> None:
    """
    Set the models for comparison.
    
    Args:
        model_ids: List of model identifiers
    """
    st.session_state[StateKeys.COMPARISON_MODELS] = model_ids


def get_comparison_results() -> Dict[str, Dict[str, Any]]:
    """Get comparison results for all selected models."""
    return st.session_state.get(StateKeys.COMPARISON_RESULTS, {})


def set_comparison_result(model_id: str, result: Dict[str, Any]) -> None:
    """
    Set comparison result for a specific model.
    
    Args:
        model_id: Model identifier
        result: Result dictionary
    """
    if StateKeys.COMPARISON_RESULTS not in st.session_state:
        st.session_state[StateKeys.COMPARISON_RESULTS] = {}
    st.session_state[StateKeys.COMPARISON_RESULTS][model_id] = result


def clear_comparison_results() -> None:
    """Clear all comparison results."""
    st.session_state[StateKeys.COMPARISON_RESULTS] = {}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def has_images() -> bool:
    """Check if both images are loaded."""
    return get_image_a() is not None and get_image_b() is not None


def reset_all() -> None:
    """Reset all session state to defaults."""
    st.session_state[StateKeys.IMAGE_A] = None
    st.session_state[StateKeys.IMAGE_B] = None
    st.session_state[StateKeys.IMAGE_A_NAME] = None
    st.session_state[StateKeys.IMAGE_B_NAME] = None
    st.session_state[StateKeys.SELECTED_MODEL] = None
    st.session_state[StateKeys.MODEL_PARAMETERS] = {}
    st.session_state[StateKeys.LAST_RESULT] = None
    st.session_state[StateKeys.RESULT_HISTORY] = []
    st.session_state[StateKeys.COMPARISON_MODELS] = []
    st.session_state[StateKeys.COMPARISON_RESULTS] = {}
