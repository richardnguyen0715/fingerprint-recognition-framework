"""
Parameter configuration panel component.

This module provides components for configuring model parameters
based on the model's metadata.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import tempfile

import streamlit as st

from src.registry import (
    MatcherRegistry,
    ParameterInfo,
    ParameterType,
    get_registry,
)


def _render_model_path_input(
    param: ParameterInfo,
    current_value: str,
    key: str,
) -> str:
    """
    Render a special input widget for model_path parameter.
    
    Provides options to:
    1. Upload a new model file
    2. Select from available checkpoints
    3. Enter a custom path
    
    Args:
        param: Parameter metadata
        current_value: Current path value
        key: Widget key
        
    Returns:
        Selected model path
    """
    st.markdown(f"**{param.display_name}**")
    if param.description:
        st.caption(param.description)
    
    # Tab options for different input methods
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📁 Browse", "⌨️ Manual"])
    
    with tab1:
        st.caption("Upload a pretrained model (.pth file)")
        uploaded_file = st.file_uploader(
            label="Choose model file",
            type=["pth", "pt"],
            key=f"{key}_upload",
            label_visibility="collapsed",
        )
        
        if uploaded_file is not None:
            # Save to temporary location or models directory
            from src.utils.model_loader import get_models_dir
            
            # Determine model type from context (if possible)
            # For now, save to a temp location and return the path
            temp_dir = Path(tempfile.gettempdir()) / "fingerprint_models"
            temp_dir.mkdir(exist_ok=True)
            
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.caption(f"Path: {temp_path}")
            return str(temp_path)
    
    with tab2:
        st.caption("Select from available checkpoints")
        try:
            from src.utils.model_loader import list_available_models
            
            available = list_available_models()
            all_models = []
            for model_type, paths in available.items():
                for path in paths:
                    all_models.append((model_type, Path(path).name, path))
            
            if all_models:
                options = [f"{mt}: {name}" for mt, name, _ in all_models]
                paths = [path for _, _, path in all_models]
                
                selected_idx = st.selectbox(
                    label="Available models",
                    options=range(len(options)),
                    format_func=lambda i: options[i],
                    key=f"{key}_browse",
                    label_visibility="collapsed",
                )
                
                if selected_idx is not None:
                    st.caption(f"Path: {paths[selected_idx]}")
                    return paths[selected_idx]
            else:
                st.info("No checkpoints found in models/checkpoints/")
                st.caption("Train models or upload weights to use this feature.")
        except Exception as e:
            st.error(f"Error listing models: {e}")
    
    with tab3:
        st.caption("Enter model path manually")
        manual_path = st.text_input(
            label="Model path",
            value=current_value,
            key=f"{key}_manual",
            label_visibility="collapsed",
        )
        
        if manual_path:
            # Validate path
            if Path(manual_path).exists():
                st.success("✅ File exists")
            elif manual_path:
                st.warning("⚠️ File not found")
        
        return manual_path
    
    return current_value


def render_parameter_input(
    param: ParameterInfo,
    current_value: Optional[Any] = None,
    key_prefix: str = "",
) -> Any:
    """
    Render an input widget for a single parameter.
    
    Args:
        param: Parameter metadata
        current_value: Current value (uses default if None)
        key_prefix: Prefix for widget key
        
    Returns:
        The input value
    """
    key = f"{key_prefix}_{param.name}"
    value = current_value if current_value is not None else param.default
    
    # Render appropriate widget based on type
    if param.param_type == ParameterType.INTEGER:
        return st.number_input(
            label=param.display_name,
            value=int(value),
            min_value=int(param.min_value) if param.min_value is not None else None,
            max_value=int(param.max_value) if param.max_value is not None else None,
            step=int(param.step) if param.step is not None else 1,
            help=param.description,
            key=key,
        )
    
    elif param.param_type == ParameterType.FLOAT:
        return st.number_input(
            label=param.display_name,
            value=float(value),
            min_value=float(param.min_value) if param.min_value is not None else None,
            max_value=float(param.max_value) if param.max_value is not None else None,
            step=float(param.step) if param.step is not None else 0.1,
            format="%.4f",
            help=param.description,
            key=key,
        )
    
    elif param.param_type == ParameterType.STRING:
        # Special handling for model_path parameter
        if param.name == "model_path":
            return _render_model_path_input(param, value, key)
        else:
            return st.text_input(
                label=param.display_name,
                value=str(value),
                help=param.description,
                key=key,
            )
    
    elif param.param_type == ParameterType.BOOLEAN:
        return st.checkbox(
            label=param.display_name,
            value=bool(value),
            help=param.description,
            key=key,
        )
    
    elif param.param_type == ParameterType.SELECT:
        options = param.options or []
        index = options.index(value) if value in options else 0
        return st.selectbox(
            label=param.display_name,
            options=options,
            index=index,
            help=param.description,
            key=key,
        )
    
    else:
        st.warning(f"Unknown parameter type: {param.param_type}")
        return param.default


def render_config_panel(
    model_id: str,
    current_params: Optional[Dict[str, Any]] = None,
    key_prefix: str = "config",
    use_expander: bool = True,
    expanded: bool = False,
) -> Dict[str, Any]:
    """
    Render a configuration panel for model parameters.
    
    Args:
        model_id: ID of the model to configure
        current_params: Current parameter values
        key_prefix: Prefix for widget keys
        use_expander: Whether to wrap in an expander
        expanded: Whether expander is initially expanded
        
    Returns:
        Dictionary of parameter values
    """
    registry = get_registry()
    matcher_info = registry.get_matcher_info(model_id)
    
    if matcher_info is None:
        st.error(f"Model '{model_id}' not found.")
        return {}
    
    # No parameters to configure
    if not matcher_info.parameters:
        st.caption("This method has no configurable parameters.")
        return {}
    
    current_params = current_params or {}
    params = {}
    
    def render_params() -> None:
        """Render all parameter inputs."""
        nonlocal params
        
        st.caption(
            f"Configure parameters for **{matcher_info.name}**. "
            "Hover over each parameter for details."
        )
        
        # Group parameters in columns if there are multiple
        if len(matcher_info.parameters) <= 2:
            for param in matcher_info.parameters:
                value = render_parameter_input(
                    param=param,
                    current_value=current_params.get(param.name),
                    key_prefix=f"{key_prefix}_{model_id}",
                )
                params[param.name] = value
        else:
            # Use two columns for more parameters
            col1, col2 = st.columns(2)
            
            for idx, param in enumerate(matcher_info.parameters):
                with col1 if idx % 2 == 0 else col2:
                    value = render_parameter_input(
                        param=param,
                        current_value=current_params.get(param.name),
                        key_prefix=f"{key_prefix}_{model_id}",
                    )
                    params[param.name] = value
    
    # Render in expander or directly
    if use_expander:
        with st.expander("Configure Parameters", expanded=expanded):
            render_params()
    else:
        render_params()
    
    return params


def render_config_panel_compact(
    model_id: str,
    current_params: Optional[Dict[str, Any]] = None,
    key_prefix: str = "config_compact",
) -> Dict[str, Any]:
    """
    Render a compact configuration panel using sliders.
    
    Args:
        model_id: ID of the model to configure
        current_params: Current parameter values
        key_prefix: Prefix for widget keys
        
    Returns:
        Dictionary of parameter values
    """
    registry = get_registry()
    matcher_info = registry.get_matcher_info(model_id)
    
    if matcher_info is None or not matcher_info.parameters:
        return {}
    
    current_params = current_params or {}
    params = {}
    
    for param in matcher_info.parameters:
        key = f"{key_prefix}_{model_id}_{param.name}"
        value = current_params.get(param.name, param.default)
        
        if param.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if param.min_value is not None and param.max_value is not None:
                # Use slider for bounded numeric values
                if param.param_type == ParameterType.INTEGER:
                    params[param.name] = st.slider(
                        label=param.display_name,
                        min_value=int(param.min_value),
                        max_value=int(param.max_value),
                        value=int(value),
                        step=int(param.step) if param.step else 1,
                        help=param.description,
                        key=key,
                    )
                else:
                    params[param.name] = st.slider(
                        label=param.display_name,
                        min_value=float(param.min_value),
                        max_value=float(param.max_value),
                        value=float(value),
                        step=float(param.step) if param.step else 0.1,
                        help=param.description,
                        key=key,
                    )
            else:
                # Fall back to number input
                params[param.name] = render_parameter_input(
                    param, value, f"{key_prefix}_{model_id}"
                )
        else:
            params[param.name] = render_parameter_input(
                param, value, f"{key_prefix}_{model_id}"
            )
    
    return params


def render_reset_button(
    model_id: str,
    key: str = "reset_params",
) -> bool:
    """
    Render a button to reset parameters to defaults.
    
    Args:
        model_id: ID of the model
        key: Widget key
        
    Returns:
        True if button was clicked
    """
    registry = get_registry()
    matcher_info = registry.get_matcher_info(model_id)
    
    if matcher_info is None or not matcher_info.parameters:
        return False
    
    if st.button("Reset to Defaults", key=key):
        # Return default values
        return True
    
    return False


def get_default_parameters(model_id: str) -> Dict[str, Any]:
    """
    Get default parameter values for a model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Dictionary of default parameter values
    """
    registry = get_registry()
    matcher_info = registry.get_matcher_info(model_id)
    
    if matcher_info is None:
        return {}
    
    return {p.name: p.default for p in matcher_info.parameters}
