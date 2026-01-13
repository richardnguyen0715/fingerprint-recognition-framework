"""
Adapters for converting between UI and core framework formats.

This module provides functions to convert data between the Streamlit
UI layer and the core fingerprint recognition framework.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import io


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load an image from a Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Grayscale image as numpy array normalized to [0, 1], or None on error
    """
    if uploaded_file is None:
        return None
    
    try:
        # Read image data
        image_data = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for potential re-read
        
        # Open with PIL
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale if needed
        if pil_image.mode != "L":
            pil_image = pil_image.convert("L")
        
        # Convert to numpy array
        image = np.array(pil_image, dtype=np.float64)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
        
    except Exception:
        return None


def normalize_image(
    image: np.ndarray,
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Normalize an image to a target range.
    
    Args:
        image: Input image array
        target_range: (min, max) tuple for target range
        
    Returns:
        Normalized image
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max - img_min < 1e-10:
        # Constant image - return middle of target range
        return np.full_like(image, (target_range[0] + target_range[1]) / 2)
    
    # Normalize to [0, 1]
    normalized = (image - img_min) / (img_max - img_min)
    
    # Scale to target range
    target_min, target_max = target_range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled


def prepare_image_for_display(
    image: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Prepare an image for display in Streamlit.
    
    Args:
        image: Input image array (expected [0, 1])
        size: Optional (width, height) to resize to
        
    Returns:
        Image array ready for st.image()
    """
    # Ensure [0, 1] range
    display_image = np.clip(image, 0.0, 1.0)
    
    # Convert to uint8 for display
    display_image = (display_image * 255).astype(np.uint8)
    
    # Resize if requested
    if size is not None:
        pil_image = Image.fromarray(display_image)
        pil_image = pil_image.resize(size, Image.Resampling.LANCZOS)
        display_image = np.array(pil_image)
    
    return display_image


def convert_result_for_display(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a match result for display in the UI.
    
    Handles conversion of numpy types to Python native types
    for JSON serialization and display.
    
    Args:
        result: Raw result dictionary
        
    Returns:
        Display-ready result dictionary
    """
    def convert_value(value: Any) -> Any:
        """Convert a single value to a display-friendly type."""
        if isinstance(value, np.ndarray):
            # For small arrays, convert to list
            if value.size <= 100:
                return value.tolist()
            else:
                # For large arrays, just return shape info
                return f"Array shape: {value.shape}"
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        else:
            return value
    
    return convert_value(result)


def format_score(score: float, precision: int = 4) -> str:
    """
    Format a similarity score for display.
    
    Args:
        score: Similarity score (typically [0, 1])
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{score:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Value to format (0.0 to 1.0)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def get_score_color(score: float) -> str:
    """
    Get a color for displaying a score.
    
    Args:
        score: Similarity score [0, 1]
        
    Returns:
        CSS color string
    """
    if score >= 0.7:
        return "#28a745"  # Green - high similarity
    elif score >= 0.4:
        return "#ffc107"  # Yellow - medium similarity
    else:
        return "#dc3545"  # Red - low similarity


def create_score_badge(score: float) -> str:
    """
    Create an HTML badge for a score.
    
    Args:
        score: Similarity score [0, 1]
        
    Returns:
        HTML string for the badge
    """
    color = get_score_color(score)
    return f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 1.2em;
    ">
        {format_score(score)}
    </span>
    """
