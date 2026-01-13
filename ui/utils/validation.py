"""
Input validation utilities for the Streamlit UI.

This module provides validation functions for user inputs,
ensuring data quality before passing to the core framework.
"""

from typing import Optional, Tuple

import numpy as np


class ImageValidationError(Exception):
    """Exception raised for image validation errors."""
    pass


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

MIN_IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 2048
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


# =============================================================================
# IMAGE VALIDATION
# =============================================================================


def validate_image(
    image: Optional[np.ndarray],
    name: str = "Image",
) -> Tuple[bool, str]:
    """
    Validate a single fingerprint image.
    
    Args:
        image: Image array to validate
        name: Name for error messages
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check if image exists
    if image is None:
        return False, f"{name} is not loaded"
    
    # Check if numpy array
    if not isinstance(image, np.ndarray):
        return False, f"{name} must be a numpy array"
    
    # Check dimensions
    if image.ndim != 2:
        return False, f"{name} must be a 2D grayscale image (got {image.ndim}D)"
    
    # Check size
    height, width = image.shape
    
    if height < MIN_IMAGE_SIZE or width < MIN_IMAGE_SIZE:
        return (
            False,
            f"{name} is too small ({width}x{height}). "
            f"Minimum size is {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}",
        )
    
    if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
        return (
            False,
            f"{name} is too large ({width}x{height}). "
            f"Maximum size is {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}",
        )
    
    # Check for valid pixel values
    if not np.isfinite(image).all():
        return False, f"{name} contains invalid pixel values (NaN or Inf)"
    
    # Check range (should be [0, 1] after normalization)
    if image.min() < -0.1 or image.max() > 1.1:
        return (
            False,
            f"{name} has unexpected value range [{image.min():.2f}, {image.max():.2f}]. "
            "Expected [0, 1]",
        )
    
    # Check for constant image (no features)
    if image.std() < 0.01:
        return (
            False,
            f"{name} appears to be a constant image with no features",
        )
    
    return True, "Valid"


def validate_image_pair(
    image_a: Optional[np.ndarray],
    image_b: Optional[np.ndarray],
    require_same_size: bool = False,
) -> Tuple[bool, str]:
    """
    Validate a pair of fingerprint images for matching.
    
    Args:
        image_a: First image
        image_b: Second image
        require_same_size: Whether images must have same dimensions
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Validate individual images
    valid_a, msg_a = validate_image(image_a, "Fingerprint A")
    if not valid_a:
        return False, msg_a
    
    valid_b, msg_b = validate_image(image_b, "Fingerprint B")
    if not valid_b:
        return False, msg_b
    
    # Check size match if required
    if require_same_size:
        if image_a.shape != image_b.shape:
            return (
                False,
                f"Images must have same dimensions: "
                f"A is {image_a.shape}, B is {image_b.shape}",
            )
    
    return True, "Both images are valid"


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """
    Validate that a file has an allowed extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not filename:
        return False, "No filename provided"
    
    # Get extension
    parts = filename.rsplit(".", 1)
    if len(parts) < 2:
        return False, "File has no extension"
    
    extension = parts[1].lower()
    
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return (
            False,
            f"Invalid file extension '.{extension}'. Allowed: {allowed}",
        )
    
    return True, "Valid extension"


def validate_parameter_value(
    value: any,
    param_type: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    options: Optional[list] = None,
) -> Tuple[bool, str]:
    """
    Validate a parameter value.
    
    Args:
        value: Value to validate
        param_type: Expected type ("integer", "float", "string", "boolean", "select")
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)
        options: Allowed options (for select type)
        
    Returns:
        Tuple of (is_valid, message)
    """
    if param_type == "integer":
        if not isinstance(value, (int, np.integer)):
            return False, f"Expected integer, got {type(value).__name__}"
        
        if min_value is not None and value < min_value:
            return False, f"Value {value} is below minimum {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Value {value} is above maximum {max_value}"
    
    elif param_type == "float":
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return False, f"Expected number, got {type(value).__name__}"
        
        if min_value is not None and value < min_value:
            return False, f"Value {value} is below minimum {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Value {value} is above maximum {max_value}"
    
    elif param_type == "string":
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
    
    elif param_type == "boolean":
        if not isinstance(value, bool):
            return False, f"Expected boolean, got {type(value).__name__}"
    
    elif param_type == "select":
        if options is None:
            return False, "No options provided for select parameter"
        
        if value not in options:
            return False, f"Value '{value}' not in allowed options: {options}"
    
    else:
        return False, f"Unknown parameter type: {param_type}"
    
    return True, "Valid"
