"""
Fingerprint image upload component.

This module provides reusable components for uploading fingerprint
images in the Streamlit UI.
"""

from typing import Optional, Tuple

import numpy as np
import streamlit as st

from ui.utils.adapters import load_image_from_upload, prepare_image_for_display
from ui.utils.validation import validate_image, validate_file_extension


def render_image_uploader(
    label: str,
    key: str,
    help_text: Optional[str] = None,
    show_preview: bool = True,
    preview_size: Tuple[int, int] = (200, 200),
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Render a fingerprint image uploader widget.
    
    Args:
        label: Label for the uploader
        key: Unique key for Streamlit widget state
        help_text: Optional help text to display
        show_preview: Whether to show image preview
        preview_size: Size (width, height) for preview
        
    Returns:
        Tuple of (image_array, filename) or (None, None) if no upload
    """
    # File uploader
    uploaded_file = st.file_uploader(
        label=label,
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key=key,
        help=help_text or "Upload a fingerprint image (PNG, JPG, BMP, or TIFF)",
    )
    
    if uploaded_file is None:
        return None, None
    
    # Validate file extension
    valid_ext, ext_msg = validate_file_extension(uploaded_file.name)
    if not valid_ext:
        st.error(ext_msg)
        return None, None
    
    # Load and process image
    image = load_image_from_upload(uploaded_file)
    
    if image is None:
        st.error("Failed to load image. Please try a different file.")
        return None, None
    
    # Validate image
    valid_img, img_msg = validate_image(image, "Uploaded image")
    if not valid_img:
        st.error(img_msg)
        return None, None
    
    # Show preview
    if show_preview:
        display_image = prepare_image_for_display(image, preview_size)
        st.image(
            display_image,
            caption=uploaded_file.name,
            width='content',
        )
        
        # Display image info
        st.caption(
            f"Size: {image.shape[1]}×{image.shape[0]} | "
            f"Range: [{image.min():.3f}, {image.max():.3f}]"
        )
    
    return image, uploaded_file.name


def render_dual_uploader(
    show_preview: bool = True,
    preview_size: Tuple[int, int] = (200, 200),
) -> Tuple[
    Optional[np.ndarray],
    Optional[str],
    Optional[np.ndarray],
    Optional[str],
]:
    """
    Render dual fingerprint uploaders side by side.
    
    Args:
        show_preview: Whether to show image previews
        preview_size: Size (width, height) for previews
        
    Returns:
        Tuple of (image_a, name_a, image_b, name_b)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fingerprint A")
        image_a, name_a = render_image_uploader(
            label="Upload first fingerprint",
            key="uploader_a",
            help_text="Upload the query fingerprint",
            show_preview=show_preview,
            preview_size=preview_size,
        )
    
    with col2:
        st.subheader("Fingerprint B")
        image_b, name_b = render_image_uploader(
            label="Upload second fingerprint",
            key="uploader_b",
            help_text="Upload the reference fingerprint",
            show_preview=show_preview,
            preview_size=preview_size,
        )
    
    return image_a, name_a, image_b, name_b


def render_image_preview(
    image: np.ndarray,
    title: str,
    caption: Optional[str] = None,
    show_stats: bool = True,
) -> None:
    """
    Render an image preview with optional statistics.
    
    Args:
        image: Image array to display
        title: Title for the preview
        caption: Optional caption
        show_stats: Whether to show image statistics
    """
    st.markdown(f"**{title}**")
    
    display_image = prepare_image_for_display(image)
    st.image(
        display_image,
        caption=caption,
        width='stretch',
    )
    
    if show_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Width", image.shape[1])
        with col2:
            st.metric("Height", image.shape[0])
        with col3:
            st.metric("Std Dev", f"{image.std():.3f}")
