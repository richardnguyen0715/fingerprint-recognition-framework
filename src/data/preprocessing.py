"""
Image preprocessing utilities for fingerprint recognition.

This module provides functions for preprocessing fingerprint images
including normalization, segmentation, and quality enhancement.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Fingerprint preprocessing aims to enhance ridge-valley contrast and
# normalize image properties for consistent feature extraction.
#
# Key operations:
# 1. Normalization: Transform pixel intensities to standard mean/variance
# 2. Segmentation: Separate fingerprint region from background
# 3. Histogram equalization: Enhance local contrast
# =============================================================================


def normalize_image(
    image: np.ndarray,
    target_mean: float = 0.0,
    target_std: float = 1.0
) -> np.ndarray:
    """
    Normalize image to have specified mean and standard deviation.
    
    Mathematical Formulation:
    -------------------------
    Given image I with mean μ and std σ:
    
    I_normalized = (I - μ) / σ * σ_target + μ_target
    
    This ensures consistent intensity distribution across images.
    
    Args:
        image: Input image (grayscale, float or uint8)
        target_mean: Desired mean value
        target_std: Desired standard deviation
        
    Returns:
        Normalized image as float32
    """
    image = image.astype(np.float32)
    
    current_mean = np.mean(image)
    current_std = np.std(image)
    
    if current_std < 1e-10:
        return np.full_like(image, target_mean)
    
    normalized = (image - current_mean) / current_std
    normalized = normalized * target_std + target_mean
    
    return normalized


def normalize_to_range(
    image: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize image to specified range.
    
    Args:
        image: Input image
        min_val: Minimum output value
        max_val: Maximum output value
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32)
    
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max - img_min < 1e-10:
        return np.full_like(image, (min_val + max_val) / 2)
    
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


def segment_fingerprint(
    image: np.ndarray,
    block_size: int = 16,
    variance_threshold: float = 0.01
) -> np.ndarray:
    """
    Segment fingerprint region from background using local variance.
    
    Mathematical Formulation:
    -------------------------
    For each block B of size w x w:
    
    variance(B) = (1/w²) * Σ(I(x,y) - μ_B)²
    
    Blocks with variance > threshold are considered fingerprint region.
    
    Algorithm:
    1. Divide image into non-overlapping blocks
    2. Compute variance of each block
    3. Threshold variance to create mask
    4. Apply morphological operations to clean mask
    
    Args:
        image: Input grayscale image (normalized to [0, 1])
        block_size: Size of blocks for variance computation
        variance_threshold: Threshold for foreground detection
        
    Returns:
        Binary mask (1 = fingerprint, 0 = background)
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Compute block-wise variance
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            variance = np.var(block)
            
            if variance > variance_threshold:
                mask[y:y+block_size, x:x+block_size] = 1
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size, block_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def adaptive_histogram_equalization(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Mathematical Formulation:
    -------------------------
    CLAHE divides image into tiles and equalizes each separately:
    
    1. For each tile, compute histogram H(k)
    2. Clip histogram: H'(k) = min(H(k), clip_limit * N / num_bins)
    3. Redistribute clipped pixels uniformly
    4. Compute CDF and apply equalization
    5. Interpolate at tile boundaries
    
    This enhances local contrast without amplifying noise.
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of tiles for local equalization
        
    Returns:
        Enhanced image
    """
    # Convert to uint8 if necessary
    if image.dtype in [np.float32, np.float64]:
        was_float = True
        image = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        was_float = False
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(image)
    
    if was_float:
        enhanced = enhanced.astype(np.float32) / 255.0
    
    return enhanced


def remove_background_gradient(
    image: np.ndarray,
    kernel_size: int = 65
) -> np.ndarray:
    """
    Remove background intensity gradient using morphological operations.
    
    Mathematical Formulation:
    -------------------------
    Background estimation via morphological opening:
    
    background = opening(I, kernel)
    corrected = I - background
    
    This removes slow intensity variations while preserving ridges.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of structuring element (odd number)
        
    Returns:
        Gradient-corrected image
    """
    if image.dtype in [np.float32, np.float64]:
        was_float = True
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        was_float = False
        img_uint8 = image
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
    
    corrected = cv2.subtract(img_uint8, background)
    
    if was_float:
        corrected = corrected.astype(np.float32) / 255.0
    
    return corrected


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def pad_to_square(
    image: np.ndarray,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad image to make it square.
    
    Args:
        image: Input image
        pad_value: Value for padding
        
    Returns:
        Square image
    """
    h, w = image.shape[:2]
    size = max(h, w)
    
    if len(image.shape) == 3:
        padded = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        padded = np.full((size, size), pad_value, dtype=image.dtype)
    
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded


def preprocess_fingerprint(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    normalize: bool = True,
    equalize: bool = True,
    segment: bool = True,
    remove_gradient: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply full preprocessing pipeline to fingerprint image.
    
    Pipeline:
    1. Convert to float and normalize range
    2. Remove background gradient (optional)
    3. Apply CLAHE for contrast enhancement (optional)
    4. Segment fingerprint region (optional)
    5. Normalize intensity (optional)
    6. Resize to target size
    
    Args:
        image: Input fingerprint image
        target_size: Target (width, height), None to keep original
        normalize: Whether to normalize intensities
        equalize: Whether to apply histogram equalization
        segment: Whether to compute segmentation mask
        remove_gradient: Whether to remove background gradient
        
    Returns:
        Tuple of (preprocessed_image, segmentation_mask)
        mask is None if segment=False
    """
    # Ensure float image
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    mask = None
    
    # Remove background gradient
    if remove_gradient:
        image = remove_background_gradient(image)
    
    # Histogram equalization
    if equalize:
        image = adaptive_histogram_equalization(image)
    
    # Segmentation
    if segment:
        mask = segment_fingerprint(image)
    
    # Normalize intensities
    if normalize:
        image = normalize_to_range(image, 0.0, 1.0)
    
    # Resize
    if target_size is not None:
        image = resize_image(image, target_size)
        if mask is not None:
            mask = resize_image(mask, target_size, cv2.INTER_NEAREST)
    
    return image, mask


class FingerprintPreprocessor:
    """
    Configurable preprocessor for fingerprint images.
    
    Encapsulates preprocessing parameters and provides a consistent
    interface for batch processing.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        equalize: bool = True,
        segment: bool = True,
        remove_gradient: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Output image size
            normalize: Whether to normalize intensities
            equalize: Whether to apply CLAHE
            segment: Whether to compute segmentation mask
            remove_gradient: Whether to remove background gradient
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile size
        """
        self.target_size = target_size
        self.normalize = normalize
        self.equalize = equalize
        self.segment = segment
        self.remove_gradient = remove_gradient
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def __call__(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess a fingerprint image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (preprocessed_image, mask)
        """
        return preprocess_fingerprint(
            image,
            target_size=self.target_size,
            normalize=self.normalize,
            equalize=self.equalize,
            segment=self.segment,
            remove_gradient=self.remove_gradient
        )
    
    def process_batch(
        self,
        images: list
    ) -> Tuple[list, list]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (processed_images, masks)
        """
        processed = []
        masks = []
        
        for img in images:
            proc_img, mask = self(img)
            processed.append(proc_img)
            masks.append(mask)
        
        return processed, masks
