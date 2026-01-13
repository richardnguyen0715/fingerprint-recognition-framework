"""
Orientation field estimation for fingerprint images.

This module implements gradient-based orientation field estimation,
which is a fundamental step in fingerprint enhancement and feature
extraction.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Orientation Field Estimation:
# ----------------------------
# The orientation field θ(x, y) represents the local ridge direction at
# each pixel. It is estimated using gradient-based methods.
#
# Algorithm (Gradient-Based):
# 1. Compute image gradients: Gx = ∂I/∂x, Gy = ∂I/∂y
# 2. For each block of size w x w, compute:
#    - Vx = Σ 2 * Gx * Gy
#    - Vy = Σ (Gx² - Gy²)
# 3. Orientation: θ = 0.5 * atan2(Vx, Vy)
#
# The factor of 0.5 accounts for the 180° ambiguity in ridge orientation
# (ridges have the same appearance at θ and θ + 180°).
#
# Reference:
# Ratha, N. K., Chen, S., & Jain, A. K. (1995).
# "Adaptive flow orientation-based feature extraction in fingerprint images."
# Pattern Recognition, 28(11), 1657-1672.
# =============================================================================


def compute_gradients(
    image: np.ndarray,
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute image gradients using Gaussian derivative filters.
    
    Mathematical Formulation:
    -------------------------
    Gx = I * ∂G/∂x
    Gy = I * ∂G/∂y
    
    where G is a Gaussian kernel and * denotes convolution.
    
    Args:
        image: Input grayscale image
        sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Tuple of (Gx, Gy) gradient arrays
    """
    # Gaussian derivative in x and y directions
    gx = ndimage.gaussian_filter1d(image, sigma, axis=1, order=1)
    gy = ndimage.gaussian_filter1d(image, sigma, axis=0, order=1)
    
    return gx, gy


def estimate_orientation_block(
    gx: np.ndarray,
    gy: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """
    Estimate orientation field using block-based gradient method.
    
    Mathematical Formulation:
    -------------------------
    For each block B:
    
    Vx = Σ_{(i,j)∈B} 2 * Gx(i,j) * Gy(i,j)
    Vy = Σ_{(i,j)∈B} (Gx(i,j)² - Gy(i,j)²)
    
    θ = 0.5 * atan2(Vx, Vy)
    
    The result is the orientation perpendicular to the gradient,
    which corresponds to the ridge direction.
    
    Args:
        gx: Gradient in x direction
        gy: Gradient in y direction
        block_size: Size of blocks for averaging
        
    Returns:
        Orientation field (in radians, range [-π/2, π/2])
    """
    h, w = gx.shape
    
    # Number of blocks
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    # Output orientation field
    orientation = np.zeros((num_blocks_y, num_blocks_x))
    
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            # Block coordinates
            y_start = by * block_size
            y_end = y_start + block_size
            x_start = bx * block_size
            x_end = x_start + block_size
            
            # Extract block gradients
            block_gx = gx[y_start:y_end, x_start:x_end]
            block_gy = gy[y_start:y_end, x_start:x_end]
            
            # Compute orientation tensor components
            vx = 2 * np.sum(block_gx * block_gy)
            vy = np.sum(block_gx ** 2 - block_gy ** 2)
            
            # Compute orientation (ridge direction)
            orientation[by, bx] = 0.5 * np.arctan2(vx, vy)
    
    return orientation


def smooth_orientation_field(
    orientation: np.ndarray,
    sigma: float = 3.0
) -> np.ndarray:
    """
    Smooth orientation field using Gaussian filtering in doubled-angle domain.
    
    Mathematical Formulation:
    -------------------------
    Because orientation has π periodicity, direct smoothing doesn't work.
    Instead, we convert to Cartesian coordinates:
    
    cos_2θ = cos(2θ)
    sin_2θ = sin(2θ)
    
    Smooth these independently, then convert back:
    θ_smoothed = 0.5 * atan2(sin_2θ_smoothed, cos_2θ_smoothed)
    
    Args:
        orientation: Input orientation field (in radians)
        sigma: Gaussian smoothing sigma
        
    Returns:
        Smoothed orientation field
    """
    # Convert to doubled-angle representation
    cos_2theta = np.cos(2 * orientation)
    sin_2theta = np.sin(2 * orientation)
    
    # Smooth in Cartesian domain
    cos_2theta_smooth = ndimage.gaussian_filter(cos_2theta, sigma)
    sin_2theta_smooth = ndimage.gaussian_filter(sin_2theta, sigma)
    
    # Convert back to orientation
    orientation_smooth = 0.5 * np.arctan2(sin_2theta_smooth, cos_2theta_smooth)
    
    return orientation_smooth


def resize_orientation_field(
    orientation: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize orientation field to match image dimensions.
    
    Uses doubled-angle interpolation for correct handling of
    orientation discontinuities.
    
    Args:
        orientation: Block-wise orientation field
        target_shape: Target (height, width)
        
    Returns:
        Pixel-wise orientation field
    """
    from scipy.ndimage import zoom
    
    # Convert to doubled-angle
    cos_2theta = np.cos(2 * orientation)
    sin_2theta = np.sin(2 * orientation)
    
    # Compute zoom factors
    zoom_y = target_shape[0] / orientation.shape[0]
    zoom_x = target_shape[1] / orientation.shape[1]
    
    # Interpolate
    cos_2theta_resized = zoom(cos_2theta, (zoom_y, zoom_x), order=3)
    sin_2theta_resized = zoom(sin_2theta, (zoom_y, zoom_x), order=3)
    
    # Convert back
    orientation_resized = 0.5 * np.arctan2(sin_2theta_resized, cos_2theta_resized)
    
    return orientation_resized


def estimate_orientation_field(
    image: np.ndarray,
    block_size: int = 16,
    gradient_sigma: float = 1.0,
    smooth_sigma: float = 3.0,
    resize_to_image: bool = True
) -> np.ndarray:
    """
    Complete orientation field estimation pipeline.
    
    Algorithm Steps:
    ----------------
    1. Compute image gradients using Gaussian derivatives
    2. Estimate block-wise orientation using gradient tensor
    3. Smooth orientation field in doubled-angle domain
    4. Optionally resize to match input image dimensions
    
    Args:
        image: Input fingerprint image
        block_size: Block size for orientation estimation
        gradient_sigma: Sigma for gradient computation
        smooth_sigma: Sigma for orientation smoothing
        resize_to_image: Whether to resize output to image dimensions
        
    Returns:
        Orientation field (radians, range [-π/2, π/2])
    """
    # Ensure float image
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Step 1: Compute gradients
    gx, gy = compute_gradients(image, gradient_sigma)
    
    # Step 2: Block-wise orientation estimation
    orientation = estimate_orientation_block(gx, gy, block_size)
    
    # Step 3: Smooth orientation field
    orientation = smooth_orientation_field(orientation, smooth_sigma)
    
    # Step 4: Resize if requested
    if resize_to_image:
        orientation = resize_orientation_field(orientation, image.shape)
    
    return orientation


def compute_coherence(
    gx: np.ndarray,
    gy: np.ndarray,
    block_size: int = 16
) -> np.ndarray:
    """
    Compute orientation coherence (reliability) map.
    
    Mathematical Formulation:
    -------------------------
    Coherence measures how consistently oriented the gradients are:
    
    E = Σ_{(i,j)∈B} ||∇I(i,j)||
    coherence = sqrt(Vx² + Vy²) / E
    
    High coherence indicates reliable orientation estimate.
    Low coherence indicates noisy region or minutiae.
    
    Args:
        gx: Gradient in x direction
        gy: Gradient in y direction
        block_size: Block size for coherence computation
        
    Returns:
        Coherence map (0 = unreliable, 1 = reliable)
    """
    h, w = gx.shape
    
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    coherence = np.zeros((num_blocks_y, num_blocks_x))
    
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * block_size
            y_end = y_start + block_size
            x_start = bx * block_size
            x_end = x_start + block_size
            
            block_gx = gx[y_start:y_end, x_start:x_end]
            block_gy = gy[y_start:y_end, x_start:x_end]
            
            # Gradient magnitude sum
            grad_sum = np.sum(np.sqrt(block_gx ** 2 + block_gy ** 2))
            
            if grad_sum < 1e-10:
                coherence[by, bx] = 0
                continue
            
            # Orientation tensor components
            vx = 2 * np.sum(block_gx * block_gy)
            vy = np.sum(block_gx ** 2 - block_gy ** 2)
            
            # Coherence
            coherence[by, bx] = np.sqrt(vx ** 2 + vy ** 2) / grad_sum
    
    return coherence


class OrientationFieldEstimator:
    """
    Configurable orientation field estimator.
    
    Encapsulates parameters and provides caching for repeated estimation.
    """
    
    def __init__(
        self,
        block_size: int = 16,
        gradient_sigma: float = 1.0,
        smooth_sigma: float = 3.0
    ):
        """
        Initialize estimator.
        
        Args:
            block_size: Block size for estimation
            gradient_sigma: Sigma for gradient computation
            smooth_sigma: Sigma for orientation smoothing
        """
        self.block_size = block_size
        self.gradient_sigma = gradient_sigma
        self.smooth_sigma = smooth_sigma
    
    def estimate(
        self,
        image: np.ndarray,
        compute_coherence_map: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate orientation field.
        
        Args:
            image: Input fingerprint image
            compute_coherence_map: Whether to also compute coherence
            
        Returns:
            Tuple of (orientation_field, coherence_map)
            coherence_map is None if compute_coherence_map=False
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        gx, gy = compute_gradients(image, self.gradient_sigma)
        
        orientation = estimate_orientation_block(gx, gy, self.block_size)
        orientation = smooth_orientation_field(orientation, self.smooth_sigma)
        orientation = resize_orientation_field(orientation, image.shape)
        
        coherence_map = None
        if compute_coherence_map:
            coherence_block = compute_coherence(gx, gy, self.block_size)
            coherence_map = resize_orientation_field(coherence_block, image.shape)
        
        return orientation, coherence_map
