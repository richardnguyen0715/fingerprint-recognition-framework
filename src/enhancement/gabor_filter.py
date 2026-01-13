"""
Gabor filter-based fingerprint enhancement.

This module implements Gabor filtering for fingerprint image enhancement,
using locally tuned filters based on orientation and frequency fields.
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Gabor Filter:
# -------------
# A Gabor filter is a sinusoidal wave modulated by a Gaussian envelope.
# It's optimal for detecting oriented features at a specific frequency.
#
# 2D Gabor kernel:
# g(x, y; θ, f, σx, σy) = exp(-x'²/2σx² - y'²/2σy²) * cos(2πf*x')
#
# where:
#   x' = x*cos(θ) + y*sin(θ)   (coordinate along filter orientation)
#   y' = -x*sin(θ) + y*cos(θ)  (coordinate perpendicular to orientation)
#   θ = filter orientation
#   f = frequency (1/wavelength)
#   σx, σy = Gaussian envelope standard deviations
#
# For fingerprint enhancement:
# - θ matches local ridge orientation
# - f matches local ridge frequency
# - σx ≈ 0.5/f (along ridge direction)
# - σy ≈ 0.5/f (perpendicular to ridges)
#
# Reference:
# Hong, L., Wan, Y., & Jain, A. (1998).
# "Fingerprint image enhancement: algorithm and performance evaluation."
# IEEE TPAMI, 20(8), 777-789.
# =============================================================================


def create_gabor_kernel(
    orientation: float,
    frequency: float,
    sigma_x: float,
    sigma_y: float,
    kernel_size: int = 25
) -> np.ndarray:
    """
    Create a single Gabor filter kernel.
    
    Mathematical Formulation:
    -------------------------
    g(x, y) = exp(-x'²/2σx² - y'²/2σy²) * cos(2πf*x')
    
    where x' and y' are rotated coordinates.
    
    Args:
        orientation: Filter orientation (radians)
        frequency: Ridge frequency (ridges per pixel)
        sigma_x: Gaussian sigma along x' direction
        sigma_y: Gaussian sigma along y' direction
        kernel_size: Size of the kernel (odd number)
        
    Returns:
        Gabor kernel as 2D array
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    half_size = kernel_size // 2
    
    # Create coordinate grids
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Rotate coordinates
    cos_t = np.cos(orientation)
    sin_t = np.sin(orientation)
    
    x_rot = x_grid * cos_t + y_grid * sin_t
    y_rot = -x_grid * sin_t + y_grid * cos_t
    
    # Gaussian envelope
    gaussian = np.exp(
        -0.5 * (x_rot ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2)
    )
    
    # Sinusoidal component
    sinusoid = np.cos(2 * np.pi * frequency * x_rot)
    
    # Combined Gabor kernel
    kernel = gaussian * sinusoid
    
    # Normalize to zero mean
    kernel = kernel - np.mean(kernel)
    
    return kernel


def create_gabor_filter_bank(
    num_orientations: int = 8,
    frequency: float = 0.1,
    sigma_x: float = 4.0,
    sigma_y: float = 4.0,
    kernel_size: int = 25
) -> list:
    """
    Create a bank of Gabor filters at different orientations.
    
    Args:
        num_orientations: Number of orientation bins
        frequency: Ridge frequency
        sigma_x: Gaussian sigma in x direction
        sigma_y: Gaussian sigma in y direction
        kernel_size: Kernel size
        
    Returns:
        List of Gabor kernels
    """
    kernels = []
    
    for i in range(num_orientations):
        orientation = i * np.pi / num_orientations
        kernel = create_gabor_kernel(
            orientation, frequency, sigma_x, sigma_y, kernel_size
        )
        kernels.append(kernel)
    
    return kernels


def apply_gabor_filter(
    image: np.ndarray,
    orientation: np.ndarray,
    frequency: np.ndarray,
    sigma_x: float = 4.0,
    sigma_y: float = 4.0,
    kernel_size: int = 25,
    block_size: int = 8
) -> np.ndarray:
    """
    Apply locally-tuned Gabor filtering for fingerprint enhancement.
    
    Algorithm:
    ----------
    1. Quantize orientation field into discrete bins
    2. Create Gabor filter bank
    3. For each orientation bin, filter the image
    4. Combine filtered results based on local orientation
    
    This is more efficient than creating a unique filter per pixel.
    
    Args:
        image: Input fingerprint image
        orientation: Orientation field (radians)
        frequency: Frequency field (ridges per pixel)
        sigma_x: Gaussian sigma in x direction
        sigma_y: Gaussian sigma in y direction
        kernel_size: Size of Gabor kernel
        block_size: Block size for orientation quantization
        
    Returns:
        Enhanced fingerprint image
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    h, w = image.shape
    
    # Number of orientation bins
    num_orientations = 16
    
    # Get median frequency for filter bank (use adaptive if needed)
    median_freq = np.median(frequency[frequency > 0]) if np.any(frequency > 0) else 0.1
    
    # Create filter bank
    filter_bank = create_gabor_filter_bank(
        num_orientations, median_freq, sigma_x, sigma_y, kernel_size
    )
    
    # Apply each filter to the image
    filtered_images = []
    for kernel in filter_bank:
        filtered = ndimage.convolve(image, kernel, mode='reflect')
        filtered_images.append(filtered)
    
    # Quantize orientation to bin indices
    # Map orientation from [-π/2, π/2] to [0, π]
    orientation_normalized = orientation + np.pi / 2
    orientation_indices = np.floor(
        orientation_normalized / np.pi * num_orientations
    ).astype(int)
    orientation_indices = np.clip(orientation_indices, 0, num_orientations - 1)
    
    # Combine filtered results based on local orientation
    enhanced = np.zeros_like(image)
    
    for i in range(num_orientations):
        mask = orientation_indices == i
        enhanced[mask] = filtered_images[i][mask]
    
    return enhanced


def apply_adaptive_gabor_filter(
    image: np.ndarray,
    orientation: np.ndarray,
    frequency: np.ndarray,
    block_size: int = 16,
    kernel_size: int = 25
) -> np.ndarray:
    """
    Apply fully adaptive Gabor filtering with per-block frequency.
    
    This creates unique filters for different frequency regions,
    providing better enhancement at the cost of computation time.
    
    Args:
        image: Input fingerprint image
        orientation: Orientation field
        frequency: Frequency field
        block_size: Block size for adaptive filtering
        kernel_size: Gabor kernel size
        
    Returns:
        Enhanced fingerprint image
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    h, w = image.shape
    enhanced = np.zeros_like(image)
    
    # Number of orientation bins
    num_orientations = 16
    
    # Get unique frequency values (quantized)
    freq_quantized = np.round(frequency * 100) / 100
    unique_freqs = np.unique(freq_quantized[freq_quantized > 0])
    
    if len(unique_freqs) == 0:
        unique_freqs = [0.1]  # Default frequency
    
    # For each unique frequency, create filter bank and process
    for freq in unique_freqs:
        # Create filter bank for this frequency
        sigma = 0.5 / freq if freq > 0 else 5.0
        sigma = np.clip(sigma, 2.0, 8.0)
        
        filter_bank = create_gabor_filter_bank(
            num_orientations, freq, sigma, sigma, kernel_size
        )
        
        # Filter image
        filtered_images = [
            ndimage.convolve(image, k, mode='reflect') for k in filter_bank
        ]
        
        # Mask for this frequency
        freq_mask = np.abs(freq_quantized - freq) < 0.01
        
        # Orientation indices
        orientation_normalized = orientation + np.pi / 2
        orientation_indices = np.floor(
            orientation_normalized / np.pi * num_orientations
        ).astype(int)
        orientation_indices = np.clip(orientation_indices, 0, num_orientations - 1)
        
        # Apply to masked region
        for i in range(num_orientations):
            mask = freq_mask & (orientation_indices == i)
            enhanced[mask] = filtered_images[i][mask]
    
    return enhanced


class GaborEnhancer:
    """
    Gabor filter-based fingerprint enhancer.
    
    Combines orientation field estimation, frequency estimation,
    and Gabor filtering into a complete enhancement pipeline.
    """
    
    def __init__(
        self,
        block_size: int = 16,
        kernel_size: int = 25,
        sigma_x: float = 4.0,
        sigma_y: float = 4.0,
        num_orientations: int = 16,
        adaptive: bool = False
    ):
        """
        Initialize enhancer.
        
        Args:
            block_size: Block size for orientation/frequency estimation
            kernel_size: Gabor kernel size
            sigma_x: Gaussian sigma in x direction
            sigma_y: Gaussian sigma in y direction
            num_orientations: Number of orientation bins
            adaptive: Whether to use adaptive frequency filtering
        """
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.num_orientations = num_orientations
        self.adaptive = adaptive
    
    def enhance(
        self,
        image: np.ndarray,
        orientation: np.ndarray,
        frequency: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Enhance fingerprint image using Gabor filtering.
        
        Args:
            image: Input fingerprint image
            orientation: Pre-computed orientation field
            frequency: Pre-computed frequency field
            mask: Optional segmentation mask
            
        Returns:
            Enhanced fingerprint image
        """
        if self.adaptive:
            enhanced = apply_adaptive_gabor_filter(
                image, orientation, frequency,
                self.block_size, self.kernel_size
            )
        else:
            enhanced = apply_gabor_filter(
                image, orientation, frequency,
                self.sigma_x, self.sigma_y,
                self.kernel_size, self.block_size
            )
        
        # Apply mask if provided
        if mask is not None:
            enhanced = enhanced * mask
        
        # Normalize output
        enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced) + 1e-10)
        
        return enhanced


def enhance_fingerprint(
    image: np.ndarray,
    orientation: np.ndarray,
    frequency: np.ndarray,
    mask: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Convenience function for fingerprint enhancement.
    
    Args:
        image: Input fingerprint image
        orientation: Orientation field
        frequency: Frequency field
        mask: Optional segmentation mask
        **kwargs: Additional arguments for GaborEnhancer
        
    Returns:
        Enhanced fingerprint image
    """
    enhancer = GaborEnhancer(**kwargs)
    return enhancer.enhance(image, orientation, frequency, mask)
