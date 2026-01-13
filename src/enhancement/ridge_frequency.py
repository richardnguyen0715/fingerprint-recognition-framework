"""
Ridge frequency estimation for fingerprint images.

This module implements ridge frequency estimation, which determines
the local spacing between fingerprint ridges. This is essential for
Gabor filter tuning in fingerprint enhancement.
"""

import numpy as np
from scipy import ndimage, signal
from typing import Tuple, Optional


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Ridge Frequency Estimation:
# --------------------------
# Ridge frequency f represents the number of ridges per unit length.
# Equivalently, wavelength λ = 1/f is the spacing between ridges.
#
# For adult fingerprints, typical ridge spacing is 3-10 pixels at 500 DPI.
#
# Algorithm (Projection-Based):
# 1. Rotate local block to align ridges vertically
# 2. Compute row-wise signature (projection)
# 3. Find peaks in signature
# 4. Frequency = 1 / (average peak distance)
#
# Reference:
# Hong, L., Wan, Y., & Jain, A. (1998).
# "Fingerprint image enhancement: algorithm and performance evaluation."
# IEEE TPAMI, 20(8), 777-789.
# =============================================================================


def estimate_frequency_block(
    image: np.ndarray,
    orientation: float,
    block_size: int = 32,
    min_wavelength: float = 5.0,
    max_wavelength: float = 15.0
) -> float:
    """
    Estimate ridge frequency for a single block.
    
    Mathematical Formulation:
    -------------------------
    1. Rotate block so ridges are vertical (perpendicular to orientation)
    2. Compute x-signature: s(x) = Σ_y I(x, y)
    3. Find peaks in s(x)
    4. frequency = 1 / mean(peak_distances)
    
    Args:
        image: Input image block
        orientation: Local ridge orientation (radians)
        block_size: Size of the block
        min_wavelength: Minimum expected ridge wavelength
        max_wavelength: Maximum expected ridge wavelength
        
    Returns:
        Estimated frequency (0 if invalid)
    """
    # Create rotation matrix to align ridges vertically
    # We rotate by -orientation to make ridges perpendicular to x-axis
    cos_t = np.cos(-orientation)
    sin_t = np.sin(-orientation)
    
    # Generate coordinates centered at block center
    center = block_size / 2
    y_coords, x_coords = np.mgrid[0:block_size, 0:block_size]
    x_centered = x_coords - center
    y_centered = y_coords - center
    
    # Rotate coordinates
    x_rot = x_centered * cos_t - y_centered * sin_t + center
    y_rot = x_centered * sin_t + y_centered * cos_t + center
    
    # Sample rotated block using bilinear interpolation
    rotated = ndimage.map_coordinates(
        image, 
        [y_rot.ravel(), x_rot.ravel()],
        order=1,
        mode='reflect'
    ).reshape(block_size, block_size)
    
    # Compute x-signature (sum along columns)
    signature = np.mean(rotated, axis=0)
    
    # Remove DC component and normalize
    signature = signature - np.mean(signature)
    
    if np.std(signature) < 1e-10:
        return 0.0
    
    signature = signature / np.std(signature)
    
    # Find peaks in signature
    peaks, _ = signal.find_peaks(signature, distance=int(min_wavelength * 0.7))
    
    if len(peaks) < 2:
        return 0.0
    
    # Compute average peak distance (wavelength)
    peak_distances = np.diff(peaks)
    wavelength = np.mean(peak_distances)
    
    # Validate wavelength
    if wavelength < min_wavelength or wavelength > max_wavelength:
        return 0.0
    
    return 1.0 / wavelength


def estimate_frequency_field(
    image: np.ndarray,
    orientation: np.ndarray,
    block_size: int = 32,
    min_wavelength: float = 5.0,
    max_wavelength: float = 15.0
) -> np.ndarray:
    """
    Estimate ridge frequency field for entire image.
    
    Args:
        image: Input fingerprint image
        orientation: Orientation field (same size as image)
        block_size: Block size for frequency estimation
        min_wavelength: Minimum expected wavelength
        max_wavelength: Maximum expected wavelength
        
    Returns:
        Frequency field (block-wise)
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    h, w = image.shape
    
    # Block-wise orientation (subsample from pixel-wise)
    orientation_block = orientation[
        block_size//2::block_size, 
        block_size//2::block_size
    ]
    
    num_blocks_y = h // block_size
    num_blocks_x = w // block_size
    
    frequency = np.zeros((num_blocks_y, num_blocks_x))
    
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            y_start = by * block_size
            x_start = bx * block_size
            
            # Extract block
            block = image[y_start:y_start+block_size, x_start:x_start+block_size]
            
            # Get orientation for this block
            oy = min(by, orientation_block.shape[0] - 1)
            ox = min(bx, orientation_block.shape[1] - 1)
            block_orientation = orientation_block[oy, ox]
            
            # Estimate frequency
            frequency[by, bx] = estimate_frequency_block(
                block, block_orientation, block_size,
                min_wavelength, max_wavelength
            )
    
    return frequency


def interpolate_frequency(
    frequency: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Interpolate invalid frequency values from neighbors.
    
    Args:
        frequency: Frequency field with some zero (invalid) values
        mask: Optional mask of valid regions
        
    Returns:
        Interpolated frequency field
    """
    valid = frequency > 0
    
    if mask is not None:
        # Only consider values inside mask
        valid = valid & (mask > 0)
    
    if np.sum(valid) == 0:
        # No valid values, return default
        return np.full_like(frequency, 1.0 / 9.0)  # Default 9-pixel wavelength
    
    # Use median of valid values for invalid regions
    median_freq = np.median(frequency[valid])
    
    # Interpolate using Gaussian-weighted average
    result = frequency.copy()
    
    # Create distance-weighted kernel
    kernel_size = 5
    y, x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    
    # Weighted average of valid neighbors
    freq_weighted = ndimage.convolve(frequency * valid, kernel, mode='reflect')
    weight_sum = ndimage.convolve(valid.astype(float), kernel, mode='reflect')
    
    # Replace invalid values
    invalid = ~valid
    result[invalid] = np.where(
        weight_sum[invalid] > 0.1,
        freq_weighted[invalid] / weight_sum[invalid],
        median_freq
    )
    
    return result


def smooth_frequency_field(
    frequency: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Smooth frequency field using Gaussian filter.
    
    Args:
        frequency: Input frequency field
        sigma: Gaussian smoothing sigma
        
    Returns:
        Smoothed frequency field
    """
    return ndimage.gaussian_filter(frequency, sigma)


def resize_frequency_field(
    frequency: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize frequency field to match image dimensions.
    
    Args:
        frequency: Block-wise frequency field
        target_shape: Target (height, width)
        
    Returns:
        Pixel-wise frequency field
    """
    from scipy.ndimage import zoom
    
    zoom_y = target_shape[0] / frequency.shape[0]
    zoom_x = target_shape[1] / frequency.shape[1]
    
    return zoom(frequency, (zoom_y, zoom_x), order=1)


class RidgeFrequencyEstimator:
    """
    Configurable ridge frequency estimator.
    """
    
    # Typical ridge wavelength range for 500 DPI images
    DEFAULT_MIN_WAVELENGTH = 5.0
    DEFAULT_MAX_WAVELENGTH = 15.0
    
    def __init__(
        self,
        block_size: int = 32,
        min_wavelength: float = DEFAULT_MIN_WAVELENGTH,
        max_wavelength: float = DEFAULT_MAX_WAVELENGTH,
        smooth_sigma: float = 1.0
    ):
        """
        Initialize estimator.
        
        Args:
            block_size: Block size for estimation
            min_wavelength: Minimum expected wavelength
            max_wavelength: Maximum expected wavelength
            smooth_sigma: Smoothing sigma for frequency field
        """
        self.block_size = block_size
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.smooth_sigma = smooth_sigma
    
    def estimate(
        self,
        image: np.ndarray,
        orientation: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate ridge frequency field.
        
        Args:
            image: Input fingerprint image
            orientation: Orientation field (pixel-wise)
            mask: Optional segmentation mask
            
        Returns:
            Frequency field (pixel-wise)
        """
        # Block-wise estimation
        frequency = estimate_frequency_field(
            image, orientation, self.block_size,
            self.min_wavelength, self.max_wavelength
        )
        
        # Interpolate invalid values
        frequency = interpolate_frequency(frequency, mask)
        
        # Smooth
        frequency = smooth_frequency_field(frequency, self.smooth_sigma)
        
        # Resize to image dimensions
        frequency = resize_frequency_field(frequency, image.shape)
        
        return frequency
