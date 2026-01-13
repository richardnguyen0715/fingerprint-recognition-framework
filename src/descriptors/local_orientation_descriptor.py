"""
Local orientation descriptor for fingerprint minutiae.

This module implements local orientation-based descriptors that
encode the ridge orientation pattern around each minutia.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.minutiae.minutiae_extraction import Minutia


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Local Orientation Descriptor:
# ----------------------------
# Encodes the local ridge orientation pattern around a minutia,
# similar to SIFT/SURF descriptors but adapted for fingerprints.
#
# Descriptor Structure:
# 1. Extract patch around minutia, aligned to minutia orientation
# 2. Divide patch into spatial cells
# 3. For each cell, compute orientation histogram
# 4. Concatenate histograms to form descriptor
#
# The descriptor captures:
# - Ridge flow pattern around the minutia
# - Rotation invariance through alignment
# - Scale invariance through normalization
#
# Reference:
# Feng, J. (2008).
# "Combining minutiae descriptors for fingerprint matching."
# Pattern Recognition, 41(1), 342-352.
# =============================================================================


@dataclass
class LODConfig:
    """Configuration for Local Orientation Descriptor."""
    # Patch size around minutia
    patch_size: int = 64
    # Number of spatial cells per dimension
    num_cells: int = 4
    # Number of orientation bins
    num_bins: int = 8
    # Gaussian weighting sigma (relative to patch size)
    gaussian_sigma: float = 0.5


def extract_aligned_patch(
    orientation_field: np.ndarray,
    minutia: Minutia,
    patch_size: int = 64
) -> Optional[np.ndarray]:
    """
    Extract orientation patch aligned to minutia direction.
    
    Args:
        orientation_field: Orientation field image
        minutia: Reference minutia
        patch_size: Size of extracted patch
        
    Returns:
        Aligned orientation patch or None if out of bounds
    """
    from scipy import ndimage
    
    h, w = orientation_field.shape
    half_size = patch_size // 2
    
    # Check bounds
    if (minutia.x - half_size < 0 or minutia.x + half_size >= w or
        minutia.y - half_size < 0 or minutia.y + half_size >= h):
        return None
    
    # Extract patch
    patch = orientation_field[
        minutia.y - half_size:minutia.y + half_size,
        minutia.x - half_size:minutia.x + half_size
    ].copy()
    
    # Rotate to align with minutia orientation
    # Convert minutia angle to degrees
    angle_deg = np.degrees(minutia.angle)
    
    # Rotate patch (using doubled angle for orientation)
    cos_2o = np.cos(2 * patch)
    sin_2o = np.sin(2 * patch)
    
    cos_2o_rot = ndimage.rotate(cos_2o, angle_deg, reshape=False, mode='reflect')
    sin_2o_rot = ndimage.rotate(sin_2o, angle_deg, reshape=False, mode='reflect')
    
    # Convert back to orientation
    patch_aligned = 0.5 * np.arctan2(sin_2o_rot, cos_2o_rot)
    
    # Shift orientation by minutia angle for invariance
    patch_aligned = patch_aligned - minutia.angle / 2
    
    return patch_aligned


def compute_orientation_histogram(
    orientations: np.ndarray,
    weights: np.ndarray,
    num_bins: int = 8
) -> np.ndarray:
    """
    Compute orientation histogram with soft binning.
    
    Args:
        orientations: Orientation values (radians, range [-π/2, π/2])
        weights: Weights for each orientation
        num_bins: Number of histogram bins
        
    Returns:
        Normalized histogram
    """
    # Normalize orientations to [0, π)
    orientations = orientations.copy()
    orientations = orientations % np.pi
    
    # Bin edges
    bin_width = np.pi / num_bins
    
    # Initialize histogram
    hist = np.zeros(num_bins)
    
    # Soft binning (linear interpolation between adjacent bins)
    bin_idx = orientations / bin_width
    lower_bin = np.floor(bin_idx).astype(int) % num_bins
    upper_bin = (lower_bin + 1) % num_bins
    
    upper_weight = bin_idx - np.floor(bin_idx)
    lower_weight = 1 - upper_weight
    
    # Accumulate with weights
    for i in range(len(orientations.flat)):
        w = weights.flat[i]
        lb = lower_bin.flat[i]
        ub = upper_bin.flat[i]
        lw = lower_weight.flat[i]
        uw = upper_weight.flat[i]
        
        hist[lb] += w * lw
        hist[ub] += w * uw
    
    # Normalize
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist


def compute_local_orientation_descriptor(
    orientation_field: np.ndarray,
    minutia: Minutia,
    config: Optional[LODConfig] = None
) -> Optional[np.ndarray]:
    """
    Compute local orientation descriptor for a minutia.
    
    Algorithm:
    ----------
    1. Extract aligned patch around minutia
    2. Divide into spatial cells
    3. Compute orientation histogram for each cell
    4. Apply Gaussian weighting
    5. Concatenate and normalize
    
    Args:
        orientation_field: Orientation field image
        minutia: Reference minutia
        config: Descriptor configuration
        
    Returns:
        Descriptor vector or None if invalid
    """
    if config is None:
        config = LODConfig()
    
    # Extract aligned patch
    patch = extract_aligned_patch(
        orientation_field, minutia, config.patch_size
    )
    
    if patch is None:
        return None
    
    # Gaussian weighting centered at minutia
    y, x = np.mgrid[0:config.patch_size, 0:config.patch_size]
    center = config.patch_size / 2
    sigma = config.gaussian_sigma * config.patch_size
    gaussian_weights = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    # Compute cell histograms
    cell_size = config.patch_size // config.num_cells
    descriptor = []
    
    for ci in range(config.num_cells):
        for cj in range(config.num_cells):
            # Cell region
            y_start = ci * cell_size
            y_end = y_start + cell_size
            x_start = cj * cell_size
            x_end = x_start + cell_size
            
            cell_orientations = patch[y_start:y_end, x_start:x_end]
            cell_weights = gaussian_weights[y_start:y_end, x_start:x_end]
            
            # Compute histogram
            hist = compute_orientation_histogram(
                cell_orientations, cell_weights, config.num_bins
            )
            
            descriptor.extend(hist)
    
    descriptor = np.array(descriptor)
    
    # L2 normalize
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor


def compute_local_descriptors(
    orientation_field: np.ndarray,
    minutiae: List[Minutia],
    config: Optional[LODConfig] = None
) -> List[Tuple[int, np.ndarray]]:
    """
    Compute local orientation descriptors for all minutiae.
    
    Args:
        orientation_field: Orientation field image
        minutiae: List of minutiae
        config: Descriptor configuration
        
    Returns:
        List of (minutia_index, descriptor) tuples
    """
    if config is None:
        config = LODConfig()
    
    descriptors = []
    
    for idx, minutia in enumerate(minutiae):
        desc = compute_local_orientation_descriptor(
            orientation_field, minutia, config
        )
        
        if desc is not None:
            descriptors.append((idx, desc))
    
    return descriptors


def descriptor_distance(
    desc1: np.ndarray,
    desc2: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute distance between two descriptors.
    
    Args:
        desc1: First descriptor
        desc2: Second descriptor
        metric: Distance metric ('euclidean', 'cosine', 'chi2')
        
    Returns:
        Distance value (lower = more similar)
    """
    if metric == 'euclidean':
        return np.linalg.norm(desc1 - desc2)
    
    elif metric == 'cosine':
        dot = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        if norm1 > 0 and norm2 > 0:
            return 1.0 - dot / (norm1 * norm2)
        return 1.0
    
    elif metric == 'chi2':
        # Chi-squared distance
        eps = 1e-10
        return 0.5 * np.sum((desc1 - desc2)**2 / (desc1 + desc2 + eps))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def descriptor_similarity(
    desc1: np.ndarray,
    desc2: np.ndarray,
    metric: str = 'cosine'
) -> float:
    """
    Compute similarity between two descriptors.
    
    Args:
        desc1: First descriptor
        desc2: Second descriptor
        metric: Similarity metric
        
    Returns:
        Similarity value in [0, 1]
    """
    if metric == 'cosine':
        dot = np.dot(desc1, desc2)
        norm1 = np.linalg.norm(desc1)
        norm2 = np.linalg.norm(desc2)
        if norm1 > 0 and norm2 > 0:
            return (dot / (norm1 * norm2) + 1) / 2
        return 0.0
    
    elif metric == 'euclidean':
        dist = np.linalg.norm(desc1 - desc2)
        return 1.0 / (1.0 + dist)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


class LocalOrientationDescriptor:
    """
    Local orientation descriptor extractor.
    """
    
    def __init__(self, config: Optional[LODConfig] = None):
        """
        Initialize descriptor extractor.
        
        Args:
            config: Descriptor configuration
        """
        self.config = config or LODConfig()
        self.descriptors: List[Tuple[int, np.ndarray]] = []
        self.minutiae: List[Minutia] = []
    
    def compute(
        self,
        orientation_field: np.ndarray,
        minutiae: List[Minutia]
    ) -> None:
        """
        Compute descriptors for all minutiae.
        
        Args:
            orientation_field: Orientation field image
            minutiae: List of minutiae
        """
        self.minutiae = minutiae
        self.descriptors = compute_local_descriptors(
            orientation_field, minutiae, self.config
        )
    
    def get_descriptor(self, minutia_idx: int) -> Optional[np.ndarray]:
        """
        Get descriptor for a specific minutia.
        
        Args:
            minutia_idx: Index of minutia
            
        Returns:
            Descriptor array or None
        """
        for idx, desc in self.descriptors:
            if idx == minutia_idx:
                return desc
        return None
    
    @property
    def descriptor_dim(self) -> int:
        """Return descriptor dimensionality."""
        return self.config.num_cells ** 2 * self.config.num_bins
    
    def __len__(self) -> int:
        """Return number of valid descriptors."""
        return len(self.descriptors)
