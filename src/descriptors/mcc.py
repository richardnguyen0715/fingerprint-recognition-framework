"""
Minutia Cylinder Code (MCC) descriptor.

This module implements the MCC descriptor, a powerful local minutiae
descriptor that encodes the spatial and directional relationships
between a minutia and its neighbors.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.minutiae.minutiae_extraction import Minutia


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Minutia Cylinder Code (MCC):
# ---------------------------
# MCC represents each minutia as a 3D cylindrical structure encoding
# the local minutiae arrangement.
#
# Cylinder Structure:
# - Base: 2D spatial grid around the minutia
# - Height: Angular direction discretization
# - Each cell contains contribution from nearby minutiae
#
# Cell Value Computation:
# For cell at position (i, j, k) around minutia m:
#
# c(i,j,k) = Σ_t Ψ_s(d_s(m, m_t, i, j)) * Ψ_d(d_φ(m, m_t, k))
#
# where:
# - m_t are neighboring minutiae
# - d_s is spatial distance contribution
# - d_φ is directional distance contribution
# - Ψ_s, Ψ_d are Gaussian-like contribution functions
#
# Contribution Functions:
# Ψ_s(v) = sigmoid((v - σ_s) / δ_s) if v ≥ 0
#        = 1 - sigmoid((-v - σ_s) / δ_s) otherwise
#
# Binarization:
# The cylinder is binarized using a threshold μ:
# b(i,j,k) = 1 if c(i,j,k) ≥ μ, else 0
#
# Reference:
# Cappelli, R., Ferrara, M., & Maltoni, D. (2010).
# "Minutia cylinder-code: A new representation and matching technique
# for fingerprint recognition." IEEE TPAMI, 32(12), 2128-2141.
# =============================================================================


@dataclass
class MCCConfig:
    """Configuration for MCC descriptor computation."""
    # Cylinder radius (spatial extent)
    radius: float = 70.0
    # Number of spatial cells along diameter
    num_spatial_cells: int = 16
    # Number of angular sections
    num_angular_sections: int = 6
    # Spatial sigmoid parameter
    sigma_s: float = 7.0
    # Directional sigmoid parameter (radians)
    sigma_d: float = np.pi / 6
    # Binarization threshold
    mu: float = 0.1
    # Minimum minutiae in cylinder for valid descriptor
    min_minutiae: int = 2


def sigmoid(x: np.ndarray, mu: float, tau: float) -> np.ndarray:
    """
    Sigmoid function for soft assignment.
    
    Args:
        x: Input values
        mu: Shift parameter
        tau: Scale parameter
        
    Returns:
        Sigmoid output
    """
    return 1.0 / (1.0 + np.exp(-tau * (x - mu)))


def spatial_contribution(
    distance: float,
    sigma_s: float
) -> float:
    """
    Compute spatial contribution using Gaussian-like function.
    
    Ψ_s(d) = exp(-d²/(2*σ_s²))
    
    Args:
        distance: Distance from cell center to minutia
        sigma_s: Spatial sigma parameter
        
    Returns:
        Contribution value in [0, 1]
    """
    return np.exp(-distance**2 / (2 * sigma_s**2))


def directional_contribution(
    angle_diff: float,
    sigma_d: float
) -> float:
    """
    Compute directional contribution.
    
    The angular contribution uses a wrapped Gaussian:
    Ψ_d(Δφ) = exp(-Δφ²/(2*σ_d²))
    
    where Δφ is the angular difference normalized to [-π, π].
    
    Args:
        angle_diff: Angular difference (radians)
        sigma_d: Directional sigma parameter
        
    Returns:
        Contribution value in [0, 1]
    """
    # Normalize angle difference to [-π, π]
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    return np.exp(-angle_diff**2 / (2 * sigma_d**2))


def compute_mcc_cylinder(
    minutia: Minutia,
    all_minutiae: List[Minutia],
    config: MCCConfig
) -> Optional[np.ndarray]:
    """
    Compute MCC cylinder descriptor for a single minutia.
    
    Algorithm:
    ----------
    1. Create cylindrical grid around minutia
    2. For each cell, sum contributions from nearby minutiae
    3. Normalize and binarize
    
    Args:
        minutia: Reference minutia
        all_minutiae: All minutiae in the fingerprint
        config: MCC configuration parameters
        
    Returns:
        3D numpy array (cylinder) or None if invalid
    """
    ns = config.num_spatial_cells
    nd = config.num_angular_sections
    
    # Initialize cylinder
    cylinder = np.zeros((ns, ns, nd))
    
    # Cell dimensions
    cell_size = 2 * config.radius / ns
    angle_size = 2 * np.pi / nd
    
    # Reference minutia position and angle
    mx, my = minutia.x, minutia.y
    mtheta = minutia.angle
    
    # Rotation matrix to align cylinder with minutia orientation
    cos_t = np.cos(-mtheta)
    sin_t = np.sin(-mtheta)
    
    # Count contributing minutiae
    num_contributors = 0
    
    # For each other minutia
    for other in all_minutiae:
        if other.x == mx and other.y == my:
            continue
        
        # Relative position
        dx = other.x - mx
        dy = other.y - my
        
        # Rotate to align with minutia orientation
        dx_rot = dx * cos_t - dy * sin_t
        dy_rot = dx * sin_t + dy * cos_t
        
        # Check if within cylinder radius
        dist_sq = dx_rot**2 + dy_rot**2
        if dist_sq > config.radius**2:
            continue
        
        num_contributors += 1
        
        # Relative angle (in local coordinate system)
        relative_angle = other.angle - mtheta
        while relative_angle < 0:
            relative_angle += 2 * np.pi
        while relative_angle >= 2 * np.pi:
            relative_angle -= 2 * np.pi
        
        # Compute contribution to each cell
        for i in range(ns):
            for j in range(ns):
                # Cell center in local coordinates
                cell_x = -config.radius + (i + 0.5) * cell_size
                cell_y = -config.radius + (j + 0.5) * cell_size
                
                # Distance from other minutia to cell center
                dist = np.sqrt((dx_rot - cell_x)**2 + (dy_rot - cell_y)**2)
                
                # Spatial contribution
                s_contrib = spatial_contribution(dist, config.sigma_s)
                
                if s_contrib < 0.01:
                    continue
                
                for k in range(nd):
                    # Cell angle center
                    cell_angle = (k + 0.5) * angle_size
                    
                    # Directional contribution
                    angle_diff = relative_angle - cell_angle
                    d_contrib = directional_contribution(angle_diff, config.sigma_d)
                    
                    # Combined contribution
                    cylinder[i, j, k] += s_contrib * d_contrib
    
    # Check minimum contributors
    if num_contributors < config.min_minutiae:
        return None
    
    # Normalize
    if np.max(cylinder) > 0:
        cylinder = cylinder / np.max(cylinder)
    
    return cylinder


def binarize_cylinder(
    cylinder: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Binarize MCC cylinder.
    
    Args:
        cylinder: Continuous-valued cylinder
        threshold: Binarization threshold
        
    Returns:
        Binary cylinder
    """
    return (cylinder >= threshold).astype(np.uint8)


def compute_mcc_descriptors(
    minutiae: List[Minutia],
    config: Optional[MCCConfig] = None
) -> List[Tuple[int, np.ndarray]]:
    """
    Compute MCC descriptors for all minutiae.
    
    Args:
        minutiae: List of minutiae
        config: MCC configuration (uses default if None)
        
    Returns:
        List of (minutia_index, descriptor) tuples
        Only includes valid descriptors
    """
    if config is None:
        config = MCCConfig()
    
    descriptors = []
    
    for idx, minutia in enumerate(minutiae):
        cylinder = compute_mcc_cylinder(minutia, minutiae, config)
        
        if cylinder is not None:
            binary_cylinder = binarize_cylinder(cylinder, config.mu)
            descriptors.append((idx, binary_cylinder))
    
    return descriptors


def cylinder_similarity(
    cyl1: np.ndarray,
    cyl2: np.ndarray
) -> float:
    """
    Compute similarity between two MCC cylinders.
    
    Uses normalized intersection (similar to Tanimoto coefficient):
    
    sim(C1, C2) = ||C1 ∩ C2|| / ||C1 ∪ C2||
    
    For binary cylinders:
    sim = (C1 AND C2).sum() / (C1 OR C2).sum()
    
    Args:
        cyl1: First cylinder (binary)
        cyl2: Second cylinder (binary)
        
    Returns:
        Similarity score in [0, 1]
    """
    cyl1 = cyl1.astype(bool)
    cyl2 = cyl2.astype(bool)
    
    intersection = np.logical_and(cyl1, cyl2).sum()
    union = np.logical_or(cyl1, cyl2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def hamming_similarity(
    cyl1: np.ndarray,
    cyl2: np.ndarray
) -> float:
    """
    Compute similarity based on Hamming distance.
    
    sim = 1 - (Hamming distance / total bits)
    
    Args:
        cyl1: First cylinder (binary)
        cyl2: Second cylinder (binary)
        
    Returns:
        Similarity score in [0, 1]
    """
    cyl1 = cyl1.astype(bool).flatten()
    cyl2 = cyl2.astype(bool).flatten()
    
    hamming_dist = np.sum(cyl1 != cyl2)
    total_bits = len(cyl1)
    
    return 1.0 - hamming_dist / total_bits


class MCCDescriptor:
    """
    MCC descriptor extractor and container.
    
    Computes and stores MCC descriptors for all minutiae
    in a fingerprint.
    """
    
    def __init__(self, config: Optional[MCCConfig] = None):
        """
        Initialize MCC descriptor extractor.
        
        Args:
            config: MCC configuration
        """
        self.config = config or MCCConfig()
        self.descriptors: List[Tuple[int, np.ndarray]] = []
        self.minutiae: List[Minutia] = []
    
    def compute(self, minutiae: List[Minutia]) -> None:
        """
        Compute MCC descriptors for minutiae set.
        
        Args:
            minutiae: List of minutiae
        """
        self.minutiae = minutiae
        self.descriptors = compute_mcc_descriptors(minutiae, self.config)
    
    def get_descriptor(self, minutia_idx: int) -> Optional[np.ndarray]:
        """
        Get descriptor for a specific minutia.
        
        Args:
            minutia_idx: Index of minutia
            
        Returns:
            Descriptor array or None if not available
        """
        for idx, desc in self.descriptors:
            if idx == minutia_idx:
                return desc
        return None
    
    def __len__(self) -> int:
        """Return number of valid descriptors."""
        return len(self.descriptors)
