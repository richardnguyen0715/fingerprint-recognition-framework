"""
Minutiae extraction from fingerprint skeleton images.

This module implements minutiae detection using the crossing number
method on skeletonized fingerprint images.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Minutiae:
# ---------
# Minutiae are local discontinuities in the ridge pattern:
# - Ridge ending: A ridge that terminates abruptly
# - Ridge bifurcation: A single ridge that splits into two ridges
#
# Crossing Number Method:
# ----------------------
# For a pixel P with 8-neighbors in clockwise order (P1...P8):
#
# CN(P) = 0.5 * Σ |P_i - P_{i+1}|  (i = 1...8, P_9 = P_1)
#
# Classification:
# - CN = 0: Isolated point (noise)
# - CN = 1: Ridge ending
# - CN = 2: Ridge continuing point
# - CN = 3: Ridge bifurcation
# - CN > 3: Complex structure (usually noise)
#
# Each minutia has:
# - Position (x, y)
# - Type (ending or bifurcation)
# - Orientation θ (direction of the associated ridge)
#
# Reference:
# Maltoni, D., Maio, D., Jain, A. K., & Prabhakar, S. (2009).
# "Handbook of Fingerprint Recognition." Springer.
# =============================================================================


class MinutiaeType(Enum):
    """Enumeration of minutiae types."""
    ENDING = 1
    BIFURCATION = 3


@dataclass
class Minutia:
    """
    Represents a single minutia point.
    
    Attributes:
        x: X coordinate (column)
        y: Y coordinate (row)
        angle: Orientation angle (radians, range [0, 2π])
        minutiae_type: Type of minutia (ending or bifurcation)
        quality: Quality/confidence score (0 to 1)
    """
    x: int
    y: int
    angle: float
    minutiae_type: MinutiaeType
    quality: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'type': self.minutiae_type.name,
            'quality': self.quality
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Minutia':
        """Create from dictionary."""
        return cls(
            x=d['x'],
            y=d['y'],
            angle=d['angle'],
            minutiae_type=MinutiaeType[d['type']],
            quality=d.get('quality', 1.0)
        )


def compute_crossing_number(skeleton: np.ndarray, y: int, x: int) -> int:
    """
    Compute crossing number at a pixel.
    
    Mathematical Definition:
    -----------------------
    CN = 0.5 * Σ_{i=1}^{8} |P_i - P_{i+1}|
    
    where neighbors are in clockwise order starting from top.
    
    Args:
        skeleton: Binary skeleton image
        y, x: Pixel coordinates
        
    Returns:
        Crossing number (integer)
    """
    # Get 8 neighbors in clockwise order starting from top
    # P9 P2 P3
    # P8 P1 P4
    # P7 P6 P5
    neighbors = [
        skeleton[y-1, x],    # P2 (top)
        skeleton[y-1, x+1],  # P3 (top-right)
        skeleton[y, x+1],    # P4 (right)
        skeleton[y+1, x+1],  # P5 (bottom-right)
        skeleton[y+1, x],    # P6 (bottom)
        skeleton[y+1, x-1],  # P7 (bottom-left)
        skeleton[y, x-1],    # P8 (left)
        skeleton[y-1, x-1],  # P9 (top-left)
    ]
    
    # Compute crossing number
    cn = 0
    for i in range(8):
        cn += abs(int(neighbors[i]) - int(neighbors[(i+1) % 8]))
    
    return cn // 2


def estimate_minutia_orientation(
    skeleton: np.ndarray,
    y: int,
    x: int,
    minutiae_type: MinutiaeType,
    search_radius: int = 10
) -> float:
    """
    Estimate minutia orientation by tracing the connected ridge.
    
    Algorithm:
    ----------
    1. For endings: trace the ridge in the only connected direction
    2. For bifurcations: average the three branch directions
    
    The orientation points away from the minutia for endings,
    and toward the junction center for bifurcations.
    
    Args:
        skeleton: Binary skeleton image
        y, x: Minutia coordinates
        minutiae_type: Type of minutia
        search_radius: Radius to search for ridge direction
        
    Returns:
        Orientation angle in radians [0, 2π]
    """
    h, w = skeleton.shape
    
    # Find connected neighbors
    neighbor_offsets = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]
    
    connected = []
    for dy, dx in neighbor_offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
            connected.append((dy, dx))
    
    if len(connected) == 0:
        return 0.0
    
    if minutiae_type == MinutiaeType.ENDING:
        # For ending: trace the ridge
        if len(connected) == 1:
            # Follow the ridge
            dy, dx = connected[0]
            
            # Trace along the ridge to get direction
            trace_y, trace_x = y + dy, x + dx
            visited = {(y, x), (trace_y, trace_x)}
            
            for _ in range(search_radius):
                found_next = False
                for ndy, ndx in neighbor_offsets:
                    ny, nx = trace_y + ndy, trace_x + ndx
                    if (0 <= ny < h and 0 <= nx < w and 
                        skeleton[ny, nx] and (ny, nx) not in visited):
                        visited.add((ny, nx))
                        trace_y, trace_x = ny, nx
                        found_next = True
                        break
                if not found_next:
                    break
            
            # Direction from traced point back to minutia
            dir_y = y - trace_y
            dir_x = x - trace_x
            
        else:
            # Average direction
            dir_y = sum(d[0] for d in connected)
            dir_x = sum(d[1] for d in connected)
    
    else:  # Bifurcation
        # Average of all three directions
        dir_y = sum(d[0] for d in connected) / len(connected)
        dir_x = sum(d[1] for d in connected) / len(connected)
    
    # Compute angle
    angle = np.arctan2(dir_y, dir_x)
    
    # Normalize to [0, 2π]
    if angle < 0:
        angle += 2 * np.pi
    
    return angle


def extract_minutiae(
    skeleton: np.ndarray,
    border_margin: int = 20,
    orientation_field: Optional[np.ndarray] = None
) -> List[Minutia]:
    """
    Extract minutiae from skeleton image using crossing number.
    
    Algorithm Steps:
    ----------------
    1. For each foreground pixel in skeleton
    2. Compute crossing number
    3. If CN = 1: ridge ending
    4. If CN = 3: bifurcation
    5. Estimate orientation
    6. Filter border minutiae
    
    Args:
        skeleton: Binary skeleton image
        border_margin: Minimum distance from image border
        orientation_field: Optional pre-computed orientation field
        
    Returns:
        List of Minutia objects
    """
    h, w = skeleton.shape
    minutiae = []
    
    # Ensure binary
    skeleton = (skeleton > 0).astype(np.uint8)
    
    # Scan image (excluding borders)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 0:
                continue
            
            # Skip border region
            if (y < border_margin or y >= h - border_margin or
                x < border_margin or x >= w - border_margin):
                continue
            
            cn = compute_crossing_number(skeleton, y, x)
            
            if cn == 1:
                minutiae_type = MinutiaeType.ENDING
            elif cn == 3:
                minutiae_type = MinutiaeType.BIFURCATION
            else:
                continue
            
            # Estimate orientation
            if orientation_field is not None:
                # Use orientation field
                angle = orientation_field[y, x]
                # Convert ridge orientation to minutia orientation
                # For endings, point along the ridge
                if minutiae_type == MinutiaeType.ENDING:
                    # Determine which direction along the ridge
                    trace_angle = estimate_minutia_orientation(
                        skeleton, y, x, minutiae_type
                    )
                    # Adjust based on which half-plane
                    if abs(trace_angle - angle) > np.pi/2 and abs(trace_angle - angle) < 3*np.pi/2:
                        angle += np.pi
                    if angle < 0:
                        angle += 2 * np.pi
                    if angle >= 2 * np.pi:
                        angle -= 2 * np.pi
                else:
                    angle = estimate_minutia_orientation(
                        skeleton, y, x, minutiae_type
                    )
            else:
                angle = estimate_minutia_orientation(
                    skeleton, y, x, minutiae_type
                )
            
            minutia = Minutia(
                x=x,
                y=y,
                angle=angle,
                minutiae_type=minutiae_type
            )
            minutiae.append(minutia)
    
    return minutiae


def remove_spurious_minutiae(
    minutiae: List[Minutia],
    min_distance: int = 10
) -> List[Minutia]:
    """
    Remove spurious minutiae that are too close together.
    
    Spurious minutiae often occur:
    - Due to noise in the skeleton
    - At the edges of scars or creases
    - From broken ridges
    
    Two minutiae within min_distance pixels are likely spurious
    if they are an ending-bifurcation pair (indicates broken ridge).
    
    Args:
        minutiae: List of detected minutiae
        min_distance: Minimum distance between valid minutiae
        
    Returns:
        Filtered list of minutiae
    """
    if len(minutiae) < 2:
        return minutiae
    
    # Mark minutiae for removal
    to_remove = set()
    
    for i, m1 in enumerate(minutiae):
        if i in to_remove:
            continue
        
        for j, m2 in enumerate(minutiae[i+1:], i+1):
            if j in to_remove:
                continue
            
            # Compute distance
            dist = np.sqrt((m1.x - m2.x)**2 + (m1.y - m2.y)**2)
            
            if dist < min_distance:
                # Remove both if ending-bifurcation pair
                if m1.minutiae_type != m2.minutiae_type:
                    to_remove.add(i)
                    to_remove.add(j)
                # Otherwise remove the one with lower quality
                elif m1.quality < m2.quality:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    return [m for i, m in enumerate(minutiae) if i not in to_remove]


def filter_by_mask(
    minutiae: List[Minutia],
    mask: np.ndarray
) -> List[Minutia]:
    """
    Filter minutiae to keep only those within a mask region.
    
    Args:
        minutiae: List of minutiae
        mask: Binary mask (1 = valid region)
        
    Returns:
        Filtered list of minutiae
    """
    return [m for m in minutiae if mask[m.y, m.x] > 0]


def limit_minutiae_count(
    minutiae: List[Minutia],
    max_count: int = 100
) -> List[Minutia]:
    """
    Limit number of minutiae by keeping highest quality ones.
    
    Args:
        minutiae: List of minutiae
        max_count: Maximum number to keep
        
    Returns:
        Limited list of minutiae
    """
    if len(minutiae) <= max_count:
        return minutiae
    
    # Sort by quality (descending)
    sorted_minutiae = sorted(minutiae, key=lambda m: m.quality, reverse=True)
    return sorted_minutiae[:max_count]


class MinutiaeExtractor:
    """
    Configurable minutiae extraction pipeline.
    """
    
    def __init__(
        self,
        border_margin: int = 20,
        spurious_distance: int = 10,
        max_minutiae: int = 100
    ):
        """
        Initialize extractor.
        
        Args:
            border_margin: Margin to exclude from border
            spurious_distance: Distance for spurious removal
            max_minutiae: Maximum number of minutiae to extract
        """
        self.border_margin = border_margin
        self.spurious_distance = spurious_distance
        self.max_minutiae = max_minutiae
    
    def extract(
        self,
        skeleton: np.ndarray,
        mask: Optional[np.ndarray] = None,
        orientation_field: Optional[np.ndarray] = None
    ) -> List[Minutia]:
        """
        Extract minutiae from skeleton image.
        
        Args:
            skeleton: Binary skeleton image
            mask: Optional segmentation mask
            orientation_field: Optional orientation field for angles
            
        Returns:
            List of extracted minutiae
        """
        # Extract raw minutiae
        minutiae = extract_minutiae(
            skeleton, self.border_margin, orientation_field
        )
        
        # Filter by mask
        if mask is not None:
            minutiae = filter_by_mask(minutiae, mask)
        
        # Remove spurious
        minutiae = remove_spurious_minutiae(minutiae, self.spurious_distance)
        
        # Limit count
        minutiae = limit_minutiae_count(minutiae, self.max_minutiae)
        
        return minutiae
