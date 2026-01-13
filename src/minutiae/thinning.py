"""
Image thinning (skeletonization) algorithms.

This module implements thinning algorithms to reduce fingerprint ridge
images to single-pixel-wide skeletons, which is a prerequisite for
minutiae extraction.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Thinning/Skeletonization:
# ------------------------
# Thinning reduces binary objects to 1-pixel-wide skeletons while:
# - Preserving topology (connectivity)
# - Maintaining shape (medial axis approximation)
#
# Zhang-Suen Algorithm:
# A parallel thinning algorithm that iterates until convergence.
# Each iteration has two sub-iterations (odd and even).
#
# For a pixel P1 with 8-neighbors P2-P9 (clockwise from top):
#     P9 P2 P3
#     P8 P1 P4
#     P7 P6 P5
#
# Conditions for deletion in sub-iteration 1:
# - 2 ≤ B(P1) ≤ 6   (B = number of non-zero neighbors)
# - A(P1) = 1        (A = number of 01 patterns in ordered neighbors)
# - P2 * P4 * P6 = 0
# - P4 * P6 * P8 = 0
#
# Sub-iteration 2 differs in last two conditions:
# - P2 * P4 * P8 = 0
# - P2 * P6 * P8 = 0
#
# Reference:
# Zhang, T. Y., & Suen, C. Y. (1984).
# "A fast parallel algorithm for thinning digital patterns."
# Communications of the ACM, 27(3), 236-239.
# =============================================================================


def get_neighbors(image: np.ndarray, y: int, x: int) -> Tuple[int, ...]:
    """
    Get 8-connected neighbors of a pixel in clockwise order.
    
    Neighbor arrangement:
        P9 P2 P3
        P8 P1 P4
        P7 P6 P5
    
    Returns (P2, P3, P4, P5, P6, P7, P8, P9)
    
    Args:
        image: Binary image
        y, x: Pixel coordinates
        
    Returns:
        Tuple of 8 neighbor values (0 or 1)
    """
    return (
        image[y-1, x],    # P2
        image[y-1, x+1],  # P3
        image[y, x+1],    # P4
        image[y+1, x+1],  # P5
        image[y+1, x],    # P6
        image[y+1, x-1],  # P7
        image[y, x-1],    # P8
        image[y-1, x-1],  # P9
    )


def count_transitions(neighbors: Tuple[int, ...]) -> int:
    """
    Count 0-to-1 transitions in ordered neighbor sequence.
    
    This is the A(P1) function in Zhang-Suen algorithm.
    A high transition count indicates a junction point.
    
    Args:
        neighbors: Tuple of 8 neighbor values
        
    Returns:
        Number of 0→1 transitions
    """
    count = 0
    n = neighbors + (neighbors[0],)  # Circular
    
    for i in range(8):
        if n[i] == 0 and n[i+1] == 1:
            count += 1
    
    return count


def count_nonzero_neighbors(neighbors: Tuple[int, ...]) -> int:
    """
    Count number of non-zero neighbors.
    
    This is the B(P1) function in Zhang-Suen algorithm.
    
    Args:
        neighbors: Tuple of 8 neighbor values
        
    Returns:
        Count of non-zero neighbors
    """
    return sum(neighbors)


def zhang_suen_iteration(image: np.ndarray, iteration: int) -> np.ndarray:
    """
    Perform one sub-iteration of Zhang-Suen thinning.
    
    Args:
        image: Binary image (1 = foreground, 0 = background)
        iteration: Sub-iteration number (0 or 1)
        
    Returns:
        Thinned image after this sub-iteration
    """
    h, w = image.shape
    result = image.copy()
    markers = []
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if image[y, x] == 0:
                continue
            
            neighbors = get_neighbors(image, y, x)
            P2, P3, P4, P5, P6, P7, P8, P9 = neighbors
            
            # Condition 1: 2 ≤ B(P1) ≤ 6
            B = count_nonzero_neighbors(neighbors)
            if B < 2 or B > 6:
                continue
            
            # Condition 2: A(P1) = 1
            A = count_transitions(neighbors)
            if A != 1:
                continue
            
            # Condition 3 and 4 depend on iteration
            if iteration == 0:
                # Sub-iteration 1: P2*P4*P6=0 and P4*P6*P8=0
                if P2 * P4 * P6 != 0:
                    continue
                if P4 * P6 * P8 != 0:
                    continue
            else:
                # Sub-iteration 2: P2*P4*P8=0 and P2*P6*P8=0
                if P2 * P4 * P8 != 0:
                    continue
                if P2 * P6 * P8 != 0:
                    continue
            
            markers.append((y, x))
    
    # Delete marked pixels
    for y, x in markers:
        result[y, x] = 0
    
    return result, len(markers)


def zhang_suen_thinning(
    image: np.ndarray,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Apply Zhang-Suen thinning algorithm.
    
    Args:
        image: Binary image (ridges = 1, background = 0)
        max_iterations: Maximum number of iteration pairs
        
    Returns:
        Thinned (skeletonized) image
    """
    # Ensure binary image
    binary = (image > 0).astype(np.uint8)
    
    # Pad to handle border pixels
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    
    for _ in range(max_iterations):
        # Sub-iteration 1
        padded, changed1 = zhang_suen_iteration(padded, 0)
        
        # Sub-iteration 2
        padded, changed2 = zhang_suen_iteration(padded, 1)
        
        # Check convergence
        if changed1 == 0 and changed2 == 0:
            break
    
    # Remove padding
    return padded[1:-1, 1:-1]


def guo_hall_thinning(
    image: np.ndarray,
    max_iterations: int = 100
) -> np.ndarray:
    """
    Apply Guo-Hall thinning algorithm.
    
    Similar to Zhang-Suen but with different deletion conditions
    that better preserve 4-connectivity.
    
    Mathematical Conditions:
    -----------------------
    For deletion of pixel P1:
    - C(P1) = 1  (connectivity check)
    - 2 ≤ N(P1) ≤ 3  (neighbor count)
    
    Different conditions for odd/even iterations.
    
    Args:
        image: Binary image
        max_iterations: Maximum iterations
        
    Returns:
        Thinned image
    """
    # Ensure binary image
    binary = (image > 0).astype(np.uint8)
    
    # Pad image
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    
    h, w = padded.shape
    
    for iteration in range(max_iterations):
        markers = []
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if padded[y, x] == 0:
                    continue
                
                neighbors = get_neighbors(padded, y, x)
                P2, P3, P4, P5, P6, P7, P8, P9 = neighbors
                
                # C(P1): Connectivity number
                # Using simplified check: A(P1) = 1
                A = count_transitions(neighbors)
                
                # N(P1): Number of non-zero in {P2, P4, P6, P8}
                N1 = (P2 | P3) + (P4 | P5) + (P6 | P7) + (P8 | P9)
                N2 = (P2 | P9) + (P4 | P3) + (P6 | P5) + (P8 | P7)
                N = min(N1, N2)
                
                if A != 1 or N < 2 or N > 3:
                    continue
                
                # Different conditions for odd/even iterations
                if iteration % 2 == 0:
                    m = (P2 | P3 | (not P5)) & P4
                else:
                    m = (P6 | P7 | (not P9)) & P8
                
                if not m:
                    markers.append((y, x))
        
        if len(markers) == 0:
            break
        
        for y, x in markers:
            padded[y, x] = 0
    
    return padded[1:-1, 1:-1]


def binarize_image(
    image: np.ndarray,
    method: str = 'adaptive',
    block_size: int = 15,
    offset: int = 10
) -> np.ndarray:
    """
    Binarize fingerprint image.
    
    Args:
        image: Grayscale fingerprint image
        method: 'global', 'adaptive', or 'otsu'
        block_size: Block size for adaptive method
        offset: Offset for adaptive thresholding
        
    Returns:
        Binary image (ridges = 1, background = 0)
    """
    import cv2
    
    # Convert to uint8 if needed
    if image.dtype in [np.float32, np.float64]:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    
    if method == 'global':
        threshold = np.mean(image)
        binary = (image < threshold).astype(np.uint8)
        
    elif method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8)
        
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, offset
        )
        binary = (binary > 0).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    return binary


def thin_fingerprint(
    image: np.ndarray,
    method: str = 'zhang_suen',
    binarize: bool = True,
    binarization_method: str = 'adaptive'
) -> np.ndarray:
    """
    Complete thinning pipeline for fingerprint images.
    
    Args:
        image: Input fingerprint image (grayscale or binary)
        method: Thinning method ('zhang_suen' or 'guo_hall')
        binarize: Whether to binarize before thinning
        binarization_method: Method for binarization
        
    Returns:
        Thinned (skeletonized) fingerprint image
    """
    # Binarize if needed
    if binarize:
        binary = binarize_image(image, binarization_method)
    else:
        binary = (image > 0).astype(np.uint8)
    
    # Apply thinning
    if method == 'zhang_suen':
        skeleton = zhang_suen_thinning(binary)
    elif method == 'guo_hall':
        skeleton = guo_hall_thinning(binary)
    else:
        raise ValueError(f"Unknown thinning method: {method}")
    
    return skeleton


class Thinner:
    """
    Configurable fingerprint thinning processor.
    """
    
    def __init__(
        self,
        method: str = 'zhang_suen',
        binarization_method: str = 'adaptive',
        block_size: int = 15,
        offset: int = 10,
        max_iterations: int = 100
    ):
        """
        Initialize thinner.
        
        Args:
            method: Thinning algorithm
            binarization_method: Binarization method
            block_size: Block size for adaptive binarization
            offset: Offset for adaptive binarization
            max_iterations: Maximum thinning iterations
        """
        self.method = method
        self.binarization_method = binarization_method
        self.block_size = block_size
        self.offset = offset
        self.max_iterations = max_iterations
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize and thin fingerprint image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Thinned image
        """
        return thin_fingerprint(
            image, 
            self.method, 
            binarize=True,
            binarization_method=self.binarization_method
        )
