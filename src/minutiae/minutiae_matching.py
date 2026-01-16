"""
Minutiae-based fingerprint matching.

This module implements minutiae matching algorithms that compare
two fingerprints based on their minutiae sets.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.minutiae.minutiae_extraction import Minutia, MinutiaeType
from src.baselines.ssim import FingerprintMatcher


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Minutiae Matching Problem:
# -------------------------
# Given two minutiae sets T = {m_1, ..., m_n} and Q = {m'_1, ..., m'_k},
# find the optimal correspondence and compute a similarity score.
#
# Challenges:
# - Non-linear distortion (skin elasticity)
# - Missing/spurious minutiae
# - Rotation and translation between captures
#
# Approach (Point Pattern Matching):
# 1. Alignment: Estimate transformation between fingerprints
# 2. Pairing: Match minutiae based on position and angle
# 3. Scoring: Compute similarity from matched pairs
#
# Matching Criteria:
# Two minutiae m = (x, y, θ) and m' = (x', y', θ') match if:
# - Spatial distance: ||T(x,y) - (x',y')|| < d_threshold
# - Angular difference: |T_θ(θ) - θ'| < θ_threshold
#
# where T is the estimated transformation.
#
# Reference:
# Maltoni, D., Maio, D., Jain, A. K., & Prabhakar, S. (2009).
# "Handbook of Fingerprint Recognition." Springer.
# =============================================================================


@dataclass
class MinutiaeMatch:
    """
    Represents a matched pair of minutiae.
    
    Attributes:
        idx1: Index in first minutiae set
        idx2: Index in second minutiae set
        distance: Spatial distance between matched minutiae
        angle_diff: Angular difference between matched minutiae
    """
    idx1: int
    idx2: int
    distance: float
    angle_diff: float


def compute_transformation(
    minutiae1: List[Minutia],
    minutiae2: List[Minutia],
    idx1: int,
    idx2: int
) -> Tuple[float, float, float]:
    """
    Compute transformation parameters aligning minutiae.
    
    Uses two corresponding minutiae to estimate:
    - Translation (dx, dy)
    - Rotation (dtheta)
    
    Args:
        minutiae1: First minutiae set
        minutiae2: Second minutiae set
        idx1: Index of reference minutia in set 1
        idx2: Index of corresponding minutia in set 2
        
    Returns:
        Tuple of (dx, dy, dtheta)
    """
    m1 = minutiae1[idx1]
    m2 = minutiae2[idx2]
    
    # Rotation: angle difference
    dtheta = m2.angle - m1.angle
    
    # Normalize to [-π, π]
    while dtheta > np.pi:
        dtheta -= 2 * np.pi
    while dtheta < -np.pi:
        dtheta += 2 * np.pi
    
    # Apply rotation to position
    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)
    
    x1_rot = m1.x * cos_t - m1.y * sin_t
    y1_rot = m1.x * sin_t + m1.y * cos_t
    
    # Translation
    dx = m2.x - x1_rot
    dy = m2.y - y1_rot
    
    return dx, dy, dtheta


def apply_transformation(
    minutiae: List[Minutia],
    dx: float,
    dy: float,
    dtheta: float
) -> List[Tuple[float, float, float]]:
    """
    Apply transformation to minutiae set.
    
    Args:
        minutiae: Minutiae to transform
        dx, dy: Translation
        dtheta: Rotation
        
    Returns:
        List of transformed (x, y, angle) tuples
    """
    transformed = []
    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)
    
    for m in minutiae:
        x_rot = m.x * cos_t - m.y * sin_t
        y_rot = m.x * sin_t + m.y * cos_t
        
        x_new = x_rot + dx
        y_new = y_rot + dy
        
        angle_new = m.angle + dtheta
        # Normalize angle to [0, 2π]
        while angle_new < 0:
            angle_new += 2 * np.pi
        while angle_new >= 2 * np.pi:
            angle_new -= 2 * np.pi
        
        transformed.append((x_new, y_new, angle_new))
    
    return transformed


def find_matching_pairs(
    transformed1: List[Tuple[float, float, float]],
    minutiae2: List[Minutia],
    distance_threshold: float = 15.0,
    angle_threshold: float = 0.26  # ~15 degrees
) -> List[MinutiaeMatch]:
    """
    Find matching minutiae pairs after alignment.
    
    Two minutiae match if:
    - Euclidean distance < distance_threshold
    - Angular difference < angle_threshold
    
    Uses greedy matching to avoid multiple assignments.
    
    Args:
        transformed1: Transformed minutiae from set 1
        minutiae2: Second minutiae set (reference)
        distance_threshold: Maximum spatial distance
        angle_threshold: Maximum angular difference (radians)
        
    Returns:
        List of matched pairs
    """
    matches = []
    used2 = set()
    
    # Compute all pairwise distances and angles
    candidates = []
    for i, (x1, y1, a1) in enumerate(transformed1):
        for j, m2 in enumerate(minutiae2):
            if j in used2:
                continue
            
            # Spatial distance
            dist = np.sqrt((x1 - m2.x)**2 + (y1 - m2.y)**2)
            
            if dist > distance_threshold:
                continue
            
            # Angular difference
            angle_diff = abs(a1 - m2.angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            if angle_diff > angle_threshold:
                continue
            
            candidates.append((dist, angle_diff, i, j))
    
    # Sort by distance and greedily assign
    candidates.sort(key=lambda x: x[0])
    used1 = set()
    
    for dist, angle_diff, i, j in candidates:
        if i in used1 or j in used2:
            continue
        
        matches.append(MinutiaeMatch(i, j, dist, angle_diff))
        used1.add(i)
        used2.add(j)
    
    return matches


def ransac_alignment(
    minutiae1: List[Minutia],
    minutiae2: List[Minutia],
    distance_threshold: float = 15.0,
    angle_threshold: float = 0.26,
    num_iterations: int = 1000,
    min_inliers: int = 5,
    random_state: Optional[int] = None
) -> Tuple[List[MinutiaeMatch], Tuple[float, float, float]]:
    """
    RANSAC-based minutiae alignment and matching.
    
    Algorithm:
    ----------
    1. Randomly select a minutia pair as reference
    2. Compute transformation from this pair
    3. Apply transformation and count inliers
    4. Keep best transformation
    
    Args:
        minutiae1: First minutiae set
        minutiae2: Second minutiae set
        distance_threshold: Distance threshold for inliers
        angle_threshold: Angle threshold for inliers
        num_iterations: Number of RANSAC iterations
        min_inliers: Minimum inliers for valid match
        random_state: Random seed for reproducibility (None for random behavior)
        
    Returns:
        Tuple of (best_matches, best_transformation)
    """
    if len(minutiae1) == 0 or len(minutiae2) == 0:
        return [], (0.0, 0.0, 0.0)
    
    # Set random seed for reproducibility
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    
    best_matches = []
    best_transform = (0.0, 0.0, 0.0)
    best_score = 0
    
    for _ in range(num_iterations):
        # Randomly select reference pair
        idx1 = rng.randint(len(minutiae1))
        idx2 = rng.randint(len(minutiae2))
        
        # Optionally: require same type
        if minutiae1[idx1].minutiae_type != minutiae2[idx2].minutiae_type:
            continue
        
        # Compute transformation
        dx, dy, dtheta = compute_transformation(minutiae1, minutiae2, idx1, idx2)
        
        # Apply transformation
        transformed = apply_transformation(minutiae1, dx, dy, dtheta)
        
        # Find matches
        matches = find_matching_pairs(
            transformed, minutiae2, distance_threshold, angle_threshold
        )
        
        # Score (number of matches)
        score = len(matches)
        
        if score > best_score:
            best_score = score
            best_matches = matches
            best_transform = (dx, dy, dtheta)
    
    return best_matches, best_transform


def compute_matching_score(
    matches: List[MinutiaeMatch],
    num_minutiae1: int,
    num_minutiae2: int,
    min_matched: int = 8
) -> float:
    """
    Compute matching score from minutiae correspondences.
    
    Score Formulation:
    ------------------
    score = n_matched² / (n_1 * n_2)
    
    This rewards more matches while penalizing unmatched minutiae.
    
    Alternative (if insufficient matches):
    score = 0 if n_matched < min_matched
    
    Args:
        matches: List of matched minutiae pairs
        num_minutiae1: Total minutiae in first set
        num_minutiae2: Total minutiae in second set
        min_matched: Minimum matches for non-zero score
        
    Returns:
        Similarity score in [0, 1]
    """
    n_matched = len(matches)
    
    if n_matched < min_matched:
        return 0.0
    
    if num_minutiae1 == 0 or num_minutiae2 == 0:
        return 0.0
    
    # Compute score
    score = (n_matched ** 2) / (num_minutiae1 * num_minutiae2)
    
    # Clamp to [0, 1]
    return min(1.0, score)


class MinutiaeMatcher(FingerprintMatcher):
    """
    Minutiae-based fingerprint matcher.
    
    This matcher compares fingerprints using their extracted minutiae
    sets, finding optimal alignment and computing similarity from
    the number of matched minutiae.
    """
    
    def __init__(
        self,
        distance_threshold: float = 15.0,
        angle_threshold: float = 0.26,
        min_matched_minutiae: int = 8,
        alignment_method: str = 'ransac',
        ransac_iterations: int = 1000,
        random_state: Optional[int] = 42
    ):
        """
        Initialize minutiae matcher.
        
        Args:
            distance_threshold: Max distance for minutiae pairing (pixels)
            angle_threshold: Max angle difference for pairing (radians)
            min_matched_minutiae: Minimum matches for valid comparison
            alignment_method: Alignment method ('ransac' or 'exhaustive')
            ransac_iterations: Number of RANSAC iterations
            random_state: Random seed for RANSAC reproducibility (None for random)
        """
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.min_matched_minutiae = min_matched_minutiae
        self.alignment_method = alignment_method
        self.ransac_iterations = ransac_iterations
        self.random_state = random_state
    
    @property
    def name(self) -> str:
        return "Minutiae"
    
    def match_minutiae(
        self,
        minutiae1: List[Minutia],
        minutiae2: List[Minutia]
    ) -> Tuple[float, List[MinutiaeMatch]]:
        """
        Match two minutiae sets.
        
        Args:
            minutiae1: First minutiae set
            minutiae2: Second minutiae set
            
        Returns:
            Tuple of (score, matched_pairs)
        """
        if len(minutiae1) == 0 or len(minutiae2) == 0:
            return 0.0, []
        
        if self.alignment_method == 'ransac':
            matches, _ = ransac_alignment(
                minutiae1, minutiae2,
                self.distance_threshold,
                self.angle_threshold,
                self.ransac_iterations,
                self.min_matched_minutiae,
                self.random_state
            )
        else:
            # Exhaustive search (slower but deterministic)
            best_matches = []
            for i in range(len(minutiae1)):
                for j in range(len(minutiae2)):
                    if minutiae1[i].minutiae_type != minutiae2[j].minutiae_type:
                        continue
                    
                    dx, dy, dtheta = compute_transformation(
                        minutiae1, minutiae2, i, j
                    )
                    transformed = apply_transformation(minutiae1, dx, dy, dtheta)
                    matches = find_matching_pairs(
                        transformed, minutiae2,
                        self.distance_threshold, self.angle_threshold
                    )
                    
                    if len(matches) > len(best_matches):
                        best_matches = matches
            
            matches = best_matches
        
        score = compute_matching_score(
            matches, len(minutiae1), len(minutiae2),
            self.min_matched_minutiae
        )
        
        return score, matches
    
    def match(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray
    ) -> float:
        """
        Match two fingerprint images using minutiae.
        
        Note: This method expects pre-extracted minutiae as input.
        For image inputs, use the full pipeline with MinutiaeExtractor.
        
        This is a placeholder that expects the inputs to be minutiae
        lists rather than images. For the full pipeline, see
        MinutiaeMatchingPipeline.
        
        Args:
            sample_a: First input (minutiae list or processed features)
            sample_b: Second input
            
        Returns:
            Similarity score
        """
        # This method is for interface compatibility
        # In practice, use match_minutiae() with extracted minutiae
        raise NotImplementedError(
            "MinutiaeMatcher.match() requires minutiae, not images. "
            "Use MinutiaeMatchingPipeline for image-to-image matching."
        )


class MinutiaeMatchingPipeline:
    """
    Complete pipeline for minutiae-based fingerprint matching.
    
    Combines:
    - Preprocessing
    - Enhancement (optional)
    - Binarization
    - Thinning
    - Minutiae extraction
    - Minutiae matching
    """
    
    def __init__(
        self,
        preprocessor=None,
        enhancer=None,
        thinner=None,
        extractor=None,
        matcher=None
    ):
        """
        Initialize pipeline.
        
        Args:
            preprocessor: FingerprintPreprocessor instance
            enhancer: GaborEnhancer instance
            thinner: Thinner instance
            extractor: MinutiaeExtractor instance
            matcher: MinutiaeMatcher instance
        """
        from src.minutiae.thinning import Thinner
        from src.minutiae.minutiae_extraction import MinutiaeExtractor
        
        self.preprocessor = preprocessor
        self.enhancer = enhancer
        self.thinner = thinner or Thinner()
        self.extractor = extractor or MinutiaeExtractor()
        self.matcher = matcher or MinutiaeMatcher(random_state=42)
    
    def extract_minutiae(self, image: np.ndarray) -> List[Minutia]:
        """
        Extract minutiae from an image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            List of extracted minutiae
        """
        # Preprocess if configured
        if self.preprocessor is not None:
            image, mask = self.preprocessor(image)
        else:
            mask = None
        
        # Enhance if configured
        if self.enhancer is not None:
            from src.enhancement import (
                estimate_orientation_field,
                RidgeFrequencyEstimator
            )
            orientation = estimate_orientation_field(image)
            freq_estimator = RidgeFrequencyEstimator()
            frequency = freq_estimator.estimate(image, orientation, mask)
            image = self.enhancer.enhance(image, orientation, frequency, mask)
        
        # Thin
        skeleton = self.thinner.process(image)
        
        # Extract minutiae
        minutiae = self.extractor.extract(skeleton, mask)
        
        return minutiae
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> float:
        """
        Match two fingerprint images.
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            
        Returns:
            Similarity score
        """
        minutiae1 = self.extract_minutiae(image1)
        minutiae2 = self.extract_minutiae(image2)
        
        score, _ = self.matcher.match_minutiae(minutiae1, minutiae2)
        
        return score
