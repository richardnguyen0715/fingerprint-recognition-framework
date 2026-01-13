"""
Descriptor-based fingerprint matching.

This module implements matching algorithms for descriptor-based
fingerprint recognition, supporting both MCC and local orientation
descriptors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from src.baselines.ssim import FingerprintMatcher
from src.minutiae.minutiae_extraction import Minutia
from src.descriptors.mcc import (
    MCCConfig,
    MCCDescriptor,
    cylinder_similarity,
    hamming_similarity
)
from src.descriptors.local_orientation_descriptor import (
    LODConfig,
    LocalOrientationDescriptor,
    descriptor_similarity
)


# =============================================================================
# MATHEMATICAL BACKGROUND
# =============================================================================
#
# Descriptor Matching:
# -------------------
# Unlike direct minutiae matching which requires global alignment,
# descriptor matching compares local feature representations.
#
# Local Similarity Sort (LSS):
# 1. Compute pairwise descriptor similarities
# 2. For each descriptor, find best match
# 3. Sort matches by similarity
# 4. Take top-k matches
# 5. Aggregate for final score
#
# Local Similarity Assignment with Relaxation (LSAR):
# 1. Initialize assignment matrix
# 2. Iteratively relax assignments
# 3. Use spatial consistency to reinforce good matches
#
# Reference:
# Cappelli, R., Ferrara, M., & Maltoni, D. (2010).
# "Minutia cylinder-code: A new representation and matching technique
# for fingerprint recognition." IEEE TPAMI, 32(12), 2128-2141.
# =============================================================================


def compute_similarity_matrix(
    descriptors1: List[Tuple[int, np.ndarray]],
    descriptors2: List[Tuple[int, np.ndarray]],
    similarity_func: callable
) -> np.ndarray:
    """
    Compute pairwise similarity matrix between two descriptor sets.
    
    Args:
        descriptors1: First set of (index, descriptor) tuples
        descriptors2: Second set of (index, descriptor) tuples
        similarity_func: Function to compute similarity between descriptors
        
    Returns:
        Similarity matrix of shape (len(desc1), len(desc2))
    """
    n1 = len(descriptors1)
    n2 = len(descriptors2)
    
    sim_matrix = np.zeros((n1, n2))
    
    for i, (_, d1) in enumerate(descriptors1):
        for j, (_, d2) in enumerate(descriptors2):
            sim_matrix[i, j] = similarity_func(d1, d2)
    
    return sim_matrix


def local_similarity_sort(
    sim_matrix: np.ndarray,
    num_top: int = 5
) -> float:
    """
    Local Similarity Sort (LSS) matching.
    
    Algorithm:
    ----------
    1. For each row (descriptor in set 1), find max similarity
    2. For each column (descriptor in set 2), find max similarity
    3. Average the top-k from both directions
    
    Args:
        sim_matrix: Pairwise similarity matrix
        num_top: Number of top matches to consider
        
    Returns:
        Global similarity score
    """
    n1, n2 = sim_matrix.shape
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Best match for each descriptor in set 1
    max_per_row = np.max(sim_matrix, axis=1)
    
    # Best match for each descriptor in set 2
    max_per_col = np.max(sim_matrix, axis=0)
    
    # Sort and take top-k
    top_row = np.sort(max_per_row)[::-1][:min(num_top, len(max_per_row))]
    top_col = np.sort(max_per_col)[::-1][:min(num_top, len(max_per_col))]
    
    # Average
    score = (np.mean(top_row) + np.mean(top_col)) / 2
    
    return score


def local_similarity_assignment_relaxation(
    sim_matrix: np.ndarray,
    minutiae1: List[Minutia],
    minutiae2: List[Minutia],
    desc_indices1: List[int],
    desc_indices2: List[int],
    num_iterations: int = 5,
    spatial_weight: float = 0.5
) -> float:
    """
    Local Similarity Assignment with Relaxation (LSAR).
    
    Uses spatial consistency to iteratively improve matching.
    
    Algorithm:
    ----------
    1. Initialize assignment from similarity matrix
    2. For each iteration:
       - Compute spatial consistency bonus
       - Update similarities
       - Recompute assignment
    3. Return final score
    
    Args:
        sim_matrix: Pairwise similarity matrix
        minutiae1: First minutiae set
        minutiae2: Second minutiae set
        desc_indices1: Indices of minutiae with descriptors (set 1)
        desc_indices2: Indices of minutiae with descriptors (set 2)
        num_iterations: Number of relaxation iterations
        spatial_weight: Weight for spatial consistency
        
    Returns:
        Global similarity score
    """
    n1, n2 = sim_matrix.shape
    
    if n1 == 0 or n2 == 0:
        return 0.0
    
    # Initialize assignment matrix
    assignment = sim_matrix.copy()
    
    # Compute pairwise distances in each minutiae set
    def compute_pairwise_distances(minutiae, indices):
        n = len(indices)
        dist = np.zeros((n, n))
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                mi = minutiae[idx_i]
                mj = minutiae[idx_j]
                dist[i, j] = np.sqrt((mi.x - mj.x)**2 + (mi.y - mj.y)**2)
        return dist
    
    dist1 = compute_pairwise_distances(minutiae1, desc_indices1)
    dist2 = compute_pairwise_distances(minutiae2, desc_indices2)
    
    # Relaxation iterations
    for _ in range(num_iterations):
        new_assignment = assignment.copy()
        
        for i in range(n1):
            for j in range(n2):
                # Spatial consistency: compare distance patterns
                spatial_bonus = 0.0
                count = 0
                
                for i2 in range(n1):
                    if i2 == i:
                        continue
                    for j2 in range(n2):
                        if j2 == j:
                            continue
                        
                        # If (i2, j2) is a good match
                        if assignment[i2, j2] > 0.5:
                            # Check distance consistency
                            d1 = dist1[i, i2]
                            d2 = dist2[j, j2]
                            
                            if d1 > 0 and d2 > 0:
                                dist_ratio = min(d1, d2) / max(d1, d2)
                                spatial_bonus += dist_ratio * assignment[i2, j2]
                                count += 1
                
                if count > 0:
                    spatial_bonus /= count
                
                # Update assignment
                new_assignment[i, j] = (
                    (1 - spatial_weight) * assignment[i, j] + 
                    spatial_weight * spatial_bonus
                )
        
        assignment = new_assignment
    
    # Compute final score from assignment matrix
    # Use greedy assignment
    score = 0.0
    used_rows = set()
    used_cols = set()
    
    # Sort all entries by value
    flat_indices = np.argsort(assignment.flatten())[::-1]
    
    for flat_idx in flat_indices:
        i = flat_idx // n2
        j = flat_idx % n2
        
        if i in used_rows or j in used_cols:
            continue
        
        score += assignment[i, j]
        used_rows.add(i)
        used_cols.add(j)
        
        if len(used_rows) >= min(n1, n2):
            break
    
    # Normalize by smaller set size
    score = score / min(n1, n2) if min(n1, n2) > 0 else 0.0
    
    return score


class MCCMatcher(FingerprintMatcher):
    """
    MCC descriptor-based fingerprint matcher.
    """
    
    def __init__(
        self,
        config: Optional[MCCConfig] = None,
        method: str = 'lss',
        num_top_pairs: int = 5,
        min_similarity: float = 0.3
    ):
        """
        Initialize MCC matcher.
        
        Args:
            config: MCC configuration
            method: Matching method ('lss' or 'lsar')
            num_top_pairs: Number of top pairs for LSS
            min_similarity: Minimum similarity threshold
        """
        self.config = config or MCCConfig()
        self.method = method
        self.num_top_pairs = num_top_pairs
        self.min_similarity = min_similarity
    
    @property
    def name(self) -> str:
        return "MCC"
    
    def compute_descriptors(self, minutiae: List[Minutia]) -> MCCDescriptor:
        """
        Compute MCC descriptors for minutiae set.
        
        Args:
            minutiae: List of minutiae
            
        Returns:
            MCCDescriptor instance
        """
        descriptor = MCCDescriptor(self.config)
        descriptor.compute(minutiae)
        return descriptor
    
    def match_descriptors(
        self,
        desc1: MCCDescriptor,
        desc2: MCCDescriptor
    ) -> float:
        """
        Match two sets of MCC descriptors.
        
        Args:
            desc1: First MCC descriptor set
            desc2: Second MCC descriptor set
            
        Returns:
            Similarity score
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        # Compute similarity matrix
        sim_matrix = compute_similarity_matrix(
            desc1.descriptors,
            desc2.descriptors,
            cylinder_similarity
        )
        
        if self.method == 'lss':
            score = local_similarity_sort(sim_matrix, self.num_top_pairs)
        elif self.method == 'lsar':
            indices1 = [idx for idx, _ in desc1.descriptors]
            indices2 = [idx for idx, _ in desc2.descriptors]
            score = local_similarity_assignment_relaxation(
                sim_matrix,
                desc1.minutiae, desc2.minutiae,
                indices1, indices2
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return score
    
    def match(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray
    ) -> float:
        """
        Match two fingerprints (placeholder for interface compatibility).
        
        Note: This requires pre-extracted minutiae. Use MCCMatchingPipeline
        for image-to-image matching.
        """
        raise NotImplementedError(
            "MCCMatcher.match() requires minutiae. "
            "Use MCCMatchingPipeline for image inputs."
        )


class LocalOrientationMatcher(FingerprintMatcher):
    """
    Local orientation descriptor-based matcher.
    """
    
    def __init__(
        self,
        config: Optional[LODConfig] = None,
        similarity_metric: str = 'cosine',
        num_top_pairs: int = 5
    ):
        """
        Initialize local orientation matcher.
        
        Args:
            config: LOD configuration
            similarity_metric: Metric for descriptor comparison
            num_top_pairs: Number of top pairs for scoring
        """
        self.config = config or LODConfig()
        self.similarity_metric = similarity_metric
        self.num_top_pairs = num_top_pairs
    
    @property
    def name(self) -> str:
        return "LocalOrientation"
    
    def compute_descriptors(
        self,
        orientation_field: np.ndarray,
        minutiae: List[Minutia]
    ) -> LocalOrientationDescriptor:
        """
        Compute local orientation descriptors.
        
        Args:
            orientation_field: Orientation field image
            minutiae: List of minutiae
            
        Returns:
            LocalOrientationDescriptor instance
        """
        descriptor = LocalOrientationDescriptor(self.config)
        descriptor.compute(orientation_field, minutiae)
        return descriptor
    
    def match_descriptors(
        self,
        desc1: LocalOrientationDescriptor,
        desc2: LocalOrientationDescriptor
    ) -> float:
        """
        Match two sets of local orientation descriptors.
        
        Args:
            desc1: First descriptor set
            desc2: Second descriptor set
            
        Returns:
            Similarity score
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        def sim_func(d1, d2):
            return descriptor_similarity(d1, d2, self.similarity_metric)
        
        sim_matrix = compute_similarity_matrix(
            desc1.descriptors,
            desc2.descriptors,
            sim_func
        )
        
        return local_similarity_sort(sim_matrix, self.num_top_pairs)
    
    def match(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray
    ) -> float:
        """Match two fingerprints (interface placeholder)."""
        raise NotImplementedError(
            "LocalOrientationMatcher requires orientation field and minutiae."
        )


class DescriptorMatchingPipeline:
    """
    Complete pipeline for descriptor-based fingerprint matching.
    """
    
    def __init__(
        self,
        descriptor_type: str = 'mcc',
        mcc_config: Optional[MCCConfig] = None,
        lod_config: Optional[LODConfig] = None,
        preprocessor=None,
        enhancer=None,
        thinner=None,
        extractor=None
    ):
        """
        Initialize descriptor matching pipeline.
        
        Args:
            descriptor_type: Type of descriptor ('mcc' or 'local_orientation')
            mcc_config: MCC configuration
            lod_config: LOD configuration
            preprocessor: FingerprintPreprocessor instance
            enhancer: GaborEnhancer instance
            thinner: Thinner instance
            extractor: MinutiaeExtractor instance
        """
        from src.minutiae.thinning import Thinner
        from src.minutiae.minutiae_extraction import MinutiaeExtractor
        
        self.descriptor_type = descriptor_type
        self.preprocessor = preprocessor
        self.enhancer = enhancer
        self.thinner = thinner or Thinner()
        self.extractor = extractor or MinutiaeExtractor()
        
        if descriptor_type == 'mcc':
            self.matcher = MCCMatcher(mcc_config)
        else:
            self.matcher = LocalOrientationMatcher(lod_config)
    
    def process_image(self, image: np.ndarray) -> Tuple[List[Minutia], np.ndarray]:
        """
        Process image to extract minutiae and orientation field.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Tuple of (minutiae, orientation_field)
        """
        from src.enhancement import estimate_orientation_field, RidgeFrequencyEstimator
        
        # Preprocess
        if self.preprocessor is not None:
            image, mask = self.preprocessor(image)
        else:
            mask = None
        
        # Estimate orientation
        orientation = estimate_orientation_field(image)
        
        # Enhance if configured
        if self.enhancer is not None:
            freq_estimator = RidgeFrequencyEstimator()
            frequency = freq_estimator.estimate(image, orientation, mask)
            image = self.enhancer.enhance(image, orientation, frequency, mask)
        
        # Thin and extract minutiae
        skeleton = self.thinner.process(image)
        minutiae = self.extractor.extract(skeleton, mask, orientation)
        
        return minutiae, orientation
    
    def match(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Match two fingerprint images using descriptors.
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            
        Returns:
            Similarity score
        """
        minutiae1, orientation1 = self.process_image(image1)
        minutiae2, orientation2 = self.process_image(image2)
        
        if self.descriptor_type == 'mcc':
            desc1 = self.matcher.compute_descriptors(minutiae1)
            desc2 = self.matcher.compute_descriptors(minutiae2)
        else:
            desc1 = self.matcher.compute_descriptors(orientation1, minutiae1)
            desc2 = self.matcher.compute_descriptors(orientation2, minutiae2)
        
        return self.matcher.match_descriptors(desc1, desc2)
