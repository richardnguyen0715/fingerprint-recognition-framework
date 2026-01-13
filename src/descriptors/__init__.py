"""
Descriptor-based fingerprint recognition modules.

This package provides descriptor extraction and matching:
- Minutia Cylinder Code (MCC)
- Local orientation descriptors
- Descriptor matching algorithms
"""

from .mcc import (
    MCCConfig,
    MCCDescriptor,
    compute_mcc_cylinder,
    compute_mcc_descriptors,
    binarize_cylinder,
    cylinder_similarity,
    hamming_similarity
)
from .local_orientation_descriptor import (
    LODConfig,
    LocalOrientationDescriptor,
    extract_aligned_patch,
    compute_orientation_histogram,
    compute_local_orientation_descriptor,
    compute_local_descriptors,
    descriptor_distance,
    descriptor_similarity
)
from .descriptor_matching import (
    compute_similarity_matrix,
    local_similarity_sort,
    local_similarity_assignment_relaxation,
    MCCMatcher,
    LocalOrientationMatcher,
    DescriptorMatchingPipeline
)

__all__ = [
    # MCC
    'MCCConfig',
    'MCCDescriptor',
    'compute_mcc_cylinder',
    'compute_mcc_descriptors',
    'binarize_cylinder',
    'cylinder_similarity',
    'hamming_similarity',
    # Local orientation
    'LODConfig',
    'LocalOrientationDescriptor',
    'extract_aligned_patch',
    'compute_orientation_histogram',
    'compute_local_orientation_descriptor',
    'compute_local_descriptors',
    'descriptor_distance',
    'descriptor_similarity',
    # Matching
    'compute_similarity_matrix',
    'local_similarity_sort',
    'local_similarity_assignment_relaxation',
    'MCCMatcher',
    'LocalOrientationMatcher',
    'DescriptorMatchingPipeline',
]
