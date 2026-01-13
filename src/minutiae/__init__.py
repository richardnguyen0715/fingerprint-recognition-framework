"""
Minutiae-based fingerprint recognition modules.

This package provides classical minutiae extraction and matching:
- Thinning (skeletonization)
- Minutiae extraction (crossing number method)
- Minutiae matching (alignment + correspondence)
"""

from .thinning import (
    zhang_suen_thinning,
    guo_hall_thinning,
    binarize_image,
    thin_fingerprint,
    Thinner
)
from .minutiae_extraction import (
    MinutiaeType,
    Minutia,
    compute_crossing_number,
    estimate_minutia_orientation,
    extract_minutiae,
    remove_spurious_minutiae,
    filter_by_mask,
    limit_minutiae_count,
    MinutiaeExtractor
)
from .minutiae_matching import (
    MinutiaeMatch,
    compute_transformation,
    apply_transformation,
    find_matching_pairs,
    ransac_alignment,
    compute_matching_score,
    MinutiaeMatcher,
    MinutiaeMatchingPipeline
)

__all__ = [
    # Thinning
    'zhang_suen_thinning',
    'guo_hall_thinning',
    'binarize_image',
    'thin_fingerprint',
    'Thinner',
    # Minutiae extraction
    'MinutiaeType',
    'Minutia',
    'compute_crossing_number',
    'estimate_minutia_orientation',
    'extract_minutiae',
    'remove_spurious_minutiae',
    'filter_by_mask',
    'limit_minutiae_count',
    'MinutiaeExtractor',
    # Minutiae matching
    'MinutiaeMatch',
    'compute_transformation',
    'apply_transformation',
    'find_matching_pairs',
    'ransac_alignment',
    'compute_matching_score',
    'MinutiaeMatcher',
    'MinutiaeMatchingPipeline',
]
