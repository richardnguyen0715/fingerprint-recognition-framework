"""
Fingerprint enhancement modules.

This package provides classical fingerprint enhancement algorithms:
- Orientation field estimation
- Ridge frequency estimation
- Gabor filter-based enhancement
"""

from .orientation_field import (
    compute_gradients,
    estimate_orientation_block,
    smooth_orientation_field,
    resize_orientation_field,
    estimate_orientation_field,
    compute_coherence,
    OrientationFieldEstimator
)
from .ridge_frequency import (
    estimate_frequency_block,
    estimate_frequency_field,
    interpolate_frequency,
    smooth_frequency_field,
    resize_frequency_field,
    RidgeFrequencyEstimator
)
from .gabor_filter import (
    create_gabor_kernel,
    create_gabor_filter_bank,
    apply_gabor_filter,
    apply_adaptive_gabor_filter,
    GaborEnhancer,
    enhance_fingerprint
)

__all__ = [
    # Orientation field
    'compute_gradients',
    'estimate_orientation_block',
    'smooth_orientation_field',
    'resize_orientation_field',
    'estimate_orientation_field',
    'compute_coherence',
    'OrientationFieldEstimator',
    # Ridge frequency
    'estimate_frequency_block',
    'estimate_frequency_field',
    'interpolate_frequency',
    'smooth_frequency_field',
    'resize_frequency_field',
    'RidgeFrequencyEstimator',
    # Gabor filter
    'create_gabor_kernel',
    'create_gabor_filter_bank',
    'apply_gabor_filter',
    'apply_adaptive_gabor_filter',
    'GaborEnhancer',
    'enhance_fingerprint',
]
