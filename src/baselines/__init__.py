"""
Baseline matchers for fingerprint recognition.
"""

from .ssim import (
    FingerprintMatcher,
    MSEMatcher,
    NCCMatcher,
    SSIMMatcher,
    mse_score,
    ncc_score,
    ssim_score,
    get_baseline_matcher
)

__all__ = [
    'FingerprintMatcher',
    'MSEMatcher',
    'NCCMatcher',
    'SSIMMatcher',
    'mse_score',
    'ncc_score',
    'ssim_score',
    'get_baseline_matcher',
]
