"""
Data handling modules for fingerprint recognition.
"""

from .fingerprint_dataset import (
    FingerprintDataset,
    FingerprintSample,
    DatasetType,
    load_fvc2004_dataset,
    load_neurotech_dataset
)
from .preprocessing import (
    normalize_image,
    normalize_to_range,
    segment_fingerprint,
    adaptive_histogram_equalization,
    remove_background_gradient,
    resize_image,
    pad_to_square,
    preprocess_fingerprint,
    FingerprintPreprocessor
)
from .pair_generator import (
    VerificationPair,
    PairGenerator,
    PairIterator,
    save_pairs_to_csv,
    load_pairs_from_csv,
    get_pair_statistics
)

__all__ = [
    # Dataset
    'FingerprintDataset',
    'FingerprintSample',
    'DatasetType',
    'load_fvc2004_dataset',
    'load_neurotech_dataset',
    # Preprocessing
    'normalize_image',
    'normalize_to_range',
    'segment_fingerprint',
    'adaptive_histogram_equalization',
    'remove_background_gradient',
    'resize_image',
    'pad_to_square',
    'preprocess_fingerprint',
    'FingerprintPreprocessor',
    # Pair generation
    'VerificationPair',
    'PairGenerator',
    'PairIterator',
    'save_pairs_to_csv',
    'load_pairs_from_csv',
    'get_pair_statistics',
]
