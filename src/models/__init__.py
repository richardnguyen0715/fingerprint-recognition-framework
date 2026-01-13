"""
Deep learning models for fingerprint recognition.

This module provides neural network architectures for:
1. Global embedding (CNN)
2. Patch-based features (around minutiae)
3. Hybrid approaches (CNN + classical)

Requirements:
- PyTorch (optional, for deep learning models)
- Models gracefully handle missing PyTorch

Usage:
------
from src.models import CNNEmbeddingMatcher, PatchCNNMatcher, HybridMatcher

# CNN embedding matcher
cnn_matcher = CNNEmbeddingMatcher()
score = cnn_matcher.match(img1, img2)

# Patch-based matcher
patch_matcher = PatchCNNMatcher()
score = patch_matcher.match(img1, img2)

# Hybrid matcher
hybrid_matcher = HybridMatcher()
score = hybrid_matcher.match(img1, img2)
"""

# Check for PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# CNN Embedding
from src.models.cnn_embedding import (
    CNNConfig,
    CNNEmbeddingMatcher,
    create_cnn_model,
    LossType
)

if TORCH_AVAILABLE:
    from src.models.cnn_embedding import (
        FingerprintCNN,
        ResNetEmbedding,
        ContrastiveLoss,
        TripletLoss,
        ArcFaceLoss,
        CNNTrainer,
        FingerprintPairDataset,
        FingerprintTripletDataset
    )

# Patch CNN
from src.models.patch_cnn import (
    PatchCNNConfig,
    PatchCNNMatcher,
    create_patch_cnn,
    extract_minutia_patch,
    extract_all_minutia_patches
)

if TORCH_AVAILABLE:
    from src.models.patch_cnn import (
        PatchCNN,
        PatchEncoder,
        AttentionAggregator,
        NetVLAD,
        PatchCNNTrainer,
        PatchPairDataset
    )

# Hybrid
from src.models.hybrid_model import (
    HybridConfig,
    HybridMatcher,
    HybridMatchingPipeline,
    ScoreFusion,
    FusionMethod,
    create_hybrid_matcher,
    create_enhancement_model
)

if TORCH_AVAILABLE:
    from src.models.hybrid_model import (
        EnhancementUNet,
        MinutiaeDetectionNet,
        LearnedFusion
    )

__all__ = [
    # Availability check
    "TORCH_AVAILABLE",
    
    # CNN Embedding
    "CNNConfig",
    "CNNEmbeddingMatcher",
    "create_cnn_model",
    "LossType",
    
    # Patch CNN
    "PatchCNNConfig",
    "PatchCNNMatcher",
    "create_patch_cnn",
    "extract_minutia_patch",
    "extract_all_minutia_patches",
    
    # Hybrid
    "HybridConfig",
    "HybridMatcher",
    "HybridMatchingPipeline",
    "ScoreFusion",
    "FusionMethod",
    "create_hybrid_matcher",
    "create_enhancement_model",
]

# Add PyTorch-only exports if available
if TORCH_AVAILABLE:
    __all__.extend([
        # CNN
        "FingerprintCNN",
        "ResNetEmbedding",
        "ContrastiveLoss",
        "TripletLoss",
        "ArcFaceLoss",
        "CNNTrainer",
        "FingerprintPairDataset",
        "FingerprintTripletDataset",
        
        # Patch
        "PatchCNN",
        "PatchEncoder",
        "AttentionAggregator",
        "NetVLAD",
        "PatchCNNTrainer",
        "PatchPairDataset",
        
        # Hybrid
        "EnhancementUNet",
        "MinutiaeDetectionNet",
        "LearnedFusion",
    ])
