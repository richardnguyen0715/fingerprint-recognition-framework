"""
Deep Learning matcher adapters for the UI layer.

This module provides adapter classes that wrap deep learning fingerprint
matching models to conform to the BaseMatcher interface required by the UI.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.registry.matcher_interface import BaseMatcher, MatchResult


# =============================================================================
# CNN EMBEDDING ADAPTER
# =============================================================================


class CNNEmbeddingMatcherAdapter(BaseMatcher):
    """
    Adapter for CNN embedding matcher.
    
    Wraps the CNNEmbeddingMatcher from src.models.cnn_embedding to provide
    the BaseMatcher interface for the UI.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        model_path: Optional[str] = None,
        backbone: str = "custom",
        device: str = "cpu",
    ):
        """
        Initialize CNN embedding matcher adapter.
        
        Args:
            embedding_dim: Dimension of embedding vector
            model_path: Path to pretrained model weights (.pth file)
            backbone: Network backbone ("custom", "resnet18", "resnet34", "resnet50")
            device: Computation device ("cpu" or "cuda")
        """
        from src.models.cnn_embedding import CNNEmbeddingMatcher, CNNConfig
        
        self._embedding_dim = embedding_dim
        self._model_path = model_path
        self._backbone = backbone
        self._device = device
        
        # Create configuration
        config = CNNConfig(
            embedding_dim=embedding_dim,
        )
        
        # Initialize matcher
        try:
            self._matcher = CNNEmbeddingMatcher(
                config=config,
                model_path=model_path,
                device=device,
            )
            self._model_loaded = model_path is not None
        except Exception as e:
            # If model loading fails, store error but don't crash
            self._matcher = None
            self._error = str(e)
            self._model_loaded = False
    
    @property
    def name(self) -> str:
        return "CNN Embedding"
    
    @property
    def description(self) -> str:
        return (
            "Deep learning model that learns discriminative embeddings for "
            "fingerprints. Compares fingerprints by computing cosine similarity "
            "between learned embedding vectors. Requires pretrained weights."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute CNN embedding similarity between two fingerprints."""
        if self._matcher is None:
            return MatchResult(
                score=0.0,
                details={
                    "error": f"Model initialization failed: {self._error}",
                    "model_loaded": False,
                },
            )
        
        if not self._model_loaded:
            return MatchResult(
                score=0.0,
                details={
                    "error": "No pretrained model loaded. Please upload model weights.",
                    "model_loaded": False,
                },
            )
        
        try:
            # Extract embeddings
            embedding_a = self._matcher.extract_embedding(image_a)
            embedding_b = self._matcher.extract_embedding(image_b)
            
            # Compute cosine similarity
            similarity = float(np.dot(embedding_a, embedding_b))
            
            # Convert to [0, 1] range (cosine is in [-1, 1])
            score = (similarity + 1.0) / 2.0
            
            # Compute Euclidean distance for additional info
            euclidean_dist = float(np.linalg.norm(embedding_a - embedding_b))
            
            details = {
                "cosine_similarity": similarity,
                "euclidean_distance": euclidean_dist,
                "embedding_dim": self._embedding_dim,
                "embedding_norm_a": float(np.linalg.norm(embedding_a)),
                "embedding_norm_b": float(np.linalg.norm(embedding_b)),
                "model_loaded": True,
            }
            
            return MatchResult(
                score=score,
                details=details,
            )
            
        except Exception as e:
            return MatchResult(
                score=0.0,
                details={
                    "error": f"Matching failed: {str(e)}",
                    "model_loaded": self._model_loaded,
                },
            )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self._embedding_dim,
            "model_path": self._model_path or "",
            "backbone": self._backbone,
            "device": self._device,
        }


# =============================================================================
# PATCH CNN ADAPTER
# =============================================================================


class PatchCNNMatcherAdapter(BaseMatcher):
    """
    Adapter for Patch CNN matcher.
    
    Wraps the PatchCNNMatcher from src.models.patch_cnn to provide
    the BaseMatcher interface for the UI.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        patch_size: int = 64,
        model_path: Optional[str] = None,
        aggregation: str = "attention",
        device: str = "cpu",
    ):
        """
        Initialize Patch CNN matcher adapter.
        
        Args:
            embedding_dim: Dimension of patch embeddings
            patch_size: Size of patches around minutiae
            model_path: Path to pretrained model weights
            aggregation: Aggregation method ("attention", "mean", "max", "netvlad")
            device: Computation device
        """
        from src.models.patch_cnn import PatchCNNMatcher, PatchCNNConfig
        
        self._embedding_dim = embedding_dim
        self._patch_size = patch_size
        self._model_path = model_path
        self._aggregation = aggregation
        self._device = device
        
        # Create configuration
        config = PatchCNNConfig(
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            aggregation_method=aggregation,
        )
        
        # Initialize matcher
        try:
            self._matcher = PatchCNNMatcher(
                config=config,
                model_path=model_path,
                device=device,
            )
            self._model_loaded = model_path is not None
        except Exception as e:
            self._matcher = None
            self._error = str(e)
            self._model_loaded = False
    
    @property
    def name(self) -> str:
        return "Patch CNN"
    
    @property
    def description(self) -> str:
        return (
            "Combines classical minutiae extraction with deep learning. "
            "Extracts patches around minutiae points and encodes them using CNN. "
            "Aggregates patch embeddings for fingerprint-level comparison."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute Patch CNN similarity between two fingerprints."""
        if self._matcher is None:
            return MatchResult(
                score=0.0,
                details={
                    "error": f"Model initialization failed: {self._error}",
                    "model_loaded": False,
                },
            )
        
        if not self._model_loaded:
            return MatchResult(
                score=0.0,
                details={
                    "error": "No pretrained model loaded. Please upload model weights.",
                    "model_loaded": False,
                },
            )
        
        try:
            # Perform matching
            score = self._matcher.match(image_a, image_b)
            
            # Extract additional information
            patches_a = self._matcher.extract_patches_from_image(image_a)
            patches_b = self._matcher.extract_patches_from_image(image_b)
            
            details = {
                "similarity_score": float(score),
                "num_patches_a": len(patches_a),
                "num_patches_b": len(patches_b),
                "patch_size": self._patch_size,
                "embedding_dim": self._embedding_dim,
                "aggregation_method": self._aggregation,
                "model_loaded": True,
            }
            
            # Normalize score to [0, 1] if needed
            if score < 0:
                score = (score + 1.0) / 2.0
            
            return MatchResult(
                score=float(score),
                details=details,
            )
            
        except Exception as e:
            return MatchResult(
                score=0.0,
                details={
                    "error": f"Matching failed: {str(e)}",
                    "model_loaded": self._model_loaded,
                },
            )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self._embedding_dim,
            "patch_size": self._patch_size,
            "model_path": self._model_path or "",
            "aggregation": self._aggregation,
            "device": self._device,
        }


# =============================================================================
# HYBRID MODEL ADAPTER
# =============================================================================


class HybridMatcherAdapter(BaseMatcher):
    """
    Adapter for Hybrid matcher.
    
    Combines CNN-based enhancement/minutiae detection with classical
    matching methods for robust fingerprint verification.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        fusion_method: str = "weighted",
        cnn_weight: float = 0.5,
        use_enhancement: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize Hybrid matcher adapter.
        
        Args:
            model_path: Path to pretrained model weights
            fusion_method: Score fusion method ("mean", "weighted", "max", "min", "product")
            cnn_weight: Weight for CNN scores in weighted fusion
            use_enhancement: Whether to use CNN enhancement
            device: Computation device
        """
        self._model_path = model_path
        self._fusion_method = fusion_method
        self._cnn_weight = cnn_weight
        self._use_enhancement = use_enhancement
        self._device = device
        self._model_loaded = False
        
        # Note: Hybrid model implementation may vary based on the actual code
        # This is a placeholder that can be adapted to the actual implementation
        self._matcher = None
        self._error = "Hybrid model not fully implemented yet"
    
    @property
    def name(self) -> str:
        return "Hybrid CNN + Classical"
    
    @property
    def description(self) -> str:
        return (
            "Hybrid approach combining CNN-based fingerprint enhancement "
            "or minutiae detection with classical matching algorithms. "
            "Fuses multiple scores for robust verification."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute hybrid similarity between two fingerprints."""
        # For now, return a placeholder indicating the model needs implementation
        return MatchResult(
            score=0.0,
            details={
                "error": "Hybrid model adapter not fully implemented. Coming soon!",
                "model_loaded": False,
                "fusion_method": self._fusion_method,
                "cnn_weight": self._cnn_weight,
            },
        )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {
            "model_path": self._model_path or "",
            "fusion_method": self._fusion_method,
            "cnn_weight": self._cnn_weight,
            "use_enhancement": self._use_enhancement,
            "device": self._device,
        }
