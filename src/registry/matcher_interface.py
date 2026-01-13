"""
Matcher interface definitions.

This module defines the base interface for all fingerprint matchers,
providing a standardized API for the UI layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MatchResult:
    """
    Result of a fingerprint matching operation.
    
    Attributes:
        score: Similarity score between 0 and 1 (higher = more similar)
        details: Dictionary containing detailed matching information
        visualization_data: Optional data for generating visualizations
    """
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    visualization_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "details": self.details,
            "visualization_data": self.visualization_data,
        }


class BaseMatcher(ABC):
    """
    Abstract base class for fingerprint matchers.
    
    All matchers registered with the UI must implement this interface.
    This provides a consistent API for matching and explanation.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the display name of the matcher.
        
        Returns:
            Human-readable name for UI display
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Return a description of the matcher.
        
        Returns:
            Description string for UI display
        """
        return ""
    
    @abstractmethod
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """
        Compute similarity between two fingerprint images.
        
        Args:
            image_a: First fingerprint image (grayscale, normalized to [0,1])
            image_b: Second fingerprint image (grayscale, normalized to [0,1])
            
        Returns:
            MatchResult containing score and detailed information
        """
        pass
    
    def explain(self) -> Dict[str, Any]:
        """
        Return explanation of the matcher's algorithm.
        
        Returns:
            Dictionary containing algorithm explanation and metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_current_parameters(),
        }
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get current parameter values.
        
        Returns:
            Dictionary of parameter names to current values
        """
        return {}
    
    def set_parameters(self, **kwargs) -> None:
        """
        Set matcher parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        pass
