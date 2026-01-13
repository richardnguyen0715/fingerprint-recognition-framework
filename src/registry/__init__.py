"""
Model registry for fingerprint matchers.

This module provides a centralized registry for fingerprint matching models,
enabling dynamic discovery and instantiation of matchers for the UI layer.
"""

from src.registry.matcher_registry import (
    MatcherRegistry,
    MatcherInfo,
    ParameterInfo,
    ParameterType,
    get_registry,
)
from src.registry.matcher_interface import (
    BaseMatcher,
    MatchResult,
)

__all__ = [
    "MatcherRegistry",
    "MatcherInfo",
    "ParameterInfo",
    "ParameterType",
    "get_registry",
    "BaseMatcher",
    "MatchResult",
]
