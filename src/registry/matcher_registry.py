"""
Matcher registry for dynamic model discovery.

This module provides a registry system that allows models to be
registered and discovered dynamically, without hardcoding model
names in the UI layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from src.registry.matcher_interface import BaseMatcher


class ParameterType(Enum):
    """Types of configurable parameters."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    SELECT = "select"


@dataclass
class ParameterInfo:
    """
    Metadata for a configurable parameter.
    
    Attributes:
        name: Parameter identifier
        display_name: Human-readable name
        param_type: Type of the parameter
        default: Default value
        description: Parameter description
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        step: Step size (for numeric types)
        options: Available options (for SELECT type)
    """
    name: str
    display_name: str
    param_type: ParameterType
    default: Any
    description: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    options: Optional[List[str]] = None


@dataclass
class MatcherInfo:
    """
    Information about a registered matcher.
    
    Attributes:
        id: Unique identifier for the matcher
        name: Display name
        description: Description of the algorithm
        category: Category (e.g., "baseline", "minutiae", "descriptor", "cnn")
        parameters: List of configurable parameters
        factory: Factory function to create the matcher
        requires_preprocessing: Whether preprocessing is required
    """
    id: str
    name: str
    description: str
    category: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    factory: Optional[Callable[..., BaseMatcher]] = None
    requires_preprocessing: bool = True


class MatcherRegistry:
    """
    Central registry for fingerprint matchers.
    
    Provides methods to register, discover, and instantiate matchers.
    This enables the UI to dynamically list available models without
    hardcoding them.
    """
    
    _instance: Optional["MatcherRegistry"] = None
    
    def __new__(cls) -> "MatcherRegistry":
        """Singleton pattern to ensure a single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._matchers = {}
            cls._instance._initialized = False
        return cls._instance
    
    def register(
        self,
        matcher_id: str,
        name: str,
        description: str,
        category: str,
        factory: Callable[..., BaseMatcher],
        parameters: Optional[List[ParameterInfo]] = None,
        requires_preprocessing: bool = True,
    ) -> None:
        """
        Register a matcher with the registry.
        
        Args:
            matcher_id: Unique identifier
            name: Display name
            description: Algorithm description
            category: Matcher category
            factory: Factory function to create instances
            parameters: List of configurable parameters
            requires_preprocessing: Whether preprocessing is needed
        """
        info = MatcherInfo(
            id=matcher_id,
            name=name,
            description=description,
            category=category,
            parameters=parameters or [],
            factory=factory,
            requires_preprocessing=requires_preprocessing,
        )
        self._matchers[matcher_id] = info
    
    def register_matcher_class(
        self,
        matcher_class: Type[BaseMatcher],
        matcher_id: str,
        category: str,
        parameters: Optional[List[ParameterInfo]] = None,
        requires_preprocessing: bool = True,
    ) -> None:
        """
        Register a matcher class directly.
        
        Args:
            matcher_class: The matcher class to register
            matcher_id: Unique identifier
            category: Matcher category
            parameters: Configurable parameters
            requires_preprocessing: Whether preprocessing is needed
        """
        # Create a factory that instantiates the class
        def factory(**kwargs) -> BaseMatcher:
            return matcher_class(**kwargs)
        
        # Get name and description from class
        temp_instance = matcher_class()
        name = temp_instance.name
        description = temp_instance.description
        
        self.register(
            matcher_id=matcher_id,
            name=name,
            description=description,
            category=category,
            factory=factory,
            parameters=parameters,
            requires_preprocessing=requires_preprocessing,
        )
    
    def unregister(self, matcher_id: str) -> None:
        """
        Remove a matcher from the registry.
        
        Args:
            matcher_id: ID of matcher to remove
        """
        if matcher_id in self._matchers:
            del self._matchers[matcher_id]
    
    def get_matcher_info(self, matcher_id: str) -> Optional[MatcherInfo]:
        """
        Get information about a specific matcher.
        
        Args:
            matcher_id: Matcher identifier
            
        Returns:
            MatcherInfo or None if not found
        """
        return self._matchers.get(matcher_id)
    
    def list_matchers(self) -> List[MatcherInfo]:
        """
        List all registered matchers.
        
        Returns:
            List of MatcherInfo for all registered matchers
        """
        return list(self._matchers.values())
    
    def list_by_category(self, category: str) -> List[MatcherInfo]:
        """
        List matchers in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of MatcherInfo in the specified category
        """
        return [m for m in self._matchers.values() if m.category == category]
    
    def get_categories(self) -> List[str]:
        """
        Get all unique categories.
        
        Returns:
            List of category names
        """
        return list(set(m.category for m in self._matchers.values()))
    
    def create_matcher(
        self,
        matcher_id: str,
        **kwargs,
    ) -> Optional[BaseMatcher]:
        """
        Create a matcher instance.
        
        Args:
            matcher_id: ID of matcher to create
            **kwargs: Parameters to pass to the factory
            
        Returns:
            BaseMatcher instance or None if matcher not found
        """
        info = self._matchers.get(matcher_id)
        if info is None or info.factory is None:
            return None
        
        return info.factory(**kwargs)
    
    def is_registered(self, matcher_id: str) -> bool:
        """
        Check if a matcher is registered.
        
        Args:
            matcher_id: Matcher identifier
            
        Returns:
            True if registered, False otherwise
        """
        return matcher_id in self._matchers
    
    def clear(self) -> None:
        """Clear all registered matchers."""
        self._matchers.clear()
        self._initialized = False


def get_registry() -> MatcherRegistry:
    """
    Get the global matcher registry instance.
    
    Returns:
        The singleton MatcherRegistry instance
    """
    registry = MatcherRegistry()
    
    # Initialize with default matchers if not already done
    if not registry._initialized:
        _register_default_matchers(registry)
        registry._initialized = True
    
    return registry


def _register_default_matchers(registry: MatcherRegistry) -> None:
    """
    Register all default matchers with the registry.
    
    This function imports and registers all built-in matchers.
    
    Args:
        registry: The registry to populate
    """
    # Import adapter modules which register themselves
    from src.registry import adapters  # noqa: F401
