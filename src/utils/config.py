"""
Configuration management for fingerprint recognition framework.

This module provides utilities for loading, validating, and accessing
configuration parameters from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    image_size: tuple = (256, 256)
    normalize: bool = True
    normalize_range: tuple = (0.0, 1.0)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation protocol."""
    num_impostor_per_genuine: int = 5
    random_seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data paths."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    pairs_dir: str = "data/pairs"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    log_dir: str = "logs"
    save_results: bool = True


@dataclass
class Config:
    """
    Main configuration container for the fingerprint recognition framework.
    
    Attributes:
        data: Data path configurations
        preprocessing: Image preprocessing settings
        evaluation: Evaluation protocol settings
        logging: Logging configuration
        model_config: Model-specific configuration dictionary
    """
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model_config: Dict[str, Any] = field(default_factory=dict)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    The override dictionary values take precedence over base values.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None
) -> Config:
    """
    Load configuration from YAML files.
    
    Optionally merges with a base configuration file.
    
    Args:
        config_path: Path to the main configuration file
        base_config_path: Optional path to base configuration to merge with
        
    Returns:
        Config object with loaded settings
    """
    config_dict = load_yaml(config_path)
    
    if base_config_path is not None:
        base_dict = load_yaml(base_config_path)
        config_dict = merge_configs(base_dict, config_dict)
    
    # Extract standard configuration sections
    data_dict = config_dict.pop('data', {})
    preprocessing_dict = config_dict.pop('preprocessing', {})
    evaluation_dict = config_dict.pop('evaluation', {})
    logging_dict = config_dict.pop('logging', {})
    
    # Build configuration objects
    data_config = DataConfig(
        raw_dir=data_dict.get('raw_dir', 'data/raw'),
        processed_dir=data_dict.get('processed_dir', 'data/processed'),
        pairs_dir=data_dict.get('pairs_dir', 'data/pairs')
    )
    
    preprocessing_config = PreprocessingConfig(
        image_size=tuple(preprocessing_dict.get('image_size', [256, 256])),
        normalize=preprocessing_dict.get('normalize', True),
        normalize_range=tuple(preprocessing_dict.get('normalize_range', [0.0, 1.0]))
    )
    
    evaluation_config = EvaluationConfig(
        num_impostor_per_genuine=evaluation_dict.get('num_impostor_per_genuine', 5),
        random_seed=evaluation_dict.get('random_seed', 42)
    )
    
    logging_config = LoggingConfig(
        level=logging_dict.get('level', 'INFO'),
        log_dir=logging_dict.get('log_dir', 'logs'),
        save_results=logging_dict.get('save_results', True)
    )
    
    return Config(
        data=data_config,
        preprocessing=preprocessing_config,
        evaluation=evaluation_config,
        logging=logging_config,
        model_config=config_dict
    )


def get_model_config(config: Config, key: str, default: Any = None) -> Any:
    """
    Get a model-specific configuration value.
    
    Args:
        config: Configuration object
        key: Dot-separated key path (e.g., 'ssim.window_size')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key.split('.')
    value = config.model_config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


# Default configuration instance
DEFAULT_CONFIG = Config()
