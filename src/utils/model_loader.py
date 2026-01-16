"""
Model weight loader utilities for deep learning models.

This module provides utilities for loading, managing, and discovering
pretrained model weights for fingerprint recognition.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


def get_models_dir() -> Path:
    """
    Get the models checkpoint directory.
    
    Returns:
        Path to models/checkpoints directory
    """
    # Get project root (assumes this file is in src/utils/)
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "models" / "checkpoints"
    
    # Create directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    return models_dir


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available model weights.
    
    Returns:
        Dictionary mapping model type to list of checkpoint files
    """
    models_dir = get_models_dir()
    
    available_models = {
        "cnn_embedding": [],
        "patch_cnn": [],
        "hybrid": [],
    }
    
    # Scan for .pth files
    for model_type in available_models.keys():
        type_dir = models_dir / model_type
        if type_dir.exists():
            for checkpoint_file in type_dir.glob("*.pth"):
                available_models[model_type].append(str(checkpoint_file))
    
    return available_models


def find_model_path(model_type: str, checkpoint_name: Optional[str] = None) -> Optional[str]:
    """
    Find path to a model checkpoint.
    
    Args:
        model_type: Type of model ("cnn_embedding", "patch_cnn", "hybrid")
        checkpoint_name: Name of checkpoint file (if None, returns latest)
        
    Returns:
        Path to checkpoint file, or None if not found
    """
    models_dir = get_models_dir()
    type_dir = models_dir / model_type
    
    if not type_dir.exists():
        return None
    
    if checkpoint_name is None:
        # Find latest checkpoint
        checkpoints = list(type_dir.glob("*.pth"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    else:
        # Find specific checkpoint
        checkpoint_path = type_dir / checkpoint_name
        if checkpoint_path.exists():
            return str(checkpoint_path)
        else:
            return None


def load_model_metadata(model_path: str) -> Dict:
    """
    Load metadata associated with a model checkpoint.
    
    Args:
        model_path: Path to model .pth file
        
    Returns:
        Dictionary containing metadata (empty if no metadata file exists)
    """
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix('.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def save_model_metadata(model_path: str, metadata: Dict) -> None:
    """
    Save metadata for a model checkpoint.
    
    Args:
        model_path: Path to model .pth file
        metadata: Dictionary containing metadata to save
    """
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix('.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_model_info(model_path: str) -> Tuple[Optional[Dict], str]:
    """
    Get information about a model checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Tuple of (metadata_dict, status_message)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return None, f"Model file not found: {model_path}"
    
    # Get file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    
    # Load metadata
    metadata = load_model_metadata(str(model_path))
    
    info = {
        "path": str(model_path),
        "filename": model_path.name,
        "size_mb": round(size_mb, 2),
        **metadata,
    }
    
    return info, "Model found"


def validate_model_path(model_path: str) -> Tuple[bool, str]:
    """
    Validate a model checkpoint path.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not model_path:
        return False, "No model path provided"
    
    path = Path(model_path)
    
    if not path.exists():
        return False, f"Model file does not exist: {model_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {model_path}"
    
    if path.suffix not in ['.pth', '.pt']:
        return False, f"Invalid model file extension: {path.suffix} (expected .pth or .pt)"
    
    # Check file size (should be at least 1KB)
    if path.stat().st_size < 1024:
        return False, f"Model file is too small (possibly corrupted): {path.stat().st_size} bytes"
    
    return True, "Model file is valid"


def create_checkpoint_structure() -> Dict[str, Path]:
    """
    Create the checkpoint directory structure.
    
    Returns:
        Dictionary mapping model types to their directories
    """
    models_dir = get_models_dir()
    
    structure = {
        "cnn_embedding": models_dir / "cnn_embedding",
        "patch_cnn": models_dir / "patch_cnn",
        "hybrid": models_dir / "hybrid",
    }
    
    # Create all directories
    for model_type, path in structure.items():
        path.mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme_path = path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"# {model_type.replace('_', ' ').title()} Checkpoints\n\n")
                f.write(f"Place your trained {model_type} model checkpoints (.pth files) here.\n\n")
                f.write("Metadata files (.json) with the same name will be automatically loaded.\n")
    
    return structure


# Initialize checkpoint structure on module import
create_checkpoint_structure()
