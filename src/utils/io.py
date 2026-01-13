"""
I/O utilities for fingerprint recognition framework.

Provides functions for loading, saving, and managing fingerprint
images and related data structures.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def load_image(
    path: Union[str, Path],
    grayscale: bool = True,
    normalize: bool = False,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from disk.
    
    Args:
        path: Path to the image file
        grayscale: Whether to load as grayscale
        normalize: Whether to normalize to [0, 1] range
        target_size: Optional (width, height) to resize to
        
    Returns:
        Image as numpy array
        
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image cannot be loaded
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    normalize_output: bool = True
) -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array
        path: Output path
        normalize_output: If True and image is float, scale to [0, 255]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize_output and image.dtype in [np.float32, np.float64]:
        # Handle both [0, 1] and [-1, 1] ranges
        if image.min() < 0:
            image = (image + 1) / 2
        image = (image * 255).clip(0, 255).astype(np.uint8)
    
    cv2.imwrite(str(path), image)


def load_pair_list(
    csv_path: Union[str, Path]
) -> List[Tuple[str, str, int]]:
    """
    Load a list of image pairs from a CSV file.
    
    CSV format: img1,img2,label (1=genuine, 0=impostor)
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['img1'], row['img2'], int(row['label'])))
    return pairs


def save_pair_list(
    pairs: List[Tuple[str, str, int]],
    csv_path: Union[str, Path]
) -> None:
    """
    Save a list of image pairs to a CSV file.
    
    Args:
        pairs: List of (img1_path, img2_path, label) tuples
        csv_path: Output path
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img1', 'img2', 'label'])
        for pair in pairs:
            writer.writerow(pair)


def discover_images(
    directory: Union[str, Path],
    extensions: Optional[set] = None,
    recursive: bool = True
) -> List[Path]:
    """
    Discover all images in a directory.
    
    Args:
        directory: Root directory to search
        extensions: Set of valid extensions (default: SUPPORTED_EXTENSIONS)
        recursive: Whether to search subdirectories
        
    Returns:
        List of paths to discovered images
    """
    directory = Path(directory)
    extensions = extensions or SUPPORTED_EXTENSIONS
    
    images = []
    pattern = '**/*' if recursive else '*'
    
    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            images.append(path)
    
    return sorted(images)


def parse_fvc_filename(filename: str) -> Tuple[int, int]:
    """
    Parse FVC-style filename to extract subject and sample IDs.
    
    FVC naming convention: {subject_id}_{sample_id}.tif
    Example: "101_1.tif" -> subject 101, sample 1
    
    Args:
        filename: Filename to parse
        
    Returns:
        Tuple of (subject_id, sample_id)
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    
    raise ValueError(f"Cannot parse FVC filename: {filename}")


def group_images_by_subject(
    images: List[Path],
    filename_parser: callable = parse_fvc_filename
) -> Dict[int, List[Path]]:
    """
    Group images by subject ID.
    
    Args:
        images: List of image paths
        filename_parser: Function to extract subject ID from filename
        
    Returns:
        Dictionary mapping subject ID to list of image paths
    """
    groups: Dict[int, List[Path]] = {}
    
    for img_path in images:
        try:
            subject_id, _ = filename_parser(img_path.name)
            if subject_id not in groups:
                groups[subject_id] = []
            groups[subject_id].append(img_path)
        except ValueError:
            continue
    
    return groups


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_minutiae(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load minutiae from a JSON file.
    
    Expected format: List of dicts with 'x', 'y', 'angle', 'type' keys.
    
    Args:
        path: Path to minutiae file
        
    Returns:
        List of minutiae dictionaries
    """
    return load_json(path)


def save_minutiae(
    minutiae: List[Dict[str, Any]],
    path: Union[str, Path]
) -> None:
    """
    Save minutiae to a JSON file.
    
    Args:
        minutiae: List of minutiae dictionaries
        path: Output path
    """
    save_json(minutiae, path)


class ImageCache:
    """
    LRU cache for loaded images to avoid repeated disk reads.
    
    Useful when the same images are accessed multiple times
    during pair generation or evaluation.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the image cache.
        
        Args:
            max_size: Maximum number of images to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
    
    def get(
        self,
        path: Union[str, Path],
        **load_kwargs
    ) -> np.ndarray:
        """
        Get an image from cache or load it.
        
        Args:
            path: Image path
            **load_kwargs: Arguments passed to load_image
            
        Returns:
            Image as numpy array
        """
        key = str(path)
        
        if key in self._cache:
            # Move to end of access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        # Load and cache
        image = load_image(path, **load_kwargs)
        
        # Evict if necessary
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = image
        self._access_order.append(key)
        
        return image
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
