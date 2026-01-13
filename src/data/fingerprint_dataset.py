"""
Fingerprint dataset handling and loading utilities.

This module provides classes and functions for loading fingerprint
datasets from various sources (FVC2004, Neurotechnology, etc.).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils.io import (
    discover_images,
    group_images_by_subject,
    load_image,
    parse_fvc_filename,
    SUPPORTED_EXTENSIONS
)


class DatasetType(Enum):
    """Enumeration of supported dataset types."""
    FVC2004 = "fvc2004"
    NEUROTECHNOLOGY_UAREU = "neurotech_uareu"
    NEUROTECHNOLOGY_CROSSMATCH = "neurotech_crossmatch"
    CUSTOM = "custom"


@dataclass
class FingerprintSample:
    """
    Represents a single fingerprint sample.
    
    Attributes:
        path: Path to the image file
        subject_id: Subject/identity identifier
        sample_id: Sample number for this subject
        sensor: Sensor/device used for capture
        image: Loaded image array (lazy loaded)
    """
    path: Path
    subject_id: int
    sample_id: int
    sensor: Optional[str] = None
    _image: Optional[np.ndarray] = None
    
    def load_image(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Load the fingerprint image.
        
        Args:
            target_size: Optional (width, height) to resize to
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Image as numpy array
        """
        if self._image is None or target_size is not None:
            self._image = load_image(
                self.path,
                grayscale=True,
                normalize=normalize,
                target_size=target_size
            )
        return self._image
    
    def clear_cache(self) -> None:
        """Clear the cached image to free memory."""
        self._image = None


class FingerprintDataset:
    """
    Dataset class for managing fingerprint images.
    
    Provides iteration, indexing, and grouping by subject.
    
    Mathematical Background:
    -----------------------
    A fingerprint dataset D consists of samples from N subjects:
    D = {(x_i, y_i)}_{i=1}^{M}
    
    where:
    - x_i is the i-th fingerprint image
    - y_i is the subject identity
    - M is the total number of samples
    
    For verification, we construct:
    - Genuine pairs: (x_i, x_j) where y_i = y_j
    - Impostor pairs: (x_i, x_j) where y_i != y_j
    """
    
    def __init__(
        self,
        root_path: Union[str, Path],
        dataset_type: DatasetType = DatasetType.FVC2004,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ):
        """
        Initialize the fingerprint dataset.
        
        Args:
            root_path: Root directory containing fingerprint images
            dataset_type: Type of dataset for parsing filenames
            target_size: Optional target size for images
            normalize: Whether to normalize images to [0, 1]
        """
        self.root_path = Path(root_path)
        self.dataset_type = dataset_type
        self.target_size = target_size
        self.normalize = normalize
        
        self.samples: List[FingerprintSample] = []
        self._by_subject: Dict[int, List[FingerprintSample]] = {}
        
        self._load_samples()
    
    def _get_filename_parser(self) -> callable:
        """Get the appropriate filename parser for the dataset type."""
        if self.dataset_type == DatasetType.FVC2004:
            return parse_fvc_filename
        elif self.dataset_type in [DatasetType.NEUROTECHNOLOGY_UAREU, 
                                    DatasetType.NEUROTECHNOLOGY_CROSSMATCH]:
            return self._parse_neurotech_filename
        else:
            return parse_fvc_filename
    
    @staticmethod
    def _parse_neurotech_filename(filename: str) -> Tuple[int, int]:
        """
        Parse Neurotechnology-style filename.
        
        Format varies, but typically: subject_finger_sample or similar
        """
        stem = Path(filename).stem
        parts = stem.replace('-', '_').split('_')
        
        # Try to extract numeric IDs
        numbers = [int(p) for p in parts if p.isdigit()]
        
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        elif len(numbers) == 1:
            return numbers[0], 1
        
        raise ValueError(f"Cannot parse Neurotech filename: {filename}")
    
    def _load_samples(self) -> None:
        """Load all samples from the root directory."""
        images = discover_images(self.root_path)
        parser = self._get_filename_parser()
        
        sensor = self._infer_sensor()
        
        for img_path in images:
            try:
                subject_id, sample_id = parser(img_path.name)
                sample = FingerprintSample(
                    path=img_path,
                    subject_id=subject_id,
                    sample_id=sample_id,
                    sensor=sensor
                )
                self.samples.append(sample)
                
                if subject_id not in self._by_subject:
                    self._by_subject[subject_id] = []
                self._by_subject[subject_id].append(sample)
                
            except ValueError:
                continue
        
        # Sort samples by subject and sample ID
        self.samples.sort(key=lambda s: (s.subject_id, s.sample_id))
        for subject_id in self._by_subject:
            self._by_subject[subject_id].sort(key=lambda s: s.sample_id)
    
    def _infer_sensor(self) -> Optional[str]:
        """Infer sensor type from directory structure."""
        path_str = str(self.root_path).lower()
        
        if 'uareu' in path_str:
            return 'uareu'
        elif 'crossmatch' in path_str:
            return 'crossmatch'
        elif 'db1' in path_str:
            return 'optical'
        elif 'db2' in path_str:
            return 'optical'
        elif 'db3' in path_str:
            return 'thermal'
        elif 'db4' in path_str:
            return 'synthetic'
        
        return None
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> FingerprintSample:
        """Get a sample by index."""
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[FingerprintSample]:
        """Iterate over all samples."""
        return iter(self.samples)
    
    @property
    def num_subjects(self) -> int:
        """Return number of unique subjects."""
        return len(self._by_subject)
    
    @property
    def subject_ids(self) -> List[int]:
        """Return list of all subject IDs."""
        return sorted(self._by_subject.keys())
    
    def get_subject_samples(self, subject_id: int) -> List[FingerprintSample]:
        """
        Get all samples for a specific subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            List of FingerprintSample objects
        """
        return self._by_subject.get(subject_id, [])
    
    def load_all_images(self) -> Dict[int, List[np.ndarray]]:
        """
        Load all images grouped by subject.
        
        Returns:
            Dictionary mapping subject_id to list of image arrays
        """
        result = {}
        for subject_id, samples in self._by_subject.items():
            result[subject_id] = [
                s.load_image(self.target_size, self.normalize)
                for s in samples
            ]
        return result
    
    def split_train_test(
        self,
        test_ratio: float = 0.2,
        seed: int = 42
    ) -> Tuple['FingerprintDataset', 'FingerprintDataset']:
        """
        Split dataset into train and test sets by subject.
        
        Note: Splits by subject to avoid identity leakage.
        
        Args:
            test_ratio: Fraction of subjects for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        np.random.seed(seed)
        subjects = np.array(self.subject_ids)
        np.random.shuffle(subjects)
        
        split_idx = int(len(subjects) * (1 - test_ratio))
        train_subjects = set(subjects[:split_idx])
        test_subjects = set(subjects[split_idx:])
        
        train_dataset = FingerprintDataset.__new__(FingerprintDataset)
        train_dataset.root_path = self.root_path
        train_dataset.dataset_type = self.dataset_type
        train_dataset.target_size = self.target_size
        train_dataset.normalize = self.normalize
        train_dataset.samples = [s for s in self.samples if s.subject_id in train_subjects]
        train_dataset._by_subject = {
            k: v for k, v in self._by_subject.items() if k in train_subjects
        }
        
        test_dataset = FingerprintDataset.__new__(FingerprintDataset)
        test_dataset.root_path = self.root_path
        test_dataset.dataset_type = self.dataset_type
        test_dataset.target_size = self.target_size
        test_dataset.normalize = self.normalize
        test_dataset.samples = [s for s in self.samples if s.subject_id in test_subjects]
        test_dataset._by_subject = {
            k: v for k, v in self._by_subject.items() if k in test_subjects
        }
        
        return train_dataset, test_dataset


def load_fvc2004_dataset(
    root_path: Union[str, Path],
    db_name: str = "DB1_B",
    target_size: Optional[Tuple[int, int]] = (256, 256),
    normalize: bool = True
) -> FingerprintDataset:
    """
    Load FVC2004 dataset.
    
    FVC2004 Structure:
    - 4 databases (DB1-DB4)
    - Each has set A (800 images: 100 subjects x 8 samples)
    - And set B (80 images: 10 subjects x 8 samples, for testing)
    
    Args:
        root_path: Root path to FVC2004 data
        db_name: Database name (DB1_B, DB2_B, etc.)
        target_size: Target image size
        normalize: Whether to normalize images
        
    Returns:
        FingerprintDataset instance
    """
    db_path = Path(root_path) / "FVC2004" / db_name
    return FingerprintDataset(
        db_path,
        DatasetType.FVC2004,
        target_size,
        normalize
    )


def load_neurotech_dataset(
    root_path: Union[str, Path],
    sensor: str = "UareU",
    target_size: Optional[Tuple[int, int]] = (256, 256),
    normalize: bool = True
) -> FingerprintDataset:
    """
    Load Neurotechnology dataset.
    
    Args:
        root_path: Root path to Neurotechnology data
        sensor: Sensor name ('UareU' or 'CrossMatch')
        target_size: Target image size
        normalize: Whether to normalize images
        
    Returns:
        FingerprintDataset instance
    """
    sensor_path = Path(root_path) / "Neurotech" / sensor
    
    dataset_type = (
        DatasetType.NEUROTECHNOLOGY_UAREU 
        if sensor.lower() == 'uareu' 
        else DatasetType.NEUROTECHNOLOGY_CROSSMATCH
    )
    
    return FingerprintDataset(
        sensor_path,
        dataset_type,
        target_size,
        normalize
    )
