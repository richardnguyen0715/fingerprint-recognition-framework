"""
Pair generation for fingerprint verification evaluation.

This module generates genuine and impostor pairs following
standard biometric verification protocols.
"""

import csv
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from src.data.fingerprint_dataset import FingerprintDataset, FingerprintSample


# =============================================================================
# MATHEMATICAL BACKGROUND - VERIFICATION PROTOCOL
# =============================================================================
#
# Fingerprint Verification Task:
# Given two fingerprint images (probe, gallery), determine if they
# belong to the same person.
#
# Pair Types:
# -----------
# 1. Genuine pairs: Two samples from the same subject
#    - Label: 1 (positive)
#    - Count: For N subjects with S samples each: N * C(S,2) = N * S*(S-1)/2
#
# 2. Impostor pairs: Two samples from different subjects
#    - Label: 0 (negative)
#    - Count: Typically sampled to balance dataset or follow protocol
#
# Evaluation Metrics:
# ------------------
# - FAR (False Accept Rate): Impostors incorrectly accepted
# - FRR (False Reject Rate): Genuines incorrectly rejected
# - EER (Equal Error Rate): Operating point where FAR = FRR
# =============================================================================


class VerificationPair:
    """
    Represents a verification pair for evaluation.
    
    Attributes:
        sample1: First fingerprint sample
        sample2: Second fingerprint sample
        label: 1 for genuine, 0 for impostor
    """
    
    def __init__(
        self,
        sample1: FingerprintSample,
        sample2: FingerprintSample,
        label: int
    ):
        self.sample1 = sample1
        self.sample2 = sample2
        self.label = label
    
    def __repr__(self) -> str:
        pair_type = "genuine" if self.label == 1 else "impostor"
        return f"VerificationPair({self.sample1.path.name}, {self.sample2.path.name}, {pair_type})"


class PairGenerator:
    """
    Generates verification pairs from a fingerprint dataset.
    
    Supports various pair generation strategies:
    - All genuine pairs (exhaustive)
    - Sampled impostor pairs (random or controlled)
    - Balanced genuine/impostor ratio
    """
    
    def __init__(
        self,
        dataset: FingerprintDataset,
        seed: int = 42
    ):
        """
        Initialize the pair generator.
        
        Args:
            dataset: FingerprintDataset to generate pairs from
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_genuine_pairs(self) -> List[VerificationPair]:
        """
        Generate all possible genuine pairs.
        
        For each subject with S samples, generates C(S,2) = S*(S-1)/2 pairs.
        
        Returns:
            List of genuine VerificationPair objects
        """
        pairs = []
        
        for subject_id in self.dataset.subject_ids:
            samples = self.dataset.get_subject_samples(subject_id)
            
            # Generate all combinations of samples from this subject
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    pairs.append(VerificationPair(samples[i], samples[j], label=1))
        
        return pairs
    
    def generate_impostor_pairs(
        self,
        num_pairs: Optional[int] = None,
        per_sample: int = 5
    ) -> List[VerificationPair]:
        """
        Generate impostor pairs (different subjects).
        
        Two strategies:
        1. If num_pairs specified: randomly sample that many pairs
        2. Otherwise: generate 'per_sample' impostors for each sample
        
        Args:
            num_pairs: Total number of impostor pairs (overrides per_sample)
            per_sample: Number of impostor pairs per sample
            
        Returns:
            List of impostor VerificationPair objects
        """
        pairs = []
        subjects = self.dataset.subject_ids
        
        if num_pairs is not None:
            # Random sampling strategy
            all_samples = list(self.dataset.samples)
            
            while len(pairs) < num_pairs:
                s1, s2 = random.sample(all_samples, 2)
                
                if s1.subject_id != s2.subject_id:
                    pairs.append(VerificationPair(s1, s2, label=0))
        else:
            # Per-sample strategy
            for sample in self.dataset.samples:
                # Get samples from other subjects
                other_subjects = [s for s in subjects if s != sample.subject_id]
                
                for _ in range(per_sample):
                    if not other_subjects:
                        break
                    
                    other_subject = random.choice(other_subjects)
                    other_samples = self.dataset.get_subject_samples(other_subject)
                    other_sample = random.choice(other_samples)
                    
                    pairs.append(VerificationPair(sample, other_sample, label=0))
        
        return pairs
    
    def generate_pairs(
        self,
        impostor_ratio: float = 1.0,
        max_impostor_per_sample: int = 10
    ) -> List[VerificationPair]:
        """
        Generate a complete set of verification pairs.
        
        Args:
            impostor_ratio: Ratio of impostor to genuine pairs
            max_impostor_per_sample: Maximum impostors per sample
            
        Returns:
            Combined list of genuine and impostor pairs
        """
        genuine_pairs = self.generate_genuine_pairs()
        num_genuine = len(genuine_pairs)
        
        # Calculate number of impostor pairs
        num_impostor = int(num_genuine * impostor_ratio)
        
        # Generate impostors
        impostor_pairs = self.generate_impostor_pairs(num_pairs=num_impostor)
        
        all_pairs = genuine_pairs + impostor_pairs
        random.shuffle(all_pairs)
        
        return all_pairs
    
    def generate_cross_sensor_pairs(
        self,
        dataset_other: FingerprintDataset,
        genuine_only: bool = False
    ) -> List[VerificationPair]:
        """
        Generate pairs across two datasets (cross-sensor evaluation).
        
        For cross-sensor evaluation, pairs are formed between samples
        captured with different sensors.
        
        Args:
            dataset_other: Second dataset (different sensor)
            genuine_only: If True, only generate genuine pairs
            
        Returns:
            List of cross-sensor VerificationPair objects
        """
        pairs = []
        
        # Find common subjects
        common_subjects = set(self.dataset.subject_ids) & set(dataset_other.subject_ids)
        
        for subject_id in common_subjects:
            samples1 = self.dataset.get_subject_samples(subject_id)
            samples2 = dataset_other.get_subject_samples(subject_id)
            
            # Genuine pairs: same subject, different sensors
            for s1 in samples1:
                for s2 in samples2:
                    pairs.append(VerificationPair(s1, s2, label=1))
        
        if not genuine_only:
            # Impostor pairs: different subjects, different sensors
            all_samples1 = list(self.dataset.samples)
            all_samples2 = list(dataset_other.samples)
            
            num_genuine = len(pairs)
            num_impostor = 0
            
            while num_impostor < num_genuine:
                s1 = random.choice(all_samples1)
                s2 = random.choice(all_samples2)
                
                if s1.subject_id != s2.subject_id:
                    pairs.append(VerificationPair(s1, s2, label=0))
                    num_impostor += 1
        
        random.shuffle(pairs)
        return pairs


def save_pairs_to_csv(
    pairs: List[VerificationPair],
    output_path: Union[str, Path]
) -> None:
    """
    Save verification pairs to a CSV file.
    
    CSV format: img1,img2,label
    
    Args:
        pairs: List of VerificationPair objects
        output_path: Output CSV file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['img1', 'img2', 'label'])
        
        for pair in pairs:
            writer.writerow([
                str(pair.sample1.path),
                str(pair.sample2.path),
                pair.label
            ])


def load_pairs_from_csv(
    csv_path: Union[str, Path]
) -> List[Tuple[str, str, int]]:
    """
    Load verification pairs from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    pairs = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row['img1'], row['img2'], int(row['label'])))
    
    return pairs


# Legacy function for backward compatibility
def generate_pairs(images: Dict, out_csv: str, num_impostor: int = 5) -> None:
    """
    Legacy pair generation function.
    
    Generates pairs from a dictionary of images grouped by subject.
    
    Args:
        images: Dict mapping subject_id to list of image paths
        out_csv: Output CSV file path
        num_impostor: Number of impostor pairs per sample
    """
    pairs = []
    subjects = list(images.keys())

    # Genuine pairs
    for subj, imgs in images.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j], 1))

    # Impostor pairs
    for subj in subjects:
        for img in images[subj]:
            others = [s for s in subjects if s != subj]
            for _ in range(num_impostor):
                other = random.choice(others)
                other_img = random.choice(images[other])
                pairs.append((img, other_img, 0))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img1", "img2", "label"])
        for p in pairs:
            writer.writerow(p)


class PairIterator:
    """
    Memory-efficient iterator over verification pairs.
    
    Loads images on-demand rather than keeping all in memory.
    """
    
    def __init__(
        self,
        pairs: List[VerificationPair],
        target_size: Optional[Tuple[int, int]] = (256, 256),
        normalize: bool = True,
        batch_size: int = 1
    ):
        """
        Initialize the pair iterator.
        
        Args:
            pairs: List of VerificationPair objects
            target_size: Target image size
            normalize: Whether to normalize images
            batch_size: Number of pairs per iteration
        """
        self.pairs = pairs
        self.target_size = target_size
        self.normalize = normalize
        self.batch_size = batch_size
        self.index = 0
    
    def __iter__(self) -> Iterator:
        self.index = 0
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.index >= len(self.pairs):
            raise StopIteration
        
        batch_pairs = self.pairs[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        
        imgs1 = []
        imgs2 = []
        labels = []
        
        for pair in batch_pairs:
            img1 = pair.sample1.load_image(self.target_size, self.normalize)
            img2 = pair.sample2.load_image(self.target_size, self.normalize)
            
            imgs1.append(img1)
            imgs2.append(img2)
            labels.append(pair.label)
        
        return (
            np.array(imgs1),
            np.array(imgs2),
            np.array(labels)
        )
    
    def __len__(self) -> int:
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size


def get_pair_statistics(pairs: List[VerificationPair]) -> Dict[str, int]:
    """
    Compute statistics about a pair set.
    
    Args:
        pairs: List of VerificationPair objects
        
    Returns:
        Dictionary with pair statistics
    """
    num_genuine = sum(1 for p in pairs if p.label == 1)
    num_impostor = sum(1 for p in pairs if p.label == 0)
    
    subjects = set()
    for p in pairs:
        subjects.add(p.sample1.subject_id)
        subjects.add(p.sample2.subject_id)
    
    return {
        'total_pairs': len(pairs),
        'genuine_pairs': num_genuine,
        'impostor_pairs': num_impostor,
        'ratio': num_impostor / num_genuine if num_genuine > 0 else 0,
        'num_subjects': len(subjects)
    }
