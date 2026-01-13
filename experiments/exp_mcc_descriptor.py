"""
Experiment: MCC Descriptor Matching Evaluation

This experiment evaluates the Minutia Cylinder Code (MCC)
descriptor-based matching on the FVC2004 dataset.

MCC encodes local minutiae arrangements in a 3D cylindrical
structure, providing:
- Rotation invariance (through alignment)
- Robustness to non-linear distortions
- Efficient binary representation

Expected Results:
- Better than direct minutiae matching
- EER typically 3-8% on FVC2004
- Good generalization across sensors
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.descriptors.mcc import MCCConfig, MCCDescriptor, compute_mcc_descriptors
from src.descriptors.local_orientation_descriptor import (
    LODConfig, LocalOrientationDescriptor
)
from src.descriptors.descriptor_matching import (
    MCCMatcher, LocalOrientationMatcher, DescriptorMatchingPipeline
)
from src.minutiae.thinning import Thinner
from src.minutiae.minutiae_extraction import MinutiaeExtractor
from src.enhancement.gabor_filter import GaborEnhancer
from src.enhancement.orientation_field import estimate_orientation_field
from src.enhancement.ridge_frequency import RidgeFrequencyEstimator
from src.data.fingerprint_dataset import FingerprintDataset
from src.data.pair_generator import PairGenerator
from src.data.preprocessing import FingerprintPreprocessor
from src.evaluation import VerificationEvaluator, compare_matchers
from src.utils.logger import setup_logger


class MCCMatchingWrapper:
    """
    Wrapper for MCC descriptor matching with full pipeline.
    """
    
    def __init__(
        self,
        mcc_config: MCCConfig = None,
        method: str = "lss",
        use_enhancement: bool = True
    ):
        """
        Initialize MCC matching wrapper.
        
        Args:
            mcc_config: MCC configuration
            method: Matching method ('lss' or 'lsar')
            use_enhancement: Whether to use enhancement
        """
        self.mcc_config = mcc_config or MCCConfig()
        self.method = method
        self.use_enhancement = use_enhancement
        
        # Initialize components
        self.preprocessor = FingerprintPreprocessor()
        self.thinner = Thinner()
        self.extractor = MinutiaeExtractor()
        self.matcher = MCCMatcher(
            config=self.mcc_config,
            method=method
        )
        
        if use_enhancement:
            self.enhancer = GaborEnhancer()
            self.freq_estimator = RidgeFrequencyEstimator()
        else:
            self.enhancer = None
    
    @property
    def name(self) -> str:
        return f"MCC_{self.method.upper()}"
    
    def _extract_minutiae(self, image: np.ndarray):
        """Extract minutiae from image."""
        # Preprocess
        processed, mask = self.preprocessor(image)
        
        # Estimate orientation
        orientation = estimate_orientation_field(processed)
        
        # Enhance if enabled
        if self.enhancer is not None:
            frequency = self.freq_estimator.estimate(processed, orientation, mask)
            enhanced = self.enhancer.enhance(processed, orientation, frequency, mask)
        else:
            enhanced = processed
        
        # Binarize
        if enhanced.max() > 1:
            enhanced = enhanced / 255.0
        binary = (enhanced < 0.5).astype(np.uint8)
        
        if mask is not None:
            binary = binary * mask
        
        # Thin and extract
        skeleton = self.thinner.process(binary)
        minutiae = self.extractor.extract(skeleton, mask, orientation)
        
        return minutiae
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Match two fingerprint images using MCC.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Similarity score
        """
        try:
            # Extract minutiae
            minutiae_a = self._extract_minutiae(sample_a)
            minutiae_b = self._extract_minutiae(sample_b)
            
            if len(minutiae_a) < 3 or len(minutiae_b) < 3:
                return 0.0
            
            # Compute MCC descriptors
            desc_a = MCCDescriptor(self.mcc_config)
            desc_a.compute(minutiae_a)
            
            desc_b = MCCDescriptor(self.mcc_config)
            desc_b.compute(minutiae_b)
            
            if len(desc_a) == 0 or len(desc_b) == 0:
                return 0.0
            
            # Match descriptors
            score = self.matcher.match_descriptors(desc_a, desc_b)
            
            return score
            
        except Exception as e:
            print(f"Warning: MCC matching failed - {e}")
            return 0.0


class LODMatchingWrapper:
    """
    Wrapper for Local Orientation Descriptor matching.
    """
    
    def __init__(
        self,
        lod_config: LODConfig = None,
        use_enhancement: bool = True
    ):
        """
        Initialize LOD matching wrapper.
        
        Args:
            lod_config: LOD configuration
            use_enhancement: Whether to use enhancement
        """
        self.lod_config = lod_config or LODConfig()
        self.use_enhancement = use_enhancement
        
        # Initialize components
        self.preprocessor = FingerprintPreprocessor()
        self.thinner = Thinner()
        self.extractor = MinutiaeExtractor()
        self.matcher = LocalOrientationMatcher(config=self.lod_config)
        
        if use_enhancement:
            self.enhancer = GaborEnhancer()
            self.freq_estimator = RidgeFrequencyEstimator()
        else:
            self.enhancer = None
    
    @property
    def name(self) -> str:
        return "LocalOrientation"
    
    def _process_image(self, image: np.ndarray):
        """Process image to get minutiae and orientation."""
        # Preprocess
        processed, mask = self.preprocessor(image)
        
        # Estimate orientation
        orientation = estimate_orientation_field(processed)
        
        # Enhance if enabled
        if self.enhancer is not None:
            frequency = self.freq_estimator.estimate(processed, orientation, mask)
            enhanced = self.enhancer.enhance(processed, orientation, frequency, mask)
        else:
            enhanced = processed
        
        # Binarize
        if enhanced.max() > 1:
            enhanced = enhanced / 255.0
        binary = (enhanced < 0.5).astype(np.uint8)
        
        if mask is not None:
            binary = binary * mask
        
        # Thin and extract
        skeleton = self.thinner.process(binary)
        minutiae = self.extractor.extract(skeleton, mask, orientation)
        
        return minutiae, orientation
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Match using local orientation descriptors."""
        try:
            minutiae_a, orientation_a = self._process_image(sample_a)
            minutiae_b, orientation_b = self._process_image(sample_b)
            
            if len(minutiae_a) < 3 or len(minutiae_b) < 3:
                return 0.0
            
            # Compute descriptors
            desc_a = LocalOrientationDescriptor(self.lod_config)
            desc_a.compute(orientation_a, minutiae_a)
            
            desc_b = LocalOrientationDescriptor(self.lod_config)
            desc_b.compute(orientation_b, minutiae_b)
            
            if len(desc_a) == 0 or len(desc_b) == 0:
                return 0.0
            
            score = self.matcher.match_descriptors(desc_a, desc_b)
            
            return score
            
        except Exception as e:
            print(f"Warning: LOD matching failed - {e}")
            return 0.0


def load_pairs_from_dataset(
    dataset: FingerprintDataset,
    num_impostor_ratio: float = 1.0
):
    """Load genuine and impostor pairs from dataset."""
    generator = PairGenerator(dataset)
    all_pairs = generator.generate_pairs(impostor_ratio=num_impostor_ratio)
    
    genuine_pairs = []
    impostor_pairs = []
    
    for pair in all_pairs:
        img1 = pair.sample1.load_image()
        img2 = pair.sample2.load_image()
        
        if pair.label == 1:
            genuine_pairs.append((img1, img2))
        else:
            impostor_pairs.append((img1, img2))
    
    return genuine_pairs, impostor_pairs


def run_mcc_experiment(
    data_dir: str,
    output_dir: str,
    mcc_config: MCCConfig = None,
    method: str = "lss"
):
    """
    Run MCC descriptor matching experiment.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
        mcc_config: MCC configuration
        method: Matching method
    """
    logger = setup_logger("mcc_experiment")
    logger.info("Starting MCC descriptor experiment")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = FingerprintDataset(data_dir, name="FVC2004")
    dataset.scan()
    
    logger.info(f"Found {len(dataset)} samples")
    
    # Generate pairs
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    logger.info(f"Generated {len(genuine_pairs)} genuine, {len(impostor_pairs)} impostor pairs")
    
    # Initialize matcher
    matcher = MCCMatchingWrapper(
        mcc_config=mcc_config,
        method=method
    )
    
    # Evaluate
    logger.info("Running evaluation")
    evaluator = VerificationEvaluator(matcher, verbose=True)
    result = evaluator.evaluate(genuine_pairs, impostor_pairs)
    
    # Generate report
    evaluator.generate_report(result, str(output_path))
    
    print("\n" + "=" * 60)
    print("MCC DESCRIPTOR EXPERIMENT RESULTS")
    print("=" * 60)
    print(result)
    
    return result


def compare_descriptors(
    data_dir: str,
    output_dir: str
):
    """
    Compare MCC vs Local Orientation descriptors.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
    """
    logger = setup_logger("descriptor_comparison")
    logger.info("Comparing descriptor methods")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = FingerprintDataset(data_dir, name="FVC2004")
    dataset.scan()
    
    # Generate pairs
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    # Initialize matchers
    matchers = {
        "MCC_LSS": MCCMatchingWrapper(method="lss"),
        "MCC_LSAR": MCCMatchingWrapper(method="lsar"),
        "LocalOrientation": LODMatchingWrapper(),
    }
    
    # Compare
    results = compare_matchers(
        matchers,
        genuine_pairs,
        impostor_pairs,
        output_dir=str(output_path),
        verbose=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DESCRIPTOR COMPARISON")
    print("=" * 60)
    
    for name, result in sorted(results.items(), key=lambda x: x[1].eer):
        print(f"\n{name}:")
        print(f"  EER: {result.eer * 100:.2f}%")
        print(f"  AUC: {result.auc:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MCC Descriptor Matching Experiment"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/FVC2004/DB1_B",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mcc_descriptor",
        help="Output directory for results"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lss",
        choices=["lss", "lsar"],
        help="MCC matching method"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all descriptor methods"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=70.0,
        help="MCC cylinder radius"
    )
    parser.add_argument(
        "--spatial_cells",
        type=int,
        default=16,
        help="Number of spatial cells"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_descriptors(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    else:
        mcc_config = MCCConfig(
            radius=args.radius,
            num_spatial_cells=args.spatial_cells
        )
        
        run_mcc_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mcc_config=mcc_config,
            method=args.method
        )


if __name__ == "__main__":
    main()
