"""
Experiment: Minutiae-Based Matching Evaluation

This experiment evaluates classical minutiae-based fingerprint
matching on the FVC2004 dataset.

Pipeline:
1. Enhancement (Gabor filtering)
2. Binarization and thinning
3. Minutiae extraction (crossing number)
4. Minutiae matching (geometric alignment)

Expected Results:
- Better than image-based baselines (SSIM)
- EER typically 5-15% on FVC2004
- Performance depends on image quality
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.minutiae.thinning import Thinner, zhang_suen_thinning
from src.minutiae.minutiae_extraction import MinutiaeExtractor
from src.minutiae.minutiae_matching import MinutiaeMatcher, MinutiaeMatchingPipeline
from src.enhancement.gabor_filter import GaborEnhancer
from src.enhancement.orientation_field import estimate_orientation_field
from src.enhancement.ridge_frequency import RidgeFrequencyEstimator
from src.data.fingerprint_dataset import FingerprintDataset
from src.data.pair_generator import PairGenerator
from src.data.preprocessing import FingerprintPreprocessor
from src.evaluation import VerificationEvaluator, VerificationResult
from src.utils.logger import setup_logger


class MinutiaeMatchingWrapper:
    """
    Wrapper class for minutiae matching that implements the matcher interface.
    """
    
    def __init__(
        self,
        use_enhancement: bool = True,
        alignment_method: str = "ransac"
    ):
        """
        Initialize minutiae matching wrapper.
        
        Args:
            use_enhancement: Whether to apply Gabor enhancement
            alignment_method: Alignment method for matching
        """
        self.use_enhancement = use_enhancement
        
        # Initialize components
        self.preprocessor = FingerprintPreprocessor()
        self.thinner = Thinner()
        self.extractor = MinutiaeExtractor()
        self.matcher = MinutiaeMatcher(alignment_method=alignment_method)
        
        if use_enhancement:
            self.enhancer = GaborEnhancer()
            self.freq_estimator = RidgeFrequencyEstimator()
        else:
            self.enhancer = None
    
    @property
    def name(self) -> str:
        return f"Minutiae_{self.matcher.alignment_method}"
    
    def _process_image(self, image: np.ndarray):
        """
        Process image to extract minutiae.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            List of Minutia objects
        """
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
        
        # Apply mask
        if mask is not None:
            binary = binary * mask
        
        # Thin
        skeleton = self.thinner.process(binary)
        
        # Extract minutiae
        minutiae = self.extractor.extract(skeleton, mask, orientation)
        
        return minutiae
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Match two fingerprint images.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Similarity score
        """
        try:
            minutiae_a = self._process_image(sample_a)
            minutiae_b = self._process_image(sample_b)
            
            if len(minutiae_a) < 2 or len(minutiae_b) < 2:
                return 0.0
            
            score = self.matcher.match(minutiae_a, minutiae_b)
            return score
            
        except Exception as e:
            print(f"Warning: Matching failed - {e}")
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


def run_experiment(
    data_dir: str,
    output_dir: str,
    use_enhancement: bool = True,
    alignment_method: str = "ransac"
):
    """
    Run minutiae matching experiment.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
        use_enhancement: Whether to use Gabor enhancement
        alignment_method: Alignment method
    """
    logger = setup_logger("minutiae_experiment")
    logger.info("Starting minutiae matching experiment")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = FingerprintDataset(data_dir, name="FVC2004")
    dataset.scan()
    
    logger.info(f"Found {len(dataset)} samples")
    
    # Generate pairs
    logger.info("Generating verification pairs")
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    logger.info(f"Generated {len(genuine_pairs)} genuine pairs")
    logger.info(f"Generated {len(impostor_pairs)} impostor pairs")
    
    # Initialize matcher
    matcher = MinutiaeMatchingWrapper(
        use_enhancement=use_enhancement,
        alignment_method=alignment_method
    )
    
    # Evaluate
    logger.info("Running evaluation")
    evaluator = VerificationEvaluator(matcher, verbose=True)
    result = evaluator.evaluate(genuine_pairs, impostor_pairs)
    
    # Generate report
    evaluator.generate_report(result, str(output_path))
    
    # Print results
    print("\n" + "=" * 60)
    print("MINUTIAE MATCHING EXPERIMENT RESULTS")
    print("=" * 60)
    print(result)
    
    logger.info(f"Results saved to {output_path}")
    
    return result


def compare_alignment_methods(
    data_dir: str,
    output_dir: str
):
    """
    Compare different alignment methods.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
    """
    logger = setup_logger("minutiae_comparison")
    logger.info("Comparing alignment methods")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = FingerprintDataset(data_dir, name="FVC2004")
    dataset.scan()
    
    # Generate pairs
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    # Test different methods
    methods = ["nearest", "hungarian", "ransac"]
    results = {}
    
    for method in methods:
        logger.info(f"Testing {method} alignment")
        
        matcher = MinutiaeMatchingWrapper(
            use_enhancement=True,
            alignment_method=method
        )
        
        evaluator = VerificationEvaluator(matcher, verbose=True)
        results[method] = evaluator.evaluate(genuine_pairs, impostor_pairs)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("ALIGNMENT METHOD COMPARISON")
    print("=" * 60)
    
    for method, result in sorted(results.items(), key=lambda x: x[1].eer):
        print(f"\n{method}:")
        print(f"  EER: {result.eer * 100:.2f}%")
        print(f"  AUC: {result.auc:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Minutiae Matching Experiment"
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
        default="results/minutiae_matching",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no_enhancement",
        action="store_true",
        help="Disable Gabor enhancement"
    )
    parser.add_argument(
        "--alignment",
        type=str,
        default="ransac",
        choices=["nearest", "hungarian", "ransac"],
        help="Alignment method"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all alignment methods"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_alignment_methods(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
    else:
        run_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_enhancement=not args.no_enhancement,
            alignment_method=args.alignment
        )


if __name__ == "__main__":
    main()
