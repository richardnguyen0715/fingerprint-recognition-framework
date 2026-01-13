"""
Experiment: SSIM Baseline Evaluation on FVC2004

This experiment evaluates the SSIM (Structural Similarity Index)
baseline matcher on the FVC2004 dataset.

SSIM is an image-based metric that compares structural patterns
without fingerprint-specific feature extraction.

Expected Results:
- SSIM provides a simple baseline but is sensitive to alignment
- EER typically high (>10%) due to lack of invariance
- Useful as a lower bound for comparison
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.ssim import SSIMMatcher, MSEMatcher, NCCMatcher
from src.data.fingerprint_dataset import FingerprintDataset
from src.data.pair_generator import PairGenerator
from src.evaluation import VerificationEvaluator, compare_matchers
from src.utils.config import load_config
from src.utils.logger import setup_logger


def load_pairs_from_dataset(
    dataset: FingerprintDataset,
    num_impostor_ratio: float = 1.0
):
    """
    Load genuine and impostor pairs from dataset.
    
    Args:
        dataset: FingerprintDataset instance
        num_impostor_ratio: Ratio of impostor to genuine pairs
        
    Returns:
        Tuple of (genuine_pairs, impostor_pairs)
    """
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
    config_path: str = None,
    dataset_name: str = "FVC2004_DB1_B"
):
    """
    Run SSIM baseline experiment.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
        config_path: Optional path to config file
        dataset_name: Name of dataset subset
    """
    logger = setup_logger("ssim_experiment")
    logger.info("Starting SSIM baseline experiment")
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = {}
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = FingerprintDataset(data_dir, name=dataset_name)
    dataset.scan()
    
    logger.info(f"Found {len(dataset)} samples from {len(dataset.subject_ids)} subjects")
    
    # Generate pairs
    logger.info("Generating verification pairs")
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    logger.info(f"Generated {len(genuine_pairs)} genuine pairs")
    logger.info(f"Generated {len(impostor_pairs)} impostor pairs")
    
    # Initialize matchers
    matchers = {
        "MSE": MSEMatcher(),
        "NCC": NCCMatcher(),
        "SSIM": SSIMMatcher(),
    }
    
    # Run comparison
    logger.info("Running matcher comparison")
    results = compare_matchers(
        matchers,
        genuine_pairs,
        impostor_pairs,
        output_dir=str(output_path),
        verbose=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SSIM BASELINE EXPERIMENT RESULTS")
    print("=" * 60)
    
    for name, result in sorted(results.items(), key=lambda x: x[1].eer):
        print(f"\n{name}:")
        print(f"  EER: {result.eer * 100:.2f}%")
        print(f"  AUC: {result.auc:.4f}")
        print(f"  d': {result.d_prime:.2f}")
    
    logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SSIM Baseline Experiment on FVC2004"
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
        default="results/ssim_baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ssim.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
