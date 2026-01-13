"""
Experiment: Cross-Sensor Evaluation

This experiment evaluates fingerprint recognition performance
across different sensors to measure generalization.

Sensors:
1. FVC2004 DB1_B (Optical sensor)
2. Neurotechnology UareU
3. Neurotechnology CrossMatch

Cross-sensor scenarios:
- Same sensor (baseline)
- Train on one sensor, test on another
- Domain adaptation analysis

Expected Results:
- Performance degradation across sensors
- Higher EER for cross-sensor evaluation
- Importance of sensor-invariant features
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from datetime import datetime

from src.data.fingerprint_dataset import FingerprintDataset
from src.data.pair_generator import PairGenerator
from src.baselines.ssim import SSIMMatcher
from src.evaluation import (
    VerificationEvaluator, 
    CrossSensorEvaluator,
    VerificationResult
)
from src.utils.logger import setup_logger

# Try to import deep learning models
try:
    from src.models import CNNEmbeddingMatcher, HybridMatcher
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


def generate_cross_sensor_pairs(
    dataset1: FingerprintDataset,
    dataset2: FingerprintDataset,
    num_pairs: int = 100
):
    """
    Generate cross-sensor pairs (if subjects overlap).
    
    In practice, this requires datasets with matching subject IDs.
    For evaluation purposes, we generate pseudo cross-sensor pairs.
    
    Args:
        dataset1: First sensor dataset
        dataset2: Second sensor dataset
        num_pairs: Number of pairs to generate
        
    Returns:
        Tuple of (genuine_pairs, impostor_pairs)
    """
    # Find common subjects (if IDs match)
    subjects1 = set(dataset1.subject_ids)
    subjects2 = set(dataset2.subject_ids)
    common_subjects = subjects1.intersection(subjects2)
    
    genuine_pairs = []
    impostor_pairs = []
    
    if len(common_subjects) > 0:
        # Generate genuine pairs across sensors
        for subject_id in common_subjects:
            samples1 = dataset1.get_subject_samples(subject_id)
            samples2 = dataset2.get_subject_samples(subject_id)
            
            for s1 in samples1:
                for s2 in samples2:
                    genuine_pairs.append((
                        s1.load_image(),
                        s2.load_image()
                    ))
    
    # Generate impostor pairs
    all_samples1 = list(dataset1.samples)
    all_samples2 = list(dataset2.samples)
    
    for _ in range(num_pairs):
        s1 = np.random.choice(all_samples1)
        s2 = np.random.choice(all_samples2)
        
        if s1.subject_id != s2.subject_id or len(common_subjects) == 0:
            impostor_pairs.append((
                s1.load_image(),
                s2.load_image()
            ))
    
    return genuine_pairs, impostor_pairs


def run_single_sensor_evaluation(
    data_dir: str,
    output_dir: str,
    sensor_name: str,
    matcher
):
    """
    Run evaluation on a single sensor.
    
    Args:
        data_dir: Path to sensor dataset
        output_dir: Output directory
        sensor_name: Name of sensor
        matcher: Fingerprint matcher
        
    Returns:
        VerificationResult
    """
    logger = setup_logger(f"single_sensor_{sensor_name}")
    
    # Load dataset
    logger.info(f"Loading {sensor_name} dataset from {data_dir}")
    
    if not Path(data_dir).exists():
        logger.warning(f"Dataset not found: {data_dir}")
        return None
    
    dataset = FingerprintDataset(data_dir, name=sensor_name)
    dataset.scan()
    
    if len(dataset) == 0:
        logger.warning(f"No samples found in {data_dir}")
        return None
    
    logger.info(f"Found {len(dataset)} samples")
    
    # Generate pairs
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    if len(genuine_pairs) == 0 or len(impostor_pairs) == 0:
        logger.warning("Insufficient pairs for evaluation")
        return None
    
    # Evaluate
    evaluator = VerificationEvaluator(matcher, verbose=True)
    result = evaluator.evaluate(genuine_pairs, impostor_pairs)
    
    # Save results
    output_path = Path(output_dir) / sensor_name
    evaluator.generate_report(result, str(output_path))
    
    return result


def run_cross_sensor_experiment(
    base_data_dir: str,
    output_dir: str,
    matcher_type: str = "ssim"
):
    """
    Run cross-sensor evaluation experiment.
    
    Args:
        base_data_dir: Base directory containing sensor subdirectories
        output_dir: Output directory for results
        matcher_type: Type of matcher to use
    """
    logger = setup_logger("cross_sensor_experiment")
    logger.info("Starting cross-sensor evaluation experiment")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define sensor datasets
    sensors = {
        "FVC2004_DB1": Path(base_data_dir) / "FVC2004" / "DB1_B",
        "FVC2004_DB2": Path(base_data_dir) / "FVC2004" / "DB2_B",
        "FVC2004_DB3": Path(base_data_dir) / "FVC2004" / "DB3_B",
        "FVC2004_DB4": Path(base_data_dir) / "FVC2004" / "DB4_B",
    }
    
    # Add Neurotechnology if available
    neurotech_uareu = Path(base_data_dir) / "Neurotech" / "UareU"
    neurotech_crossmatch = Path(base_data_dir) / "Neurotech" / "CrossMatch"
    
    if neurotech_uareu.exists():
        sensors["UareU"] = neurotech_uareu
    if neurotech_crossmatch.exists():
        sensors["CrossMatch"] = neurotech_crossmatch
    
    # Filter to existing sensors
    available_sensors = {
        name: path for name, path in sensors.items() 
        if path.exists()
    }
    
    if len(available_sensors) == 0:
        logger.error("No sensor datasets found")
        return None
    
    logger.info(f"Found {len(available_sensors)} sensor datasets")
    
    # Create matcher
    if matcher_type == "ssim":
        matcher = SSIMMatcher()
    elif matcher_type == "cnn" and TORCH_AVAILABLE:
        matcher = CNNEmbeddingMatcher()
    elif matcher_type == "hybrid" and TORCH_AVAILABLE:
        from src.models import HybridMatcher
        matcher = HybridMatcher()
    else:
        matcher = SSIMMatcher()
    
    # Evaluate each sensor
    results = {}
    
    for sensor_name, sensor_path in available_sensors.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {sensor_name}")
        logger.info(f"{'='*50}")
        
        result = run_single_sensor_evaluation(
            str(sensor_path),
            str(output_path),
            sensor_name,
            matcher
        )
        
        if result is not None:
            results[sensor_name] = result
    
    # Generate cross-sensor summary
    generate_cross_sensor_summary(results, output_path)
    
    return results


def generate_cross_sensor_summary(
    results: dict,
    output_path: Path
):
    """
    Generate cross-sensor evaluation summary.
    
    Args:
        results: Dictionary of sensor -> VerificationResult
        output_path: Output directory
    """
    lines = [
        "Cross-Sensor Evaluation Summary",
        "=" * 60,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Single-Sensor Results:",
        "-" * 40,
    ]
    
    # Sort by EER
    sorted_results = sorted(results.items(), key=lambda x: x[1].eer)
    
    for sensor_name, result in sorted_results:
        lines.extend([
            f"\n{sensor_name}:",
            f"  Samples: {result.num_genuine + result.num_impostor} pairs",
            f"  EER: {result.eer * 100:.2f}%",
            f"  AUC: {result.auc:.4f}",
            f"  d': {result.d_prime:.2f}",
        ])
    
    # Overall statistics
    lines.extend([
        "",
        "Overall Statistics:",
        "-" * 40,
        f"Number of sensors evaluated: {len(results)}",
        f"Average EER: {np.mean([r.eer for r in results.values()]) * 100:.2f}%",
        f"EER Range: {np.min([r.eer for r in results.values()]) * 100:.2f}% - "
        f"{np.max([r.eer for r in results.values()]) * 100:.2f}%",
    ])
    
    # Save summary
    summary_path = output_path / "cross_sensor_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines))
    
    # Save JSON results
    json_results = {
        name: result.to_dict() for name, result in results.items()
    }
    json_path = output_path / "cross_sensor_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Sensor Evaluation Experiment"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Base directory containing sensor datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cross_sensor",
        help="Output directory for results"
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="ssim",
        choices=["ssim", "minutiae", "mcc", "cnn", "hybrid"],
        help="Matcher type to use"
    )
    
    args = parser.parse_args()
    
    # Import appropriate matcher
    if args.matcher == "minutiae":
        from experiments.exp_minutiae_matching import MinutiaeMatchingWrapper
    elif args.matcher == "mcc":
        from experiments.exp_mcc_descriptor import MCCMatchingWrapper
    
    run_cross_sensor_experiment(
        base_data_dir=args.data_dir,
        output_dir=args.output_dir,
        matcher_type=args.matcher
    )


if __name__ == "__main__":
    main()
