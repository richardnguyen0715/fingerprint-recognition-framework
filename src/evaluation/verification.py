"""
Fingerprint verification evaluation protocol.

This module implements the complete verification protocol for
evaluating fingerprint recognition systems following biometric
standards.

Verification Protocol:
---------------------
1. Generate genuine and impostor pairs
2. Compute similarity scores for all pairs
3. Compute performance metrics (ROC, EER, FAR@FRR)
4. Generate evaluation report

Pair Generation:
- Genuine pairs: Same subject, different samples
- Impostor pairs: Different subjects

Standard Metrics:
- EER (Equal Error Rate)
- FAR@FRR=0.1% (Security-oriented)
- FRR@FAR=1% (Convenience-oriented)
- AUC (Overall performance)

Reference:
ISO/IEC 19795-1:2021 - Biometric performance testing and reporting.
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluation.roc import ROCCurve, compute_roc_curve, compare_roc_curves
from src.evaluation.eer import (
    EERResult, compute_eer, compute_eer_from_labels,
    compute_d_prime, compute_far_at_frr, compute_frr_at_far,
    plot_score_distributions, plot_far_frr_curve
)


@dataclass
class VerificationResult:
    """
    Complete verification evaluation result.
    
    Attributes:
        method_name: Name of the matching method
        num_genuine: Number of genuine pairs
        num_impostor: Number of impostor pairs
        eer: Equal Error Rate
        eer_threshold: Threshold at EER
        auc: Area Under ROC Curve
        d_prime: Discriminability index
        far_at_frr_01: FAR at FRR=0.1%
        frr_at_far_1: FRR at FAR=1%
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        roc_curve: ROC curve data
        eer_result: Detailed EER result
        processing_time: Total processing time in seconds
    """
    method_name: str
    num_genuine: int
    num_impostor: int
    eer: float
    eer_threshold: float
    auc: float
    d_prime: float
    far_at_frr_01: float
    frr_at_far_1: float
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray
    roc_curve: ROCCurve
    eer_result: EERResult
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method_name": self.method_name,
            "num_genuine": self.num_genuine,
            "num_impostor": self.num_impostor,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "auc": self.auc,
            "d_prime": self.d_prime,
            "far_at_frr_01": self.far_at_frr_01,
            "frr_at_far_1": self.frr_at_far_1,
            "processing_time": self.processing_time
        }
    
    def __str__(self) -> str:
        """Format as readable string."""
        return (
            f"Verification Results: {self.method_name}\n"
            f"{'='*50}\n"
            f"Pairs: {self.num_genuine} genuine, {self.num_impostor} impostor\n"
            f"EER: {self.eer*100:.2f}% (threshold: {self.eer_threshold:.4f})\n"
            f"AUC: {self.auc:.4f}\n"
            f"d': {self.d_prime:.2f}\n"
            f"FAR@FRR=0.1%: {self.far_at_frr_01*100:.3f}%\n"
            f"FRR@FAR=1%: {self.frr_at_far_1*100:.2f}%\n"
            f"Processing time: {self.processing_time:.1f}s\n"
        )


class VerificationEvaluator:
    """
    Evaluator for fingerprint verification systems.
    
    This class provides a complete evaluation framework for
    fingerprint matchers, including:
    - Pair generation
    - Score computation
    - Metric calculation
    - Report generation
    
    Usage:
    ------
    evaluator = VerificationEvaluator(matcher)
    result = evaluator.evaluate(genuine_pairs, impostor_pairs)
    evaluator.generate_report(result, output_dir)
    """
    
    def __init__(
        self,
        matcher: Any,
        num_workers: int = 1,
        verbose: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            matcher: Fingerprint matcher with match(img1, img2) method
            num_workers: Number of parallel workers for score computation
            verbose: Whether to print progress
        """
        self.matcher = matcher
        self.num_workers = num_workers
        self.verbose = verbose
    
    def _compute_score(
        self,
        pair: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """
        Compute similarity score for a pair.
        
        Args:
            pair: Tuple of (image1, image2)
            
        Returns:
            Similarity score
        """
        try:
            return self.matcher.match(pair[0], pair[1])
        except Exception as e:
            if self.verbose:
                print(f"Warning: Matching failed - {e}")
            return 0.0
    
    def compute_scores(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
        label: str = "pairs"
    ) -> np.ndarray:
        """
        Compute scores for multiple pairs.
        
        Args:
            pairs: List of (image1, image2) tuples
            label: Label for progress printing
            
        Returns:
            Array of similarity scores
        """
        scores = []
        
        if self.verbose:
            print(f"Computing scores for {len(pairs)} {label}...")
        
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._compute_score, pair): i 
                    for i, pair in enumerate(pairs)
                }
                
                results = [None] * len(pairs)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                
                scores = results
        else:
            for i, pair in enumerate(pairs):
                if self.verbose and (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(pairs)}")
                scores.append(self._compute_score(pair))
        
        return np.array(scores)
    
    def evaluate(
        self,
        genuine_pairs: List[Tuple[np.ndarray, np.ndarray]],
        impostor_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> VerificationResult:
        """
        Perform complete verification evaluation.
        
        Args:
            genuine_pairs: List of genuine (same-subject) image pairs
            impostor_pairs: List of impostor (different-subject) image pairs
            
        Returns:
            VerificationResult with all metrics
        """
        start_time = time.time()
        
        # Compute scores
        genuine_scores = self.compute_scores(genuine_pairs, "genuine pairs")
        impostor_scores = self.compute_scores(impostor_pairs, "impostor pairs")
        
        # Compute metrics
        eer_result = compute_eer(genuine_scores, impostor_scores)
        
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        roc_curve = compute_roc_curve(y_true, y_scores)
        
        d_prime = compute_d_prime(genuine_scores, impostor_scores)
        
        far_at_frr_01, _ = compute_far_at_frr(genuine_scores, impostor_scores, 0.001)
        frr_at_far_1, _ = compute_frr_at_far(genuine_scores, impostor_scores, 0.01)
        
        processing_time = time.time() - start_time
        
        return VerificationResult(
            method_name=getattr(self.matcher, 'name', 'Unknown'),
            num_genuine=len(genuine_scores),
            num_impostor=len(impostor_scores),
            eer=eer_result.eer,
            eer_threshold=eer_result.threshold,
            auc=roc_curve.auc,
            d_prime=d_prime,
            far_at_frr_01=far_at_frr_01,
            frr_at_far_1=frr_at_far_1,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            roc_curve=roc_curve,
            eer_result=eer_result,
            processing_time=processing_time
        )
    
    def evaluate_from_dataset(
        self,
        dataset: Any,
        num_impostor_ratio: float = 1.0
    ) -> VerificationResult:
        """
        Evaluate using a FingerprintDataset.
        
        Args:
            dataset: FingerprintDataset instance
            num_impostor_ratio: Ratio of impostor to genuine pairs
            
        Returns:
            VerificationResult
        """
        from src.data.pair_generator import PairGenerator
        
        generator = PairGenerator(dataset)
        pairs = generator.generate_pairs(impostor_ratio=num_impostor_ratio)
        
        genuine_pairs = []
        impostor_pairs = []
        
        for pair in pairs:
            img1 = pair.sample1.load_image()
            img2 = pair.sample2.load_image()
            
            if pair.label == 1:
                genuine_pairs.append((img1, img2))
            else:
                impostor_pairs.append((img1, img2))
        
        return self.evaluate(genuine_pairs, impostor_pairs)
    
    def generate_report(
        self,
        result: VerificationResult,
        output_dir: str,
        include_plots: bool = True
    ) -> None:
        """
        Generate evaluation report with plots.
        
        Args:
            result: Verification result
            output_dir: Directory to save report
        """
        from src.evaluation.roc import plot_roc_curve
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = output_path / f"{result.method_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save text report
        report_path = output_path / f"{result.method_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(str(result))
        
        # Generate plots
        if include_plots:
            try:
                # ROC curve
                plot_roc_curve(
                    result.roc_curve,
                    title=f"ROC Curve - {result.method_name}",
                    save_path=str(output_path / f"{result.method_name}_roc.png")
                )
                
                # Score distributions
                plot_score_distributions(
                    result.genuine_scores,
                    result.impostor_scores,
                    result.eer_result,
                    title=f"Score Distributions - {result.method_name}",
                    save_path=str(output_path / f"{result.method_name}_scores.png")
                )
                
                # FAR/FRR curve
                plot_far_frr_curve(
                    result.eer_result,
                    title=f"FAR/FRR Curve - {result.method_name}",
                    save_path=str(output_path / f"{result.method_name}_far_frr.png")
                )
            except Exception as e:
                print(f"Warning: Could not generate plots - {e}")
        
        if self.verbose:
            print(f"Report saved to {output_path}")


class CrossSensorEvaluator:
    """
    Evaluator for cross-sensor fingerprint verification.
    
    This evaluator measures performance degradation when
    training and testing on different sensors.
    
    Cross-sensor scenarios:
    1. Same sensor (baseline)
    2. Train sensor A, test sensor B
    3. Sensor fusion
    """
    
    def __init__(
        self,
        matcher: Any,
        verbose: bool = True
    ):
        """
        Initialize cross-sensor evaluator.
        
        Args:
            matcher: Fingerprint matcher
            verbose: Whether to print progress
        """
        self.matcher = matcher
        self.verbose = verbose
        self.evaluator = VerificationEvaluator(matcher, verbose=verbose)
    
    def evaluate_cross_sensor(
        self,
        dataset_train: Any,
        dataset_test: Any,
        train_sensor_name: str = "Sensor A",
        test_sensor_name: str = "Sensor B"
    ) -> Dict[str, VerificationResult]:
        """
        Evaluate cross-sensor performance.
        
        Args:
            dataset_train: Training dataset (for reference)
            dataset_test: Testing dataset
            train_sensor_name: Name of training sensor
            test_sensor_name: Name of testing sensor
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Evaluate on training sensor (baseline)
        if self.verbose:
            print(f"\nEvaluating on {train_sensor_name} (baseline)...")
        results[train_sensor_name] = self.evaluator.evaluate_from_dataset(dataset_train)
        
        # Evaluate on test sensor
        if self.verbose:
            print(f"\nEvaluating on {test_sensor_name} (cross-sensor)...")
        results[test_sensor_name] = self.evaluator.evaluate_from_dataset(dataset_test)
        
        # Compute degradation
        eer_degradation = results[test_sensor_name].eer - results[train_sensor_name].eer
        
        if self.verbose:
            print(f"\nCross-Sensor Summary:")
            print(f"  {train_sensor_name} EER: {results[train_sensor_name].eer*100:.2f}%")
            print(f"  {test_sensor_name} EER: {results[test_sensor_name].eer*100:.2f}%")
            print(f"  EER Degradation: {eer_degradation*100:+.2f}%")
        
        return results
    
    def evaluate_multiple_sensors(
        self,
        datasets: Dict[str, Any]
    ) -> Dict[str, Dict[str, VerificationResult]]:
        """
        Evaluate all sensor combinations.
        
        Args:
            datasets: Dictionary mapping sensor names to datasets
            
        Returns:
            Nested dictionary of results [train_sensor][test_sensor]
        """
        sensor_names = list(datasets.keys())
        all_results = {name: {} for name in sensor_names}
        
        for train_name in sensor_names:
            for test_name in sensor_names:
                if self.verbose:
                    print(f"\nTrain: {train_name}, Test: {test_name}")
                
                result = self.evaluator.evaluate_from_dataset(datasets[test_name])
                all_results[train_name][test_name] = result
        
        return all_results
    
    def generate_cross_sensor_report(
        self,
        results: Dict[str, Dict[str, VerificationResult]],
        output_path: str
    ) -> None:
        """
        Generate cross-sensor evaluation report.
        
        Args:
            results: Results from evaluate_multiple_sensors
            output_path: Path to save report
        """
        lines = ["Cross-Sensor Evaluation Report", "=" * 40, ""]
        
        # EER matrix
        sensor_names = list(results.keys())
        lines.append("EER Matrix (Train x Test):")
        lines.append("-" * 40)
        
        # Header
        header = "Train\\Test".ljust(15)
        for name in sensor_names:
            header += name[:10].rjust(12)
        lines.append(header)
        
        # Rows
        for train_name in sensor_names:
            row = train_name[:15].ljust(15)
            for test_name in sensor_names:
                eer = results[train_name][test_name].eer
                row += f"{eer*100:10.2f}%"
            lines.append(row)
        
        lines.append("")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


def run_verification_experiment(
    matcher: Any,
    genuine_pairs: List[Tuple[np.ndarray, np.ndarray]],
    impostor_pairs: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> VerificationResult:
    """
    Convenience function to run a complete verification experiment.
    
    Args:
        matcher: Fingerprint matcher
        genuine_pairs: List of genuine pairs
        impostor_pairs: List of impostor pairs
        output_dir: Optional output directory for report
        verbose: Whether to print progress
        
    Returns:
        VerificationResult
    """
    evaluator = VerificationEvaluator(matcher, verbose=verbose)
    result = evaluator.evaluate(genuine_pairs, impostor_pairs)
    
    if verbose:
        print(result)
    
    if output_dir:
        evaluator.generate_report(result, output_dir)
    
    return result


def compare_matchers(
    matchers: Dict[str, Any],
    genuine_pairs: List[Tuple[np.ndarray, np.ndarray]],
    impostor_pairs: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, VerificationResult]:
    """
    Compare multiple matchers on the same dataset.
    
    Args:
        matchers: Dictionary mapping names to matcher instances
        genuine_pairs: List of genuine pairs
        impostor_pairs: List of impostor pairs
        output_dir: Optional output directory
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping names to results
    """
    results = {}
    
    for name, matcher in matchers.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating: {name}")
            print('='*50)
        
        evaluator = VerificationEvaluator(matcher, verbose=verbose)
        results[name] = evaluator.evaluate(genuine_pairs, impostor_pairs)
        
        if verbose:
            print(results[name])
    
    # Generate comparison
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ROC comparison
        try:
            roc_curves = {name: r.roc_curve for name, r in results.items()}
            compare_roc_curves(
                roc_curves,
                title="Matcher Comparison",
                save_path=str(output_path / "comparison_roc.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate comparison plot - {e}")
        
        # Summary table
        summary_lines = ["Matcher Comparison Summary", "=" * 60, ""]
        summary_lines.append(
            f"{'Method':<20} {'EER':>10} {'AUC':>10} {'d\\':>10}"
        )
        summary_lines.append("-" * 60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].eer)
        for name, result in sorted_results:
            summary_lines.append(
                f"{name:<20} {result.eer*100:>9.2f}% {result.auc:>10.4f} {result.d_prime:>10.2f}"
            )
        
        with open(output_path / "comparison_summary.txt", 'w') as f:
            f.write("\n".join(summary_lines))
    
    return results
