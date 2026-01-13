"""
Evaluation module for fingerprint recognition.

This module provides comprehensive evaluation tools for biometric
verification systems, including:
- ROC curve computation and visualization
- EER (Equal Error Rate) calculation
- Complete verification protocol
- Cross-sensor evaluation

Usage:
------
from src.evaluation import (
    VerificationEvaluator,
    compute_eer,
    compute_roc_curve
)

# Evaluate a matcher
evaluator = VerificationEvaluator(matcher)
result = evaluator.evaluate(genuine_pairs, impostor_pairs)
print(f"EER: {result.eer * 100:.2f}%")
"""

from src.evaluation.metrics import compute_eer
from src.evaluation.roc import (
    ROCCurve,
    compute_roc_curve,
    compute_det_curve,
    plot_roc_curve,
    plot_det_curve,
    compare_roc_curves
)
from src.evaluation.eer import (
    EERResult,
    compute_eer,
    compute_eer_from_labels,
    compute_far_frr,
    compute_d_prime,
    compute_far_at_frr,
    compute_frr_at_far,
    plot_score_distributions,
    plot_far_frr_curve,
    compare_eer
)
from src.evaluation.verification import (
    VerificationResult,
    VerificationEvaluator,
    CrossSensorEvaluator,
    run_verification_experiment,
    compare_matchers
)

__all__ = [
    # ROC
    "ROCCurve",
    "compute_roc_curve",
    "compute_det_curve",
    "plot_roc_curve",
    "plot_det_curve",
    "compare_roc_curves",
    
    # EER
    "EERResult",
    "compute_eer",
    "compute_eer_from_labels",
    "compute_far_frr",
    "compute_d_prime",
    "compute_far_at_frr",
    "compute_frr_at_far",
    "plot_score_distributions",
    "plot_far_frr_curve",
    "compare_eer",
    
    # Verification
    "VerificationResult",
    "VerificationEvaluator",
    "CrossSensorEvaluator",
    "run_verification_experiment",
    "compare_matchers",
]
