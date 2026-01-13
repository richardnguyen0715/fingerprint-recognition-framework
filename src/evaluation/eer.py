"""
Equal Error Rate (EER) computation for biometric verification.

This module implements EER computation and related metrics for
evaluating fingerprint verification systems.

Mathematical Background:
-----------------------
Equal Error Rate (EER):
The operating point where False Accept Rate equals False Reject Rate.

FAR(t) = FRR(t) at t = t_EER
EER = FAR(t_EER) = FRR(t_EER)

Lower EER indicates better system performance.

Interpretation:
- EER = 0%: Perfect system
- EER = 50%: Random guessing
- EER < 1%: Excellent system
- EER < 5%: Good system

Reference:
ISO/IEC 19795-1:2021 - Biometric performance testing and reporting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EERResult:
    """
    Container for EER computation results.
    
    Attributes:
        eer: Equal Error Rate
        threshold: Threshold at EER
        far_at_eer: FAR at EER threshold
        frr_at_eer: FRR at EER threshold
        far_array: All FAR values
        frr_array: All FRR values
        thresholds: All threshold values
    """
    eer: float
    threshold: float
    far_at_eer: float
    frr_at_eer: float
    far_array: np.ndarray
    frr_array: np.ndarray
    thresholds: np.ndarray
    
    def __str__(self) -> str:
        return (f"EER: {self.eer*100:.2f}% at threshold {self.threshold:.4f}\n"
                f"FAR: {self.far_at_eer*100:.2f}%, FRR: {self.frr_at_eer*100:.2f}%")


def compute_far_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    num_thresholds: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR and FRR at multiple thresholds.
    
    False Accept Rate (FAR):
    FAR(t) = #{impostor scores >= t} / #{total impostor scores}
    
    False Reject Rate (FRR):
    FRR(t) = #{genuine scores < t} / #{total genuine scores}
    
    Args:
        genuine_scores: Scores for genuine pairs (higher = more similar)
        impostor_scores: Scores for impostor pairs
        thresholds: Optional threshold values (computed if None)
        num_thresholds: Number of thresholds if not provided
        
    Returns:
        Tuple of (FAR array, FRR array, thresholds)
    """
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)
    
    # Determine thresholds
    if thresholds is None:
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        thresholds = np.linspace(min_score, max_score, num_thresholds)
    
    num_genuine = len(genuine_scores)
    num_impostor = len(impostor_scores)
    
    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))
    
    for i, t in enumerate(thresholds):
        # FAR: Impostors incorrectly accepted (score >= threshold)
        far[i] = np.sum(impostor_scores >= t) / num_impostor if num_impostor > 0 else 0
        
        # FRR: Genuines incorrectly rejected (score < threshold)
        frr[i] = np.sum(genuine_scores < t) / num_genuine if num_genuine > 0 else 0
    
    return far, frr, thresholds


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_thresholds: int = 10000
) -> EERResult:
    """
    Compute Equal Error Rate.
    
    Algorithm:
    ---------
    1. Compute FAR and FRR at multiple thresholds
    2. Find threshold where FAR ≈ FRR
    3. Interpolate for exact EER
    
    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        num_thresholds: Number of threshold points for precision
        
    Returns:
        EERResult object
    """
    far, frr, thresholds = compute_far_frr(
        genuine_scores, impostor_scores, 
        num_thresholds=num_thresholds
    )
    
    # Find crossing point where FAR = FRR
    # Using linear interpolation for better precision
    diff = far - frr
    
    # Find sign change
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) == 0:
        # No crossing found - use closest point
        eer_idx = np.argmin(np.abs(diff))
        eer = (far[eer_idx] + frr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
    else:
        # Interpolate at crossing point
        idx = sign_changes[0]
        
        # Linear interpolation
        t1, t2 = thresholds[idx], thresholds[idx + 1]
        far1, far2 = far[idx], far[idx + 1]
        frr1, frr2 = frr[idx], frr[idx + 1]
        
        # Solve: far1 + (far2-far1)*x = frr1 + (frr2-frr1)*x
        d_far = far2 - far1
        d_frr = frr2 - frr1
        
        if abs(d_far - d_frr) > 1e-10:
            x = (frr1 - far1) / (d_far - d_frr)
            x = np.clip(x, 0, 1)
        else:
            x = 0.5
        
        eer_threshold = t1 + (t2 - t1) * x
        eer = far1 + d_far * x
        eer_idx = idx
    
    return EERResult(
        eer=eer,
        threshold=eer_threshold,
        far_at_eer=far[eer_idx],
        frr_at_eer=frr[eer_idx],
        far_array=far,
        frr_array=frr,
        thresholds=thresholds
    )


def compute_eer_from_labels(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 10000
) -> EERResult:
    """
    Compute EER from labels and scores.
    
    Args:
        y_true: Ground truth labels (1=genuine, 0=impostor)
        y_scores: Similarity scores
        num_thresholds: Number of threshold points
        
    Returns:
        EERResult object
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    genuine_scores = y_scores[y_true == 1]
    impostor_scores = y_scores[y_true == 0]
    
    return compute_eer(genuine_scores, impostor_scores, num_thresholds)


def compute_far_at_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_frr: float
) -> Tuple[float, float]:
    """
    Compute FAR at a specific FRR value.
    
    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        target_frr: Target FRR value (e.g., 0.01 for 1%)
        
    Returns:
        Tuple of (FAR, threshold)
    """
    far, frr, thresholds = compute_far_frr(genuine_scores, impostor_scores)
    
    idx = np.argmin(np.abs(frr - target_frr))
    
    return far[idx], thresholds[idx]


def compute_frr_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float
) -> Tuple[float, float]:
    """
    Compute FRR at a specific FAR value.
    
    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        target_far: Target FAR value (e.g., 0.001 for 0.1%)
        
    Returns:
        Tuple of (FRR, threshold)
    """
    far, frr, thresholds = compute_far_frr(genuine_scores, impostor_scores)
    
    idx = np.argmin(np.abs(far - target_far))
    
    return frr[idx], thresholds[idx]


def compute_d_prime(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray
) -> float:
    """
    Compute d-prime (discriminability index).
    
    d' = |μ_genuine - μ_impostor| / √((σ²_genuine + σ²_impostor) / 2)
    
    Higher d' indicates better separation between genuine and impostor
    score distributions.
    
    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        
    Returns:
        d-prime value
    """
    mu_g = np.mean(genuine_scores)
    mu_i = np.mean(impostor_scores)
    
    var_g = np.var(genuine_scores)
    var_i = np.var(impostor_scores)
    
    pooled_std = np.sqrt((var_g + var_i) / 2)
    
    if pooled_std < 1e-10:
        return 0.0
    
    return abs(mu_g - mu_i) / pooled_std


def plot_score_distributions(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    eer_result: Optional[EERResult] = None,
    title: str = "Score Distributions",
    save_path: Optional[str] = None
) -> None:
    """
    Plot genuine and impostor score distributions.
    
    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        eer_result: Optional EER result to mark threshold
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    bins = np.linspace(
        min(np.min(genuine_scores), np.min(impostor_scores)),
        max(np.max(genuine_scores), np.max(impostor_scores)),
        50
    )
    
    plt.hist(genuine_scores, bins=bins, alpha=0.6, density=True,
             label='Genuine', color='green')
    plt.hist(impostor_scores, bins=bins, alpha=0.6, density=True,
             label='Impostor', color='red')
    
    # Mark EER threshold
    if eer_result is not None:
        plt.axvline(x=eer_result.threshold, color='blue', linestyle='--',
                   linewidth=2, label=f'EER threshold ({eer_result.eer*100:.2f}%)')
    
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_far_frr_curve(
    eer_result: EERResult,
    title: str = "FAR/FRR vs Threshold",
    save_path: Optional[str] = None
) -> None:
    """
    Plot FAR and FRR curves vs threshold.
    
    Args:
        eer_result: EER computation result
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(eer_result.thresholds, eer_result.far_array, 'r-',
             linewidth=2, label='FAR')
    plt.plot(eer_result.thresholds, eer_result.frr_array, 'b-',
             linewidth=2, label='FRR')
    
    # Mark EER point
    plt.axvline(x=eer_result.threshold, color='green', linestyle='--',
               linewidth=2, label=f'EER = {eer_result.eer*100:.2f}%')
    plt.axhline(y=eer_result.eer, color='gray', linestyle=':',
               linewidth=1)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_eer(
    results: Dict[str, EERResult],
    title: str = "EER Comparison"
) -> str:
    """
    Compare EER results from multiple methods.
    
    Args:
        results: Dictionary mapping method names to EERResult
        title: Title for the comparison
        
    Returns:
        Formatted comparison string
    """
    lines = [title, "=" * len(title), ""]
    
    # Sort by EER
    sorted_results = sorted(results.items(), key=lambda x: x[1].eer)
    
    for name, result in sorted_results:
        lines.append(f"{name}:")
        lines.append(f"  EER: {result.eer*100:.2f}%")
        lines.append(f"  Threshold: {result.threshold:.4f}")
        lines.append("")
    
    return "\n".join(lines)
