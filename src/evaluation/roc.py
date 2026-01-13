"""
ROC curve computation for biometric verification.

This module implements ROC (Receiver Operating Characteristic) curve
analysis for evaluating fingerprint verification systems.

Mathematical Background:
-----------------------
For a verification system with threshold t:
- True Positive Rate (TPR) = Genuine Accept Rate (GAR)
  TPR(t) = P(score > t | genuine pair)
  
- False Positive Rate (FPR) = False Accept Rate (FAR)
  FPR(t) = P(score > t | impostor pair)
  
- True Negative Rate (TNR) = 1 - FAR
- False Negative Rate (FNR) = False Reject Rate (FRR) = 1 - GAR

ROC Curve:
- Plot of TPR vs FPR for all possible thresholds
- Area Under Curve (AUC) measures overall performance
- Perfect system: AUC = 1.0
- Random system: AUC = 0.5

Reference:
Maltoni, D., et al. (2009). "Handbook of Fingerprint Recognition."
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ROCCurve:
    """
    Container for ROC curve data.
    
    Attributes:
        fpr: False Positive Rates (array)
        tpr: True Positive Rates (array)
        thresholds: Threshold values (array)
        auc: Area Under Curve
        eer: Equal Error Rate
        eer_threshold: Threshold at EER
    """
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    eer: float
    eer_threshold: float
    
    def get_tar_at_far(self, target_far: float) -> Tuple[float, float]:
        """
        Get True Accept Rate at a specific False Accept Rate.
        
        Args:
            target_far: Target FAR value
            
        Returns:
            Tuple of (TAR, threshold)
        """
        # Find index where FAR is closest to target
        idx = np.argmin(np.abs(self.fpr - target_far))
        return self.tpr[idx], self.thresholds[idx]
    
    def get_far_at_tar(self, target_tar: float) -> Tuple[float, float]:
        """
        Get False Accept Rate at a specific True Accept Rate.
        
        Args:
            target_tar: Target TAR value
            
        Returns:
            Tuple of (FAR, threshold)
        """
        idx = np.argmin(np.abs(self.tpr - target_tar))
        return self.fpr[idx], self.thresholds[idx]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
            "thresholds": self.thresholds.tolist(),
            "auc": self.auc,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold
        }


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 1000
) -> ROCCurve:
    """
    Compute ROC curve from ground truth and scores.
    
    Algorithm:
    ---------
    1. Sort scores and determine threshold values
    2. For each threshold, compute TPR and FPR
    3. Compute AUC using trapezoidal integration
    4. Find EER where FPR ≈ FNR
    
    Args:
        y_true: Ground truth labels (1=genuine, 0=impostor)
        y_scores: Similarity scores
        num_thresholds: Number of threshold points
        
    Returns:
        ROCCurve object with all metrics
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # Determine thresholds
    min_score = np.min(y_scores)
    max_score = np.max(y_scores)
    thresholds = np.linspace(max_score, min_score, num_thresholds)
    
    # Compute TPR and FPR for each threshold
    num_genuine = np.sum(y_true == 1)
    num_impostor = np.sum(y_true == 0)
    
    tpr = np.zeros(num_thresholds)
    fpr = np.zeros(num_thresholds)
    
    for i, thresh in enumerate(thresholds):
        # Predictions at this threshold
        predictions = (y_scores >= thresh).astype(int)
        
        # True positives: genuine pairs correctly accepted
        tp = np.sum((predictions == 1) & (y_true == 1))
        
        # False positives: impostor pairs incorrectly accepted
        fp = np.sum((predictions == 1) & (y_true == 0))
        
        tpr[i] = tp / num_genuine if num_genuine > 0 else 0
        fpr[i] = fp / num_impostor if num_impostor > 0 else 0
    
    # Compute AUC using trapezoidal rule
    # Sort by FPR for proper integration
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    auc = np.trapz(tpr_sorted, fpr_sorted)
    
    # Compute EER (where FPR = FNR = 1 - TPR)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return ROCCurve(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        auc=auc,
        eer=eer,
        eer_threshold=eer_threshold
    )


def compute_det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DET (Detection Error Tradeoff) curve.
    
    DET curve plots FRR vs FAR on a warped scale (normal deviate)
    which makes the curve more linear and easier to compare.
    
    Args:
        y_true: Ground truth labels
        y_scores: Similarity scores
        num_thresholds: Number of threshold points
        
    Returns:
        Tuple of (FAR, FRR, thresholds)
    """
    roc = compute_roc_curve(y_true, y_scores, num_thresholds)
    
    far = roc.fpr
    frr = 1 - roc.tpr
    
    return far, frr, roc.thresholds


def plot_roc_curve(
    roc: ROCCurve,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.
    
    Args:
        roc: ROCCurve object
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(roc.fpr, roc.tpr, 'b-', linewidth=2, 
             label=f'ROC (AUC = {roc.auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    # Mark EER point
    eer_idx = np.argmin(np.abs(roc.fpr - (1 - roc.tpr)))
    plt.plot(roc.fpr[eer_idx], roc.tpr[eer_idx], 'ro', 
             markersize=10, label=f'EER = {roc.eer:.4f}')
    
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (GAR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_det_curve(
    far: np.ndarray,
    frr: np.ndarray,
    title: str = "DET Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot DET curve with normal deviate scale.
    
    Args:
        far: False Accept Rates
        frr: False Reject Rates
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib and scipy required for DET plot")
        return
    
    # Convert to normal deviate scale
    # Clip to avoid inf values
    far_clipped = np.clip(far, 1e-6, 1 - 1e-6)
    frr_clipped = np.clip(frr, 1e-6, 1 - 1e-6)
    
    far_norm = stats.norm.ppf(far_clipped)
    frr_norm = stats.norm.ppf(frr_clipped)
    
    plt.figure(figsize=(8, 6))
    plt.plot(far_norm, frr_norm, 'b-', linewidth=2)
    
    # Reference line (equal error)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, x, 'k--', linewidth=1, label='EER Line')
    
    # Custom tick labels (in percentage)
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    tick_positions = stats.norm.ppf(ticks)
    tick_labels = [f'{t*100:.1f}%' for t in ticks]
    
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    
    plt.xlabel('False Accept Rate', fontsize=12)
    plt.ylabel('False Reject Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_roc_curves(
    roc_curves: Dict[str, ROCCurve],
    title: str = "ROC Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple ROC curves for comparison.
    
    Args:
        roc_curves: Dictionary mapping method names to ROCCurve objects
        title: Plot title
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))
    
    for (name, roc), color in zip(roc_curves.items(), colors):
        plt.plot(roc.fpr, roc.tpr, color=color, linewidth=2,
                label=f'{name} (AUC={roc.auc:.4f}, EER={roc.eer:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate (FAR)', fontsize=12)
    plt.ylabel('True Positive Rate (GAR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
