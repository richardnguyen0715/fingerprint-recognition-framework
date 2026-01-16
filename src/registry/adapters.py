"""
Matcher adapters for the UI layer.

This module provides adapter classes that wrap the core fingerprint
matching implementations to conform to the BaseMatcher interface
required by the UI.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.registry.matcher_interface import BaseMatcher, MatchResult
from src.registry.matcher_registry import (
    MatcherRegistry,
    ParameterInfo,
    ParameterType,
)


# =============================================================================
# SSIM ADAPTER
# =============================================================================


class SSIMMatcherAdapter(BaseMatcher):
    """
    Adapter for the SSIM baseline matcher.
    
    Wraps the SSIMMatcher from src.baselines.ssim to provide
    the BaseMatcher interface for the UI.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        """
        Initialize SSIM matcher adapter.
        
        Args:
            window_size: Size of sliding window (must be odd)
            sigma: Standard deviation for Gaussian window
            k1: Constant for luminance comparison
            k2: Constant for contrast comparison
        """
        from src.baselines.ssim import SSIMMatcher
        
        self._window_size = window_size
        self._sigma = sigma
        self._k1 = k1
        self._k2 = k2
        self._matcher = SSIMMatcher(
            window_size=window_size,
            sigma=sigma,
            k1=k1,
            k2=k2,
        )
    
    @property
    def name(self) -> str:
        return "SSIM (Structural Similarity)"
    
    @property
    def description(self) -> str:
        return (
            "Measures structural similarity between images using luminance, "
            "contrast, and structure comparison. Provides a perceptually "
            "meaningful similarity metric."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute SSIM similarity between two fingerprints."""
        # Compute SSIM map for detailed analysis
        ssim_map = self._matcher.compute_ssim_map(image_a, image_b)
        score = float(np.mean(ssim_map))
        
        # Compute regional SSIM for detailed breakdown
        h, w = ssim_map.shape
        h_mid, w_mid = h // 2, w // 2
        
        regions = {
            "top_left": float(np.mean(ssim_map[:h_mid, :w_mid])),
            "top_right": float(np.mean(ssim_map[:h_mid, w_mid:])),
            "bottom_left": float(np.mean(ssim_map[h_mid:, :w_mid])),
            "bottom_right": float(np.mean(ssim_map[h_mid:, w_mid:])),
        }
        
        details = {
            "mean_ssim": score,
            "min_ssim": float(np.min(ssim_map)),
            "max_ssim": float(np.max(ssim_map)),
            "std_ssim": float(np.std(ssim_map)),
            "regional_ssim": regions,
            "window_size": self._window_size,
            "sigma": self._sigma,
        }
        
        visualization_data = {
            "ssim_map": ssim_map,
        }
        
        return MatchResult(
            score=score,
            details=details,
            visualization_data=visualization_data,
        )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {
            "window_size": self._window_size,
            "sigma": self._sigma,
            "k1": self._k1,
            "k2": self._k2,
        }
    
    def set_parameters(self, **kwargs) -> None:
        from src.baselines.ssim import SSIMMatcher
        
        self._window_size = kwargs.get("window_size", self._window_size)
        self._sigma = kwargs.get("sigma", self._sigma)
        self._k1 = kwargs.get("k1", self._k1)
        self._k2 = kwargs.get("k2", self._k2)
        
        self._matcher = SSIMMatcher(
            window_size=self._window_size,
            sigma=self._sigma,
            k1=self._k1,
            k2=self._k2,
        )


# =============================================================================
# MSE ADAPTER
# =============================================================================


class MSEMatcherAdapter(BaseMatcher):
    """
    Adapter for the MSE baseline matcher.
    
    Wraps the MSEMatcher from src.baselines.ssim to provide
    the BaseMatcher interface for the UI.
    """
    
    def __init__(self):
        """Initialize MSE matcher adapter."""
        from src.baselines.ssim import MSEMatcher
        self._matcher = MSEMatcher()
    
    @property
    def name(self) -> str:
        return "MSE (Mean Squared Error)"
    
    @property
    def description(self) -> str:
        return (
            "Computes pixel-wise mean squared error between images. "
            "Simple baseline that is sensitive to misalignment and "
            "intensity differences."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute MSE similarity between two fingerprints."""
        score = self._matcher.match(image_a, image_b)
        raw_mse = self._matcher.compute_mse(image_a, image_b)
        
        # Compute per-region MSE
        h, w = image_a.shape
        h_mid, w_mid = h // 2, w // 2
        
        def region_mse(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.mean((a - b) ** 2))
        
        regions = {
            "top_left": region_mse(
                image_a[:h_mid, :w_mid], image_b[:h_mid, :w_mid]
            ),
            "top_right": region_mse(
                image_a[:h_mid, w_mid:], image_b[:h_mid, w_mid:]
            ),
            "bottom_left": region_mse(
                image_a[h_mid:, :w_mid], image_b[h_mid:, :w_mid]
            ),
            "bottom_right": region_mse(
                image_a[h_mid:, w_mid:], image_b[h_mid:, w_mid:]
            ),
        }
        
        # Compute difference image
        diff_image = np.abs(image_a - image_b)
        
        details = {
            "raw_mse": raw_mse,
            "similarity_score": score,
            "max_pixel_diff": float(np.max(diff_image)),
            "regional_mse": regions,
        }
        
        visualization_data = {
            "difference_image": diff_image,
        }
        
        return MatchResult(
            score=score,
            details=details,
            visualization_data=visualization_data,
        )


# =============================================================================
# NCC ADAPTER
# =============================================================================


class NCCMatcherAdapter(BaseMatcher):
    """
    Adapter for the NCC baseline matcher.
    
    Wraps the NCCMatcher from src.baselines.ssim to provide
    the BaseMatcher interface for the UI.
    """
    
    def __init__(self):
        """Initialize NCC matcher adapter."""
        from src.baselines.ssim import NCCMatcher
        self._matcher = NCCMatcher()
    
    @property
    def name(self) -> str:
        return "NCC (Normalized Cross-Correlation)"
    
    @property
    def description(self) -> str:
        return (
            "Computes normalized cross-correlation (Pearson correlation) "
            "between images. Invariant to linear intensity transformations "
            "but sensitive to geometric changes."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute NCC similarity between two fingerprints."""
        score = self._matcher.match(image_a, image_b)
        raw_ncc = self._matcher.compute_ncc(image_a, image_b)
        
        # Compute local correlation map for visualization
        # Using a sliding window approach
        window_size = 11
        pad = window_size // 2
        
        # Pad images
        img_a_padded = np.pad(image_a, pad, mode='reflect')
        img_b_padded = np.pad(image_b, pad, mode='reflect')
        
        h, w = image_a.shape
        correlation_map = np.zeros((h, w), dtype=np.float32)
        
        # Compute local correlation at each pixel
        for i in range(h):
            for j in range(w):
                patch_a = img_a_padded[i:i+window_size, j:j+window_size].flatten()
                patch_b = img_b_padded[i:i+window_size, j:j+window_size].flatten()
                
                # Normalize patches
                a_norm = patch_a - np.mean(patch_a)
                b_norm = patch_b - np.mean(patch_b)
                
                std_a = np.std(patch_a)
                std_b = np.std(patch_b)
                
                if std_a > 1e-10 and std_b > 1e-10:
                    correlation_map[i, j] = np.dot(a_norm, b_norm) / (
                        len(patch_a) * std_a * std_b
                    )
                else:
                    correlation_map[i, j] = 0.0
        
        # Normalize to [0, 1] for display
        correlation_map = (correlation_map + 1) / 2
        
        details = {
            "raw_ncc": raw_ncc,
            "similarity_score": score,
            "mean_a": float(np.mean(image_a)),
            "mean_b": float(np.mean(image_b)),
            "std_a": float(np.std(image_a)),
            "std_b": float(np.std(image_b)),
        }
        
        visualization_data = {
            "correlation_map": correlation_map,
        }
        
        return MatchResult(
            score=score,
            details=details,
            visualization_data=visualization_data,
        )
        


# =============================================================================
# MINUTIAE ADAPTER
# =============================================================================


class MinutiaeMatcherAdapter(BaseMatcher):
    """
    Adapter for the minutiae-based matcher.
    
    Wraps the MinutiaeMatchingPipeline from src.minutiae.minutiae_matching
    to provide the BaseMatcher interface for the UI.
    """
    
    def __init__(
        self,
        distance_threshold: float = 15.0,
        angle_threshold: float = 0.26,
        min_matched_minutiae: int = 8,
        ransac_iterations: int = 1000,
    ):
        """
        Initialize minutiae matcher adapter.
        
        Args:
            distance_threshold: Maximum distance for minutiae pairing (pixels)
            angle_threshold: Maximum angle difference for pairing (radians)
            min_matched_minutiae: Minimum matches for valid comparison
            ransac_iterations: Number of RANSAC iterations
        """
        self._distance_threshold = distance_threshold
        self._angle_threshold = angle_threshold
        self._min_matched_minutiae = min_matched_minutiae
        self._ransac_iterations = ransac_iterations
        self._create_pipeline()
    
    def _create_pipeline(self) -> None:
        """Create the minutiae matching pipeline."""
        from src.minutiae.minutiae_matching import (
            MinutiaeMatchingPipeline,
            MinutiaeMatcher,
        )
        
        matcher = MinutiaeMatcher(
            distance_threshold=self._distance_threshold,
            angle_threshold=self._angle_threshold,
            min_matched_minutiae=self._min_matched_minutiae,
            ransac_iterations=self._ransac_iterations,
            random_state=42  # Fixed seed for reproducibility
        )
        self._pipeline = MinutiaeMatchingPipeline(matcher=matcher)
        self._matcher = matcher
    
    @property
    def name(self) -> str:
        return "Minutiae Matching"
    
    @property
    def description(self) -> str:
        return (
            "Compares fingerprints based on minutiae (ridge endings and "
            "bifurcations). Uses RANSAC for alignment and greedy matching "
            "for correspondence finding."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute minutiae-based similarity between two fingerprints."""
        # Extract minutiae from both images
        minutiae_a = self._pipeline.extract_minutiae(image_a)
        minutiae_b = self._pipeline.extract_minutiae(image_b)
        
        # Perform matching
        score, matches = self._matcher.match_minutiae(minutiae_a, minutiae_b)
        
        # Count minutiae by type
        from src.minutiae.minutiae_extraction import MinutiaeType
        
        endings_a = sum(1 for m in minutiae_a if m.minutiae_type == MinutiaeType.ENDING)
        bifurcations_a = len(minutiae_a) - endings_a
        endings_b = sum(1 for m in minutiae_b if m.minutiae_type == MinutiaeType.ENDING)
        bifurcations_b = len(minutiae_b) - bifurcations_a
        
        # Compute match statistics
        if matches:
            avg_distance = sum(m.distance for m in matches) / len(matches)
            avg_angle_diff = sum(m.angle_diff for m in matches) / len(matches)
        else:
            avg_distance = 0.0
            avg_angle_diff = 0.0
        
        details = {
            "minutiae_count_a": len(minutiae_a),
            "minutiae_count_b": len(minutiae_b),
            "endings_a": endings_a,
            "bifurcations_a": bifurcations_a,
            "endings_b": endings_b,
            "bifurcations_b": bifurcations_b,
            "matched_pairs": len(matches),
            "average_distance": avg_distance,
            "average_angle_diff": avg_angle_diff,
            "distance_threshold": self._distance_threshold,
            "angle_threshold": self._angle_threshold,
        }
        
        visualization_data = {
            "minutiae_a": [m.to_dict() for m in minutiae_a],
            "minutiae_b": [m.to_dict() for m in minutiae_b],
            "matches": [
                {"idx1": m.idx1, "idx2": m.idx2, "distance": m.distance}
                for m in matches
            ],
        }
        
        return MatchResult(
            score=score,
            details=details,
            visualization_data=visualization_data,
        )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {
            "distance_threshold": self._distance_threshold,
            "angle_threshold": self._angle_threshold,
            "min_matched_minutiae": self._min_matched_minutiae,
            "ransac_iterations": self._ransac_iterations,
        }
    
    def set_parameters(self, **kwargs) -> None:
        self._distance_threshold = kwargs.get(
            "distance_threshold", self._distance_threshold
        )
        self._angle_threshold = kwargs.get(
            "angle_threshold", self._angle_threshold
        )
        self._min_matched_minutiae = kwargs.get(
            "min_matched_minutiae", self._min_matched_minutiae
        )
        self._ransac_iterations = kwargs.get(
            "ransac_iterations", self._ransac_iterations
        )
        self._create_pipeline()


# =============================================================================
# MCC ADAPTER
# =============================================================================


class MCCMatcherAdapter(BaseMatcher):
    """
    Adapter for the MCC (Minutia Cylinder Code) descriptor matcher.
    
    Wraps the MCC descriptor matching from src.descriptors.mcc
    to provide the BaseMatcher interface for the UI.
    """
    
    def __init__(
        self,
        radius: float = 70.0,
        num_spatial_cells: int = 16,
        num_angular_sections: int = 6,
        sigma_s: float = 7.0,
        sigma_d: float = 0.52,  # ~30 degrees
        mu: float = 0.1,
    ):
        """
        Initialize MCC matcher adapter.
        
        Args:
            radius: Cylinder radius (spatial extent)
            num_spatial_cells: Number of spatial cells along diameter
            num_angular_sections: Number of angular sections
            sigma_s: Spatial sigma parameter
            sigma_d: Directional sigma parameter
            mu: Binarization threshold
        """
        self._radius = radius
        self._num_spatial_cells = num_spatial_cells
        self._num_angular_sections = num_angular_sections
        self._sigma_s = sigma_s
        self._sigma_d = sigma_d
        self._mu = mu
        self._create_config()
    
    def _create_config(self) -> None:
        """Create MCC configuration."""
        from src.descriptors.mcc import MCCConfig
        
        self._config = MCCConfig(
            radius=self._radius,
            num_spatial_cells=self._num_spatial_cells,
            num_angular_sections=self._num_angular_sections,
            sigma_s=self._sigma_s,
            sigma_d=self._sigma_d,
            mu=self._mu,
        )
    
    @property
    def name(self) -> str:
        return "MCC (Minutia Cylinder Code)"
    
    @property
    def description(self) -> str:
        return (
            "Uses Minutia Cylinder Code descriptors that encode spatial "
            "and directional relationships between minutiae. Provides "
            "robust matching through local descriptor comparison."
        )
    
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> MatchResult:
        """Compute MCC-based similarity between two fingerprints."""
        from src.minutiae.minutiae_matching import MinutiaeMatchingPipeline
        from src.descriptors.mcc import (
            MCCDescriptor,
            cylinder_similarity,
        )
        
        # Extract minutiae
        pipeline = MinutiaeMatchingPipeline()
        minutiae_a = pipeline.extract_minutiae(image_a)
        minutiae_b = pipeline.extract_minutiae(image_b)
        
        # Compute MCC descriptors
        mcc_a = MCCDescriptor(self._config)
        mcc_a.compute(minutiae_a)
        
        mcc_b = MCCDescriptor(self._config)
        mcc_b.compute(minutiae_b)
        
        # Match descriptors
        if len(mcc_a) == 0 or len(mcc_b) == 0:
            return MatchResult(
                score=0.0,
                details={
                    "valid_descriptors_a": len(mcc_a),
                    "valid_descriptors_b": len(mcc_b),
                    "matched_descriptors": 0,
                    "error": "Insufficient valid descriptors",
                },
            )
        
        # Compute pairwise similarities and find best matches
        similarities = []
        for idx_a, desc_a in mcc_a.descriptors:
            best_sim = 0.0
            for idx_b, desc_b in mcc_b.descriptors:
                sim = cylinder_similarity(desc_a, desc_b)
                best_sim = max(best_sim, sim)
            similarities.append(best_sim)
        
        # Compute overall score
        if similarities:
            score = float(np.mean(similarities))
        else:
            score = 0.0
        
        details = {
            "minutiae_count_a": len(minutiae_a),
            "minutiae_count_b": len(minutiae_b),
            "valid_descriptors_a": len(mcc_a),
            "valid_descriptors_b": len(mcc_b),
            "mean_similarity": score,
            "max_similarity": float(max(similarities)) if similarities else 0.0,
            "min_similarity": float(min(similarities)) if similarities else 0.0,
            "cylinder_radius": self._radius,
            "spatial_cells": self._num_spatial_cells,
            "angular_sections": self._num_angular_sections,
        }
        
        return MatchResult(
            score=score,
            details=details,
        )
    
    def get_current_parameters(self) -> Dict[str, Any]:
        return {
            "radius": self._radius,
            "num_spatial_cells": self._num_spatial_cells,
            "num_angular_sections": self._num_angular_sections,
            "sigma_s": self._sigma_s,
            "sigma_d": self._sigma_d,
            "mu": self._mu,
        }
    
    def set_parameters(self, **kwargs) -> None:
        self._radius = kwargs.get("radius", self._radius)
        self._num_spatial_cells = kwargs.get(
            "num_spatial_cells", self._num_spatial_cells
        )
        self._num_angular_sections = kwargs.get(
            "num_angular_sections", self._num_angular_sections
        )
        self._sigma_s = kwargs.get("sigma_s", self._sigma_s)
        self._sigma_d = kwargs.get("sigma_d", self._sigma_d)
        self._mu = kwargs.get("mu", self._mu)
        self._create_config()


# =============================================================================
# REGISTRATION
# =============================================================================


def register_all_matchers(registry: MatcherRegistry) -> None:
    """
    Register all matcher adapters with the registry.
    
    Args:
        registry: The registry to populate
    """
    # SSIM Matcher
    registry.register(
        matcher_id="ssim",
        name="SSIM (Structural Similarity)",
        description=(
            "Measures structural similarity between images using luminance, "
            "contrast, and structure comparison."
        ),
        category="baseline",
        factory=SSIMMatcherAdapter,
        parameters=[
            ParameterInfo(
                name="window_size",
                display_name="Window Size",
                param_type=ParameterType.INTEGER,
                default=11,
                description="Size of sliding window for local SSIM computation",
                min_value=3,
                max_value=31,
                step=2,
            ),
            ParameterInfo(
                name="sigma",
                display_name="Sigma",
                param_type=ParameterType.FLOAT,
                default=1.5,
                description="Standard deviation for Gaussian window",
                min_value=0.5,
                max_value=5.0,
                step=0.1,
            ),
            ParameterInfo(
                name="k1",
                display_name="K1",
                param_type=ParameterType.FLOAT,
                default=0.01,
                description="Constant for luminance comparison",
                min_value=0.001,
                max_value=0.1,
                step=0.001,
            ),
            ParameterInfo(
                name="k2",
                display_name="K2",
                param_type=ParameterType.FLOAT,
                default=0.03,
                description="Constant for contrast comparison",
                min_value=0.001,
                max_value=0.1,
                step=0.001,
            ),
        ],
        requires_preprocessing=False,
    )
    
    # MSE Matcher
    registry.register(
        matcher_id="mse",
        name="MSE (Mean Squared Error)",
        description=(
            "Computes pixel-wise mean squared error between images. "
            "Simple baseline metric."
        ),
        category="baseline",
        factory=MSEMatcherAdapter,
        parameters=[],
        requires_preprocessing=False,
    )
    
    # NCC Matcher
    registry.register(
        matcher_id="ncc",
        name="NCC (Normalized Cross-Correlation)",
        description=(
            "Computes normalized cross-correlation between images. "
            "Invariant to linear intensity transformations."
        ),
        category="baseline",
        factory=NCCMatcherAdapter,
        parameters=[],
        requires_preprocessing=False,
    )
    
    # Minutiae Matcher
    registry.register(
        matcher_id="minutiae",
        name="Minutiae Matching",
        description=(
            "Matches fingerprints based on minutiae points using "
            "RANSAC alignment and greedy pairing."
        ),
        category="minutiae",
        factory=MinutiaeMatcherAdapter,
        parameters=[
            ParameterInfo(
                name="distance_threshold",
                display_name="Distance Threshold",
                param_type=ParameterType.FLOAT,
                default=15.0,
                description="Maximum distance for minutiae pairing (pixels)",
                min_value=5.0,
                max_value=50.0,
                step=1.0,
            ),
            ParameterInfo(
                name="angle_threshold",
                display_name="Angle Threshold",
                param_type=ParameterType.FLOAT,
                default=0.26,
                description="Maximum angle difference for pairing (radians)",
                min_value=0.1,
                max_value=0.5,
                step=0.02,
            ),
            ParameterInfo(
                name="min_matched_minutiae",
                display_name="Min Matched Minutiae",
                param_type=ParameterType.INTEGER,
                default=8,
                description="Minimum matches for valid comparison",
                min_value=1,
                max_value=20,
                step=1,
            ),
            ParameterInfo(
                name="ransac_iterations",
                display_name="RANSAC Iterations",
                param_type=ParameterType.INTEGER,
                default=1000,
                description="Number of RANSAC iterations for alignment",
                min_value=100,
                max_value=5000,
                step=100,
            ),
        ],
        requires_preprocessing=True,
    )
    
    # MCC Matcher
    registry.register(
        matcher_id="mcc",
        name="MCC (Minutia Cylinder Code)",
        description=(
            "Uses Minutia Cylinder Code descriptors for robust "
            "minutiae-based matching."
        ),
        category="descriptor",
        factory=MCCMatcherAdapter,
        parameters=[
            ParameterInfo(
                name="radius",
                display_name="Cylinder Radius",
                param_type=ParameterType.FLOAT,
                default=70.0,
                description="Spatial extent of the cylinder",
                min_value=30.0,
                max_value=150.0,
                step=5.0,
            ),
            ParameterInfo(
                name="num_spatial_cells",
                display_name="Spatial Cells",
                param_type=ParameterType.INTEGER,
                default=16,
                description="Number of spatial cells along diameter",
                min_value=8,
                max_value=32,
                step=2,
            ),
            ParameterInfo(
                name="num_angular_sections",
                display_name="Angular Sections",
                param_type=ParameterType.INTEGER,
                default=6,
                description="Number of angular sections",
                min_value=4,
                max_value=12,
                step=1,
            ),
            ParameterInfo(
                name="sigma_s",
                display_name="Spatial Sigma",
                param_type=ParameterType.FLOAT,
                default=7.0,
                description="Spatial sigma parameter",
                min_value=3.0,
                max_value=15.0,
                step=0.5,
            ),
            ParameterInfo(
                name="mu",
                display_name="Binarization Threshold",
                param_type=ParameterType.FLOAT,
                default=0.1,
                description="Threshold for cylinder binarization",
                min_value=0.01,
                max_value=0.5,
                step=0.01,
            ),
        ],
        requires_preprocessing=True,
    )


# Auto-register when this module is imported
_registry = MatcherRegistry()
register_all_matchers(_registry)
