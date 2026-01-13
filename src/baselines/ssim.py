"""
Image-based baseline matchers for fingerprint recognition.

This module implements classical image similarity metrics used as
baselines for fingerprint verification:
- Mean Squared Error (MSE)
- Normalized Cross-Correlation (NCC)
- Structural Similarity Index (SSIM)

These methods compare images pixel-by-pixel without extracting
fingerprint-specific features.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage


# =============================================================================
# BASE CLASSES
# =============================================================================


class FingerprintMatcher(ABC):
    """
    Abstract base class for fingerprint matchers.
    
    All fingerprint matching algorithms should inherit from this class
    and implement the match() method.
    """
    
    @abstractmethod
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute similarity score between two fingerprint samples.
        
        Args:
            sample_a: First fingerprint image (H x W, normalized to [0, 1])
            sample_b: Second fingerprint image (H x W, normalized to [0, 1])
            
        Returns:
            Similarity score (higher = more similar)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the matcher."""
        pass


# =============================================================================
# MEAN SQUARED ERROR (MSE) MATCHER
# =============================================================================


class MSEMatcher(FingerprintMatcher):
    """
    Mean Squared Error based fingerprint matcher.
    
    Mathematical Formulation:
    -------------------------
    Given two images I1 and I2 of size M x N:
    
    MSE = (1 / MN) * Σ_i Σ_j (I1(i,j) - I2(i,j))²
    
    Since MSE is a distance (lower = more similar), we convert to
    similarity score:
    
    Similarity = 1 / (1 + MSE)
    
    Properties:
    - Range: [0, 1] (1 = identical images)
    - Sensitive to intensity differences and misalignment
    - Simple and fast to compute
    
    Note: MSE is sensitive to minor translations and rotations,
    making it a weak baseline for fingerprint matching.
    """
    
    @property
    def name(self) -> str:
        return "MSE"
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute MSE-based similarity between two fingerprints.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Similarity score in [0, 1]
        """
        # Ensure same shape
        if sample_a.shape != sample_b.shape:
            raise ValueError(
                f"Image shapes must match: {sample_a.shape} vs {sample_b.shape}"
            )
        
        # Compute MSE
        mse = np.mean((sample_a - sample_b) ** 2)
        
        # Convert to similarity (higher = more similar)
        similarity = 1.0 / (1.0 + mse)
        
        return float(similarity)
    
    def compute_mse(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute raw MSE value.
        
        Args:
            sample_a: First image
            sample_b: Second image
            
        Returns:
            MSE value (lower = more similar)
        """
        return float(np.mean((sample_a - sample_b) ** 2))


# =============================================================================
# NORMALIZED CROSS-CORRELATION (NCC) MATCHER
# =============================================================================


class NCCMatcher(FingerprintMatcher):
    """
    Normalized Cross-Correlation based fingerprint matcher.
    
    Mathematical Formulation:
    -------------------------
    Given two images I1 and I2 with means μ1, μ2 and stds σ1, σ2:
    
    NCC = (1 / MN) * Σ_i Σ_j [(I1(i,j) - μ1) * (I2(i,j) - μ2)] / (σ1 * σ2)
    
    This is equivalent to the Pearson correlation coefficient.
    
    Properties:
    - Range: [-1, 1] (1 = perfectly correlated, -1 = inverse correlation)
    - Invariant to linear intensity transformations
    - Still sensitive to geometric transformations
    
    For similarity, we map to [0, 1]: similarity = (NCC + 1) / 2
    """
    
    @property
    def name(self) -> str:
        return "NCC"
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute NCC-based similarity between two fingerprints.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Similarity score in [0, 1]
        """
        if sample_a.shape != sample_b.shape:
            raise ValueError(
                f"Image shapes must match: {sample_a.shape} vs {sample_b.shape}"
            )
        
        # Flatten images
        a = sample_a.flatten().astype(np.float64)
        b = sample_b.flatten().astype(np.float64)
        
        # Compute means
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        
        # Compute standard deviations
        std_a = np.std(a)
        std_b = np.std(b)
        
        # Handle edge case of zero variance
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        
        # Compute NCC
        ncc = np.mean((a - mean_a) * (b - mean_b)) / (std_a * std_b)
        
        # Map from [-1, 1] to [0, 1]
        similarity = (ncc + 1.0) / 2.0
        
        return float(similarity)
    
    def compute_ncc(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute raw NCC value.
        
        Args:
            sample_a: First image
            sample_b: Second image
            
        Returns:
            NCC value in [-1, 1]
        """
        a = sample_a.flatten().astype(np.float64)
        b = sample_b.flatten().astype(np.float64)
        
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        std_a = np.std(a)
        std_b = np.std(b)
        
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        
        return float(np.mean((a - mean_a) * (b - mean_b)) / (std_a * std_b))


# =============================================================================
# STRUCTURAL SIMILARITY INDEX (SSIM) MATCHER
# =============================================================================


class SSIMMatcher(FingerprintMatcher):
    """
    Structural Similarity Index based fingerprint matcher.
    
    Mathematical Formulation:
    -------------------------
    SSIM decomposes image similarity into three components:
    
    1. Luminance: l(x,y) = (2*μx*μy + C1) / (μx² + μy² + C1)
    2. Contrast: c(x,y) = (2*σx*σy + C2) / (σx² + σy² + C2)
    3. Structure: s(x,y) = (σxy + C3) / (σx*σy + C3)
    
    Combined: SSIM(x,y) = l(x,y)^α * c(x,y)^β * s(x,y)^γ
    
    With α = β = γ = 1 and C3 = C2/2:
    SSIM(x,y) = (2*μx*μy + C1)(2*σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    
    where:
    - C1 = (K1*L)², C2 = (K2*L)² are stability constants
    - L is the dynamic range (1.0 for normalized images)
    - K1 = 0.01, K2 = 0.03 are default constants
    
    Properties:
    - Range: [-1, 1] (typically [0, 1] for similar images)
    - Captures structural information, not just pixel differences
    - More perceptually meaningful than MSE
    
    Reference:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    "Image quality assessment: from error visibility to structural similarity."
    IEEE TIP, 13(4), 600-612.
    """
    
    # Default SSIM constants
    DEFAULT_WINDOW_SIZE = 11
    DEFAULT_K1 = 0.01
    DEFAULT_K2 = 0.03
    DEFAULT_SIGMA = 1.5
    
    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        sigma: float = DEFAULT_SIGMA,
        k1: float = DEFAULT_K1,
        k2: float = DEFAULT_K2,
        data_range: float = 1.0,
        use_gaussian: bool = True
    ):
        """
        Initialize SSIM matcher.
        
        Args:
            window_size: Size of sliding window (must be odd)
            sigma: Standard deviation for Gaussian window
            k1: Constant for luminance (default 0.01)
            k2: Constant for contrast (default 0.03)
            data_range: Dynamic range of images (1.0 for [0,1], 255 for uint8)
            use_gaussian: Whether to use Gaussian-weighted window
        """
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range
        self.use_gaussian = use_gaussian
        
        # Precompute constants
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        
        # Create Gaussian window
        self._window = self._create_window()
    
    @property
    def name(self) -> str:
        return "SSIM"
    
    def _create_window(self) -> np.ndarray:
        """Create Gaussian window for local SSIM computation."""
        if not self.use_gaussian:
            return np.ones((self.window_size, self.window_size)) / (self.window_size ** 2)
        
        # Create 1D Gaussian
        coords = np.arange(self.window_size) - self.window_size // 2
        gauss = np.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        
        # Create 2D Gaussian
        window = np.outer(gauss, gauss)
        window = window / window.sum()
        
        return window
    
    def _compute_local_stats(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local mean and variance.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (local_mean, local_variance) arrays
        """
        # Local mean via convolution with window
        mu = ndimage.convolve(image, self._window, mode='reflect')
        
        # Local variance: E[X²] - E[X]²
        mu_sq = mu ** 2
        sigma_sq = ndimage.convolve(image ** 2, self._window, mode='reflect') - mu_sq
        
        return mu, sigma_sq
    
    def compute_ssim_map(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray
    ) -> np.ndarray:
        """
        Compute local SSIM map.
        
        Args:
            sample_a: First image
            sample_b: Second image
            
        Returns:
            SSIM map (same size as input)
        """
        sample_a = sample_a.astype(np.float64)
        sample_b = sample_b.astype(np.float64)
        
        # Compute local statistics
        mu_a, sigma_a_sq = self._compute_local_stats(sample_a)
        mu_b, sigma_b_sq = self._compute_local_stats(sample_b)
        
        # Covariance
        sigma_ab = (
            ndimage.convolve(sample_a * sample_b, self._window, mode='reflect')
            - mu_a * mu_b
        )
        
        # SSIM formula
        numerator = (2 * mu_a * mu_b + self.c1) * (2 * sigma_ab + self.c2)
        denominator = (mu_a ** 2 + mu_b ** 2 + self.c1) * (sigma_a_sq + sigma_b_sq + self.c2)
        
        ssim_map = numerator / denominator
        
        return ssim_map
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute SSIM-based similarity between two fingerprints.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Mean SSIM score in approximately [0, 1]
        """
        if sample_a.shape != sample_b.shape:
            raise ValueError(
                f"Image shapes must match: {sample_a.shape} vs {sample_b.shape}"
            )
        
        ssim_map = self.compute_ssim_map(sample_a, sample_b)
        
        # Return mean SSIM
        return float(np.mean(ssim_map))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def mse_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute MSE-based similarity score.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Similarity score in [0, 1]
    """
    matcher = MSEMatcher()
    return matcher.match(img1, img2)


def ncc_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute NCC-based similarity score.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Similarity score in [0, 1]
    """
    matcher = NCCMatcher()
    return matcher.match(img1, img2)


def ssim_score(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute SSIM-based similarity score.
    
    Args:
        img1: First image
        img2: Second image
        data_range: Dynamic range of images
        
    Returns:
        SSIM score
    """
    matcher = SSIMMatcher(data_range=data_range)
    return matcher.match(img1, img2)


def get_baseline_matcher(name: str, **kwargs) -> FingerprintMatcher:
    """
    Factory function to create baseline matchers.
    
    Args:
        name: Matcher name ('mse', 'ncc', 'ssim')
        **kwargs: Additional arguments for matcher initialization
        
    Returns:
        FingerprintMatcher instance
    """
    name = name.lower()
    
    if name == 'mse':
        return MSEMatcher()
    elif name == 'ncc':
        return NCCMatcher()
    elif name == 'ssim':
        return SSIMMatcher(**kwargs)
    else:
        raise ValueError(f"Unknown baseline matcher: {name}")
