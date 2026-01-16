"""
Hybrid model combining deep learning with classical minutiae matching.

This module implements hybrid approaches that leverage both:
1. CNN-based feature extraction or enhancement
2. Classical minutiae-based matching

Hybrid Approaches:
-----------------
1. CNN Enhancement + Classical Matching:
   - Use CNN to enhance fingerprint image
   - Apply classical minutiae extraction and matching

2. CNN Minutiae Detection + Classical Matching:
   - Use CNN to detect minutiae locations
   - Use classical descriptor matching

3. Fusion:
   - Combine scores from CNN and classical matchers
   - Learn optimal fusion weights

Reference:
- Tang, Y., et al. (2017). "FingerNet: An Unified Deep Network for 
  Fingerprint Minutiae Extraction." IJCAI.
- Cao, K., & Jain, A. K. (2019). "Automated Latent Fingerprint Recognition."
  IEEE TPAMI.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class FusionMethod(Enum):
    """Methods for fusing multiple matcher scores."""
    MEAN = "mean"
    WEIGHTED = "weighted"
    MAX = "max"
    MIN = "min"
    PRODUCT = "product"
    LEARNED = "learned"


@dataclass
class HybridConfig:
    """Configuration for hybrid model."""
    # CNN enhancement settings
    use_cnn_enhancement: bool = True
    enhancement_model: str = "unet"
    
    # Minutiae detection
    use_cnn_minutiae: bool = False
    minutiae_model: str = "fingernet"
    
    # Fusion settings
    fusion_method: FusionMethod = FusionMethod.WEIGHTED
    cnn_weight: float = 0.5
    classical_weight: float = 0.5
    
    # Component matchers
    cnn_matcher: str = "embedding"  # "embedding" or "patch"
    classical_matcher: str = "mcc"  # "mcc", "minutiae", "local_orientation"


# =============================================================================
# CNN ENHANCEMENT MODULE
# =============================================================================

if TORCH_AVAILABLE:
    class ConvBlock2d(nn.Module):
        """Double convolution block for U-Net."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None
        ):
            super().__init__()
            if mid_channels is None:
                mid_channels = out_channels
            
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            return self.double_conv(x)
    
    
    class DownBlock(nn.Module):
        """Downsampling block with maxpool and double conv."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock2d(in_channels, out_channels)
            )
        
        def forward(self, x):
            return self.maxpool_conv(x)
    
    
    class UpBlock(nn.Module):
        """Upsampling block with transpose conv and skip connection."""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = ConvBlock2d(in_channels, out_channels)
        
        def forward(self, x1, x2):
            x1 = self.up(x1)
            
            # Handle size differences
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]
            
            x1 = F.pad(x1, [
                diff_x // 2, diff_x - diff_x // 2,
                diff_y // 2, diff_y - diff_y // 2
            ])
            
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
    
    
    class EnhancementUNet(nn.Module):
        """
        U-Net architecture for fingerprint enhancement.
        
        This network learns to:
        1. Remove noise and artifacts
        2. Enhance ridge clarity
        3. Fill in broken ridges
        
        Architecture:
        Encoder -> Bottleneck -> Decoder with skip connections
        
        Input: Grayscale fingerprint image
        Output: Enhanced fingerprint image
        """
        
        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            features: List[int] = None
        ):
            """
            Initialize U-Net.
            
            Args:
                in_channels: Number of input channels
                out_channels: Number of output channels
                features: Feature dimensions for each level
            """
            super().__init__()
            
            if features is None:
                features = [64, 128, 256, 512]
            
            # Encoder
            self.inc = ConvBlock2d(in_channels, features[0])
            self.down1 = DownBlock(features[0], features[1])
            self.down2 = DownBlock(features[1], features[2])
            self.down3 = DownBlock(features[2], features[3])
            
            # Bottleneck
            self.down4 = DownBlock(features[3], features[3] * 2)
            
            # Decoder
            self.up1 = UpBlock(features[3] * 2, features[3])
            self.up2 = UpBlock(features[3], features[2])
            self.up3 = UpBlock(features[2], features[1])
            self.up4 = UpBlock(features[1], features[0])
            
            # Output
            self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input image (batch, 1, H, W)
                
            Returns:
                Enhanced image (batch, 1, H, W)
            """
            # Encoder
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Decoder
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            
            # Output
            x = self.outc(x)
            x = torch.sigmoid(x)
            
            return x
    
    
    class MinutiaeDetectionNet(nn.Module):
        """
        CNN for minutiae detection.
        
        This network predicts:
        1. Minutiae probability map
        2. Minutiae type (ending vs bifurcation)
        3. Minutiae orientation
        
        Based on FingerNet architecture.
        """
        
        def __init__(
            self,
            in_channels: int = 1,
            num_orientation_bins: int = 12
        ):
            """
            Initialize minutiae detection network.
            
            Args:
                in_channels: Number of input channels
                num_orientation_bins: Number of discrete orientation bins
            """
            super().__init__()
            
            self.num_orientation_bins = num_orientation_bins
            
            # Shared backbone
            self.backbone = nn.Sequential(
                ConvBlock2d(in_channels, 64),
                nn.MaxPool2d(2),
                ConvBlock2d(64, 128),
                nn.MaxPool2d(2),
                ConvBlock2d(128, 256),
                ConvBlock2d(256, 256),
            )
            
            # Minutiae probability head
            self.prob_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
            # Type classification head (ending vs bifurcation)
            self.type_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 2, kernel_size=1)  # 2 classes
            )
            
            # Orientation head
            self.orientation_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_orientation_bins, kernel_size=1)
            )
        
        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input image (batch, 1, H, W)
                
            Returns:
                Tuple of (prob_map, type_logits, orientation_logits)
            """
            features = self.backbone(x)
            
            prob_map = self.prob_head(features)
            type_logits = self.type_head(features)
            orientation_logits = self.orientation_head(features)
            
            return prob_map, type_logits, orientation_logits
        
        def detect_minutiae(
            self,
            x: torch.Tensor,
            threshold: float = 0.5
        ) -> List[List[Tuple[int, int, float, int]]]:
            """
            Detect minutiae from input image.
            
            Args:
                x: Input image (batch, 1, H, W)
                threshold: Probability threshold for detection
                
            Returns:
                List of minutiae per image: [(x, y, angle, type), ...]
            """
            prob_map, type_logits, orientation_logits = self.forward(x)
            
            batch_size = x.size(0)
            results = []
            
            for b in range(batch_size):
                minutiae = []
                
                # Get detections above threshold
                prob = prob_map[b, 0].cpu().numpy()
                type_pred = torch.argmax(type_logits[b], dim=0).cpu().numpy()
                orient_pred = torch.argmax(orientation_logits[b], dim=0).cpu().numpy()
                
                # Non-maximum suppression
                from scipy import ndimage
                local_max = ndimage.maximum_filter(prob, size=3) == prob
                
                y_coords, x_coords = np.where((prob > threshold) & local_max)
                
                for y, x_pos in zip(y_coords, x_coords):
                    # Scale back to original coordinates
                    orig_x = x_pos * 4  # Due to pooling
                    orig_y = y * 4
                    
                    # Get orientation
                    angle = orient_pred[y, x_pos] * (2 * np.pi / self.num_orientation_bins)
                    
                    # Get type (0=ending, 1=bifurcation)
                    mtype = type_pred[y, x_pos]
                    
                    minutiae.append((orig_x, orig_y, angle, mtype))
                
                results.append(minutiae)
            
            return results


# =============================================================================
# SCORE FUSION
# =============================================================================

class ScoreFusion:
    """
    Fusion of multiple matcher scores.
    
    Mathematical Formulations:
    -------------------------
    Mean: s = (s1 + s2 + ... + sn) / n
    
    Weighted: s = w1*s1 + w2*s2 + ... + wn*sn, Σwi = 1
    
    Product: s = (s1 * s2 * ... * sn)^(1/n)
    
    Min: s = min(s1, s2, ..., sn)
    
    Max: s = max(s1, s2, ..., sn)
    """
    
    def __init__(
        self,
        method: FusionMethod = FusionMethod.WEIGHTED,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize score fusion.
        
        Args:
            method: Fusion method
            weights: Optional weights for weighted fusion
        """
        self.method = method
        self.weights = weights
    
    def fuse(self, scores: List[float]) -> float:
        """
        Fuse multiple scores into a single score.
        
        Args:
            scores: List of similarity scores
            
        Returns:
            Fused score
        """
        if len(scores) == 0:
            return 0.0
        
        scores = np.array(scores)
        
        if self.method == FusionMethod.MEAN:
            return float(np.mean(scores))
        
        elif self.method == FusionMethod.WEIGHTED:
            if self.weights is None:
                weights = np.ones(len(scores)) / len(scores)
            else:
                weights = np.array(self.weights)
                weights = weights / weights.sum()
            return float(np.dot(scores, weights))
        
        elif self.method == FusionMethod.MAX:
            return float(np.max(scores))
        
        elif self.method == FusionMethod.MIN:
            return float(np.min(scores))
        
        elif self.method == FusionMethod.PRODUCT:
            # Geometric mean
            return float(np.prod(scores) ** (1.0 / len(scores)))
        
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")


if TORCH_AVAILABLE:
    class LearnedFusion(nn.Module):
        """
        Learned score fusion using a small neural network.
        
        Learns optimal non-linear combination of matcher scores.
        """
        
        def __init__(self, num_matchers: int):
            """
            Initialize learned fusion.
            
            Args:
                num_matchers: Number of matchers to fuse
            """
            super().__init__()
            
            self.fc = nn.Sequential(
                nn.Linear(num_matchers, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        
        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            """
            Fuse scores.
            
            Args:
                scores: (batch, num_matchers)
                
            Returns:
                Fused scores (batch, 1)
            """
            return self.fc(scores)


# =============================================================================
# HYBRID MATCHER
# =============================================================================

class HybridMatcher:
    """
    Hybrid fingerprint matcher combining CNN and classical methods.
    
    This matcher provides flexibility to combine:
    1. CNN-based global embedding
    2. Patch-based CNN around minutiae
    3. Classical minutiae matching
    4. Descriptor-based matching (MCC, Local Orientation)
    
    The combination can use various fusion strategies.
    """
    
    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        cnn_model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize hybrid matcher.
        
        Args:
            config: Hybrid model configuration
            cnn_model_path: Path to pretrained CNN weights
            device: Computation device
        """
        self.config = config or HybridConfig()
        self.device = device
        
        # Initialize component matchers
        self._init_matchers(cnn_model_path)
        
        # Initialize fusion
        self.fusion = ScoreFusion(
            method=self.config.fusion_method,
            weights=[self.config.cnn_weight, self.config.classical_weight]
        )
        
        # Enhancement model (optional)
        self.enhancement_model = None
        if self.config.use_cnn_enhancement and TORCH_AVAILABLE:
            self.enhancement_model = EnhancementUNet()
            self.enhancement_model.eval()
    
    def _init_matchers(self, cnn_model_path: Optional[str]) -> None:
        """Initialize component matchers."""
        # CNN matcher
        self.cnn_matcher = None
        if TORCH_AVAILABLE:
            if self.config.cnn_matcher == "embedding":
                from src.models.cnn_embedding import CNNEmbeddingMatcher
                self.cnn_matcher = CNNEmbeddingMatcher(
                    model_path=cnn_model_path,
                    device=self.device
                )
            elif self.config.cnn_matcher == "patch":
                from src.models.patch_cnn import PatchCNNMatcher
                self.cnn_matcher = PatchCNNMatcher(
                    model_path=cnn_model_path,
                    device=self.device
                )
        
        # Classical matcher
        if self.config.classical_matcher == "mcc":
            from src.descriptors.descriptor_matching import MCCMatcher
            self.classical_matcher = MCCMatcher()
        elif self.config.classical_matcher == "minutiae":
            from src.minutiae.minutiae_matching import MinutiaeMatcher
            self.classical_matcher = MinutiaeMatcher(random_state=42)
        elif self.config.classical_matcher == "local_orientation":
            from src.descriptors.descriptor_matching import LocalOrientationMatcher
            self.classical_matcher = LocalOrientationMatcher()
        else:
            self.classical_matcher = None
    
    @property
    def name(self) -> str:
        return "Hybrid"
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance fingerprint image using CNN.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Enhanced image
        """
        if self.enhancement_model is None:
            return image
        
        # Preprocess
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            enhanced = self.enhancement_model(tensor)
        
        return enhanced.cpu().numpy().squeeze()
    
    def match(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        return_components: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute similarity using hybrid matching.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            return_components: If True, return individual matcher scores
            
        Returns:
            Similarity score, optionally with component scores
        """
        # Enhance images if configured
        if self.config.use_cnn_enhancement:
            sample_a = self.enhance_image(sample_a)
            sample_b = self.enhance_image(sample_b)
        
        scores = []
        component_scores = {}
        
        # CNN matching
        if self.cnn_matcher is not None:
            cnn_score = self.cnn_matcher.match(sample_a, sample_b)
            scores.append(cnn_score)
            component_scores["cnn"] = cnn_score
        
        # Classical matching
        if self.classical_matcher is not None:
            try:
                classical_score = self._classical_match(sample_a, sample_b)
                scores.append(classical_score)
                component_scores["classical"] = classical_score
            except Exception as e:
                # Classical matching may fail on poor quality images
                if len(scores) == 0:
                    scores.append(0.0)
                component_scores["classical"] = 0.0
        
        # Fuse scores
        final_score = self.fusion.fuse(scores)
        
        if return_components:
            return final_score, component_scores
        return final_score
    
    def _classical_match(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray
    ) -> float:
        """
        Perform classical minutiae-based matching.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Similarity score
        """
        from src.minutiae.thinning import Thinner
        from src.minutiae.minutiae_extraction import MinutiaeExtractor
        
        thinner = Thinner()
        extractor = MinutiaeExtractor()
        
        # Preprocess
        def preprocess(img):
            if img.max() > 1:
                img = img / 255.0
            binary = (img < 0.5).astype(np.uint8)
            return binary
        
        binary_a = preprocess(sample_a)
        binary_b = preprocess(sample_b)
        
        # Skeletonize
        skeleton_a = thinner.process(binary_a)
        skeleton_b = thinner.process(binary_b)
        
        # Extract minutiae
        minutiae_a = extractor.extract(skeleton_a)
        minutiae_b = extractor.extract(skeleton_b)
        
        # Match based on configured method
        if self.config.classical_matcher == "mcc":
            from src.descriptors.mcc import MCCDescriptor
            
            mcc_a = MCCDescriptor()
            mcc_a.compute(minutiae_a)
            
            mcc_b = MCCDescriptor()
            mcc_b.compute(minutiae_b)
            
            return self.classical_matcher.match_descriptors(mcc_a, mcc_b)
        
        elif self.config.classical_matcher == "minutiae":
            return self.classical_matcher.match_minutiae(minutiae_a, minutiae_b)
        
        else:
            # Need orientation field for local orientation descriptor
            from src.enhancement.orientation_field import estimate_orientation_field
            
            orient_a = estimate_orientation_field(sample_a)
            orient_b = estimate_orientation_field(sample_b)
            
            desc_a = self.classical_matcher.compute_descriptors(orient_a, minutiae_a)
            desc_b = self.classical_matcher.compute_descriptors(orient_b, minutiae_b)
            
            return self.classical_matcher.match_descriptors(desc_a, desc_b)


# =============================================================================
# END-TO-END HYBRID PIPELINE
# =============================================================================

class HybridMatchingPipeline:
    """
    Complete hybrid matching pipeline for fingerprint verification.
    
    This pipeline combines all stages:
    1. Preprocessing
    2. Enhancement (optional CNN)
    3. Minutiae extraction (classical or CNN)
    4. Feature extraction (CNN embedding + descriptors)
    5. Score fusion
    
    The pipeline is configurable and can be adapted for
    different use cases and quality levels.
    """
    
    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        preprocessor=None,
        enhancer=None,
        device: str = "cpu"
    ):
        """
        Initialize hybrid pipeline.
        
        Args:
            config: Hybrid configuration
            preprocessor: Optional preprocessor
            enhancer: Optional classical enhancer (Gabor)
            device: Computation device
        """
        self.config = config or HybridConfig()
        self.preprocessor = preprocessor
        self.enhancer = enhancer
        self.device = device
        
        # Initialize hybrid matcher
        self.matcher = HybridMatcher(config, device=device)
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess fingerprint image.
        
        Args:
            image: Raw fingerprint image
            
        Returns:
            Preprocessed image
        """
        # Normalize
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        # Apply preprocessor
        if self.preprocessor is not None:
            image, _ = self.preprocessor(image)
        
        # Apply enhancer
        if self.enhancer is not None:
            from src.enhancement.orientation_field import estimate_orientation_field
            from src.enhancement.ridge_frequency import RidgeFrequencyEstimator
            
            orientation = estimate_orientation_field(image)
            freq_estimator = RidgeFrequencyEstimator()
            frequency = freq_estimator.estimate(image, orientation)
            image = self.enhancer.enhance(image, orientation, frequency)
        
        return image
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> float:
        """
        Match two fingerprint images.
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            
        Returns:
            Similarity score
        """
        # Preprocess
        processed1 = self.process_image(image1)
        processed2 = self.process_image(image2)
        
        # Match using hybrid matcher
        return self.matcher.match(processed1, processed2)
    
    def evaluate_pair(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a pair with detailed component scores.
        
        Args:
            image1: First fingerprint image
            image2: Second fingerprint image
            
        Returns:
            Dictionary with component scores
        """
        processed1 = self.process_image(image1)
        processed2 = self.process_image(image2)
        
        final_score, components = self.matcher.match(
            processed1, processed2, return_components=True
        )
        
        return {
            "final_score": final_score,
            **components
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_hybrid_matcher(
    cnn_type: str = "embedding",
    classical_type: str = "mcc",
    fusion_method: str = "weighted",
    cnn_weight: float = 0.5,
    use_enhancement: bool = False,
    device: str = "cpu"
) -> HybridMatcher:
    """
    Factory function to create hybrid matcher.
    
    Args:
        cnn_type: CNN matcher type ("embedding" or "patch")
        classical_type: Classical matcher type ("mcc", "minutiae", "local_orientation")
        fusion_method: Fusion method
        cnn_weight: Weight for CNN matcher
        use_enhancement: Whether to use CNN enhancement
        device: Computation device
        
    Returns:
        Configured HybridMatcher
    """
    config = HybridConfig(
        cnn_matcher=cnn_type,
        classical_matcher=classical_type,
        fusion_method=FusionMethod(fusion_method),
        cnn_weight=cnn_weight,
        classical_weight=1 - cnn_weight,
        use_cnn_enhancement=use_enhancement
    )
    
    return HybridMatcher(config, device=device)


def create_enhancement_model(
    model_type: str = "unet",
    pretrained_path: Optional[str] = None
) -> Optional['EnhancementUNet']:
    """
    Factory function to create enhancement model.
    
    Args:
        model_type: Model type ("unet")
        pretrained_path: Path to pretrained weights
        
    Returns:
        Enhancement model or None if PyTorch unavailable
    """
    if not TORCH_AVAILABLE:
        return None
    
    if model_type == "unet":
        model = EnhancementUNet()
        
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        return model
    
    raise ValueError(f"Unknown enhancement model: {model_type}")
