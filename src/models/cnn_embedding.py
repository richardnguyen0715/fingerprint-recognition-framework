"""
CNN-based fingerprint embedding model.

This module implements deep learning models for fingerprint recognition
using CNN architectures that learn discriminative embeddings.

Mathematical Framework:
----------------------
The CNN model learns a function f: R^(HxW) -> R^d that maps fingerprint
images to a d-dimensional embedding space where:
- Same-finger samples are close (small distance)
- Different-finger samples are far apart (large distance)

Training Objective:
------------------
1. Contrastive Loss:
   L = (1-y) * 0.5 * D^2 + y * 0.5 * max(0, m - D)^2
   where D = ||f(x1) - f(x2)||, y = 0 (genuine), 1 (impostor)

2. Triplet Loss:
   L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
   where a=anchor, p=positive, n=negative

Reference:
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015).
  "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class LossType(Enum):
    """Supported loss functions for training."""
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    ARCFACE = "arcface"


@dataclass
class CNNConfig:
    """Configuration for CNN embedding model."""
    # Input configuration
    input_size: Tuple[int, int] = (256, 256)
    input_channels: int = 1
    
    # Architecture
    embedding_dim: int = 128
    dropout: float = 0.5
    use_batch_norm: bool = True
    
    # Convolutional layers: (out_channels, kernel_size, stride, padding)
    conv_layers: List[Tuple[int, int, int, int]] = field(default_factory=lambda: [
        (32, 3, 1, 1),
        (64, 3, 1, 1),
        (128, 3, 1, 1),
        (256, 3, 1, 1),
    ])
    pool_size: int = 2
    fc_layers: List[int] = field(default_factory=lambda: [512, 256])
    
    # Training
    loss_type: LossType = LossType.CONTRASTIVE
    contrastive_margin: float = 1.0
    triplet_margin: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 100


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

if TORCH_AVAILABLE:
    class ContrastiveLoss(nn.Module):
        """
        Contrastive loss for siamese network training.
        
        Mathematical Formulation:
        -------------------------
        For embedding pair (z1, z2) with label y (0=genuine, 1=impostor):
        
        L = (1-y) * 0.5 * D^2 + y * 0.5 * max(0, margin - D)^2
        
        where D = ||z1 - z2||_2
        
        This encourages:
        - Genuine pairs (y=0): Minimize distance D
        - Impostor pairs (y=1): Push distance beyond margin
        """
        
        def __init__(self, margin: float = 1.0):
            """
            Initialize contrastive loss.
            
            Args:
                margin: Margin for impostor pairs
            """
            super().__init__()
            self.margin = margin
        
        def forward(
            self,
            embedding1: torch.Tensor,
            embedding2: torch.Tensor,
            labels: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute contrastive loss.
            
            Args:
                embedding1: First embeddings (batch_size, embedding_dim)
                embedding2: Second embeddings (batch_size, embedding_dim)
                labels: Binary labels (batch_size,), 0=genuine, 1=impostor
                
            Returns:
                Loss value
            """
            # Euclidean distance
            distances = F.pairwise_distance(embedding1, embedding2)
            
            # Contrastive loss
            genuine_loss = (1 - labels) * 0.5 * distances.pow(2)
            impostor_loss = labels * 0.5 * F.relu(self.margin - distances).pow(2)
            
            loss = torch.mean(genuine_loss + impostor_loss)
            return loss
    
    
    class TripletLoss(nn.Module):
        """
        Triplet loss for metric learning.
        
        Mathematical Formulation:
        -------------------------
        For triplet (anchor, positive, negative):
        
        L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
        
        This encourages:
        - Anchor-positive distance < Anchor-negative distance by at least margin
        """
        
        def __init__(self, margin: float = 0.5):
            """
            Initialize triplet loss.
            
            Args:
                margin: Margin between positive and negative distances
            """
            super().__init__()
            self.margin = margin
        
        def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute triplet loss.
            
            Args:
                anchor: Anchor embeddings (batch_size, embedding_dim)
                positive: Positive embeddings (batch_size, embedding_dim)
                negative: Negative embeddings (batch_size, embedding_dim)
                
            Returns:
                Loss value
            """
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
            
            loss = F.relu(pos_dist.pow(2) - neg_dist.pow(2) + self.margin)
            return torch.mean(loss)
    
    
    class ArcFaceLoss(nn.Module):
        """
        ArcFace loss for face/fingerprint recognition.
        
        Mathematical Formulation:
        -------------------------
        L = -log(exp(s * cos(θ_yi + m)) / 
                 (exp(s * cos(θ_yi + m)) + Σ_{j≠yi} exp(s * cos(θ_j))))
        
        where:
        - θ_j = arccos(W_j^T * x / ||W_j|| ||x||)
        - s = scale factor
        - m = angular margin
        
        Reference:
        Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019).
        "ArcFace: Additive Angular Margin Loss for Deep Face Recognition."
        """
        
        def __init__(
            self,
            embedding_dim: int,
            num_classes: int,
            scale: float = 30.0,
            margin: float = 0.5
        ):
            """
            Initialize ArcFace loss.
            
            Args:
                embedding_dim: Dimension of embeddings
                num_classes: Number of identity classes
                scale: Scale factor
                margin: Angular margin in radians
            """
            super().__init__()
            self.scale = scale
            self.margin = margin
            
            # Weight matrix (class centers)
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
            nn.init.xavier_uniform_(self.weight)
        
        def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute ArcFace loss.
            
            Args:
                embeddings: Normalized embeddings (batch_size, embedding_dim)
                labels: Class labels (batch_size,)
                
            Returns:
                Loss value
            """
            # Normalize embeddings and weights
            embeddings = F.normalize(embeddings, p=2, dim=1)
            weights = F.normalize(self.weight, p=2, dim=1)
            
            # Compute cosine similarity
            cos_theta = F.linear(embeddings, weights)
            
            # Clamp for numerical stability
            cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            
            # Get angle
            theta = torch.acos(cos_theta)
            
            # Add margin to target class
            one_hot = F.one_hot(labels, num_classes=cos_theta.size(1)).float()
            theta_with_margin = theta + self.margin * one_hot
            
            # Convert back to cosine
            cos_theta_m = torch.cos(theta_with_margin)
            
            # Scale and compute loss
            logits = self.scale * cos_theta_m
            loss = F.cross_entropy(logits, labels)
            
            return loss


# =============================================================================
# CNN ARCHITECTURE
# =============================================================================

if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        """
        Convolutional block with optional batch normalization.
        
        Architecture:
        Conv -> BatchNorm (optional) -> ReLU -> MaxPool
        """
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            pool_size: int = 2,
            use_batch_norm: bool = True
        ):
            """
            Initialize convolutional block.
            
            Args:
                in_channels: Number of input channels
                out_channels: Number of output channels
                kernel_size: Convolution kernel size
                stride: Convolution stride
                padding: Convolution padding
                pool_size: Max pooling kernel size
                use_batch_norm: Whether to use batch normalization
            """
            super().__init__()
            
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            ]
            
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.extend([
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size)
            ])
            
            self.block = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.block(x)
    
    
    class FingerprintCNN(nn.Module):
        """
        Custom CNN architecture for fingerprint embedding.
        
        Architecture Overview:
        ---------------------
        1. Feature extraction: Stack of ConvBlocks
        2. Global pooling: AdaptiveAvgPool2d
        3. Embedding: FC layers with dropout
        4. L2 normalization
        
        The network learns to map fingerprint images to a compact
        embedding space where similar fingerprints are close.
        """
        
        def __init__(self, config: CNNConfig):
            """
            Initialize fingerprint CNN.
            
            Args:
                config: Model configuration
            """
            super().__init__()
            self.config = config
            
            # Build convolutional layers
            conv_blocks = []
            in_channels = config.input_channels
            
            for out_channels, kernel_size, stride, padding in config.conv_layers:
                conv_blocks.append(
                    ConvBlock(
                        in_channels, out_channels,
                        kernel_size, stride, padding,
                        config.pool_size, config.use_batch_norm
                    )
                )
                in_channels = out_channels
            
            self.conv_blocks = nn.Sequential(*conv_blocks)
            
            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Calculate flattened size
            final_channels = config.conv_layers[-1][0]
            
            # Fully connected layers
            fc_layers = []
            in_features = final_channels
            
            for fc_size in config.fc_layers:
                fc_layers.extend([
                    nn.Linear(in_features, fc_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.dropout)
                ])
                in_features = fc_size
            
            # Final embedding layer
            fc_layers.append(nn.Linear(in_features, config.embedding_dim))
            
            self.fc = nn.Sequential(*fc_layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass to compute embedding.
            
            Args:
                x: Input images (batch_size, channels, height, width)
                
            Returns:
                L2-normalized embeddings (batch_size, embedding_dim)
            """
            # Convolutional feature extraction
            features = self.conv_blocks(x)
            
            # Global pooling
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
            
            # Embedding
            embedding = self.fc(features)
            
            # L2 normalize
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding
        
        def get_embedding_dim(self) -> int:
            """Return embedding dimension."""
            return self.config.embedding_dim
    
    
    class ResNetEmbedding(nn.Module):
        """
        ResNet-based embedding model using pretrained backbone.
        
        Uses a pretrained ResNet and adds embedding layers on top.
        Suitable for transfer learning scenarios.
        """
        
        def __init__(
            self,
            embedding_dim: int = 128,
            backbone: str = "resnet18",
            pretrained: bool = True,
            dropout: float = 0.5
        ):
            """
            Initialize ResNet embedding model.
            
            Args:
                embedding_dim: Output embedding dimension
                backbone: ResNet variant ("resnet18", "resnet34", "resnet50")
                pretrained: Whether to use pretrained weights
                dropout: Dropout rate
            """
            super().__init__()
            
            # Import torchvision for ResNet
            try:
                import torchvision.models as models
                
                if backbone == "resnet18":
                    self.backbone = models.resnet18(pretrained=pretrained)
                    fc_in = 512
                elif backbone == "resnet34":
                    self.backbone = models.resnet34(pretrained=pretrained)
                    fc_in = 512
                elif backbone == "resnet50":
                    self.backbone = models.resnet50(pretrained=pretrained)
                    fc_in = 2048
                else:
                    raise ValueError(f"Unknown backbone: {backbone}")
                
                # Modify first conv for grayscale input
                self.backbone.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                
                # Remove final FC layer
                self.backbone.fc = nn.Identity()
                
            except ImportError:
                raise ImportError("torchvision required for ResNet backbone")
            
            # Embedding layers
            self.embedding = nn.Sequential(
                nn.Linear(fc_in, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, embedding_dim)
            )
            
            self.embedding_dim = embedding_dim
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input images (batch_size, 1, height, width)
                
            Returns:
                L2-normalized embeddings
            """
            features = self.backbone(x)
            embedding = self.embedding(features)
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        
        def get_embedding_dim(self) -> int:
            """Return embedding dimension."""
            return self.embedding_dim


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

if TORCH_AVAILABLE:
    class FingerprintPairDataset(Dataset):
        """
        Dataset for pair-based training with contrastive loss.
        
        Generates pairs of fingerprint images with labels:
        - 0: Genuine pair (same finger)
        - 1: Impostor pair (different fingers)
        """
        
        def __init__(
            self,
            image_paths: List[Tuple[str, str, int]],
            transform=None,
            input_size: Tuple[int, int] = (256, 256)
        ):
            """
            Initialize dataset.
            
            Args:
                image_paths: List of (path1, path2, label) tuples
                transform: Optional image transforms
                input_size: Target image size
            """
            self.pairs = image_paths
            self.transform = transform
            self.input_size = input_size
        
        def __len__(self) -> int:
            return len(self.pairs)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
            """
            Get a pair of images.
            
            Args:
                idx: Index
                
            Returns:
                Tuple of (image1, image2, label)
            """
            import cv2
            
            path1, path2, label = self.pairs[idx]
            
            # Load images
            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            
            # Resize
            img1 = cv2.resize(img1, self.input_size)
            img2 = cv2.resize(img2, self.input_size)
            
            # Normalize to [0, 1]
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0
            
            # Apply transforms
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            # Convert to tensors with channel dimension
            img1 = torch.from_numpy(img1).unsqueeze(0)
            img2 = torch.from_numpy(img2).unsqueeze(0)
            
            return img1, img2, label
    
    
    class FingerprintTripletDataset(Dataset):
        """
        Dataset for triplet-based training.
        
        Generates triplets: (anchor, positive, negative)
        - Anchor and positive are from the same finger
        - Negative is from a different finger
        """
        
        def __init__(
            self,
            samples_by_subject: Dict[str, List[str]],
            transform=None,
            input_size: Tuple[int, int] = (256, 256),
            triplets_per_anchor: int = 10
        ):
            """
            Initialize dataset.
            
            Args:
                samples_by_subject: Dict mapping subject_id to list of image paths
                transform: Optional transforms
                input_size: Target image size
                triplets_per_anchor: Number of triplets per anchor image
            """
            self.samples_by_subject = samples_by_subject
            self.transform = transform
            self.input_size = input_size
            self.triplets_per_anchor = triplets_per_anchor
            
            # Build triplet list
            self.triplets = self._generate_triplets()
        
        def _generate_triplets(self) -> List[Tuple[str, str, str]]:
            """Generate all triplets."""
            triplets = []
            subjects = list(self.samples_by_subject.keys())
            
            for subject_id in subjects:
                samples = self.samples_by_subject[subject_id]
                
                if len(samples) < 2:
                    continue
                
                other_subjects = [s for s in subjects if s != subject_id]
                
                for anchor_path in samples:
                    # Select positive (same subject)
                    positive_options = [p for p in samples if p != anchor_path]
                    
                    for _ in range(self.triplets_per_anchor):
                        positive_path = np.random.choice(positive_options)
                        
                        # Select negative (different subject)
                        neg_subject = np.random.choice(other_subjects)
                        negative_path = np.random.choice(
                            self.samples_by_subject[neg_subject]
                        )
                        
                        triplets.append((anchor_path, positive_path, negative_path))
            
            return triplets
        
        def __len__(self) -> int:
            return len(self.triplets)
        
        def __getitem__(
            self, idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Get a triplet of images.
            
            Args:
                idx: Index
                
            Returns:
                Tuple of (anchor, positive, negative) tensors
            """
            import cv2
            
            anchor_path, positive_path, negative_path = self.triplets[idx]
            
            # Load and process images
            images = []
            for path in [anchor_path, positive_path, negative_path]:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.input_size)
                img = img.astype(np.float32) / 255.0
                
                if self.transform:
                    img = self.transform(img)
                
                img = torch.from_numpy(img).unsqueeze(0)
                images.append(img)
            
            return tuple(images)


# =============================================================================
# FINGERPRINT MATCHER INTERFACE
# =============================================================================

class CNNEmbeddingMatcher:
    """
    CNN-based fingerprint matcher using learned embeddings.
    
    This class provides the FingerprintMatcher interface for
    CNN embedding models.
    
    Matching Algorithm:
    ------------------
    1. Extract embeddings for both fingerprints
    2. Compute cosine similarity: sim = (e1 · e2) / (||e1|| ||e2||)
    3. Return similarity score
    """
    
    def __init__(
        self,
        config: Optional[CNNConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize CNN matcher.
        
        Args:
            config: Model configuration
            model_path: Path to pretrained model weights
            device: Computation device ("cpu" or "cuda")
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN models")
        
        self.config = config or CNNConfig()
        self.device = torch.device(device)
        
        # Initialize model
        self.model = FingerprintCNN(self.config)
        self.model.to(self.device)
        
        # Load pretrained weights if provided
        if model_path is not None:
            self.load(model_path)
        
        self.model.eval()
    
    @property
    def name(self) -> str:
        return "CNN_Embedding"
    
    def load(self, path: str) -> None:
        """
        Load model weights.
        
        Args:
            path: Path to weights file
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def save(self, path: str) -> None:
        """
        Save model weights.
        
        Args:
            path: Path to save weights
        """
        torch.save(self.model.state_dict(), path)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (H, W) or (H, W, 1)
            
        Returns:
            Preprocessed tensor (1, 1, H, W)
        """
        import cv2
        
        # Ensure 2D
        if image.ndim == 3:
            image = image[:, :, 0]
        
        # Resize
        target_size = self.config.input_size
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from fingerprint image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            Embedding vector
        """
        with torch.no_grad():
            tensor = self._preprocess(image)
            embedding = self.model(tensor)
            return embedding.cpu().numpy().flatten()
    
    def match(self, sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """
        Compute similarity between two fingerprints.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            
        Returns:
            Similarity score in [0, 1]
        """
        # Extract embeddings
        emb_a = self.extract_embedding(sample_a)
        emb_b = self.extract_embedding(sample_b)
        
        # Cosine similarity
        similarity = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        )
        
        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)


# =============================================================================
# TRAINING LOOP
# =============================================================================

if TORCH_AVAILABLE:
    class CNNTrainer:
        """
        Trainer for CNN embedding models.
        
        Supports:
        - Contrastive loss training (pairs)
        - Triplet loss training (triplets)
        - ArcFace loss training (classification)
        """
        
        def __init__(
            self,
            model: nn.Module,
            config: CNNConfig,
            device: str = "cpu"
        ):
            """
            Initialize trainer.
            
            Args:
                model: CNN model to train
                config: Training configuration
                device: Computation device
            """
            self.model = model
            self.config = config
            self.device = torch.device(device)
            
            self.model.to(self.device)
            
            # Optimizer
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            # Scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
            
            # Loss function
            if config.loss_type == LossType.CONTRASTIVE:
                self.loss_fn = ContrastiveLoss(config.contrastive_margin)
            elif config.loss_type == LossType.TRIPLET:
                self.loss_fn = TripletLoss(config.triplet_margin)
            else:
                raise ValueError(f"Unsupported loss type: {config.loss_type}")
            
            self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        
        def train_epoch_contrastive(
            self,
            dataloader: DataLoader
        ) -> float:
            """
            Train one epoch with contrastive loss.
            
            Args:
                dataloader: Training data loader
                
            Returns:
                Average loss for the epoch
            """
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for img1, img2, labels in dataloader:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                self.optimizer.zero_grad()
                emb1 = self.model(img1)
                emb2 = self.model(img2)
                
                # Compute loss
                loss = self.loss_fn(emb1, emb2, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
        
        def train_epoch_triplet(
            self,
            dataloader: DataLoader
        ) -> float:
            """
            Train one epoch with triplet loss.
            
            Args:
                dataloader: Training data loader
                
            Returns:
                Average loss for the epoch
            """
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for anchor, positive, negative in dataloader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                emb_a = self.model(anchor)
                emb_p = self.model(positive)
                emb_n = self.model(negative)
                
                # Compute loss
                loss = self.loss_fn(emb_a, emb_p, emb_n)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
        
        def validate(
            self,
            dataloader: DataLoader
        ) -> float:
            """
            Validate model.
            
            Args:
                dataloader: Validation data loader
                
            Returns:
                Validation loss
            """
            self.model.eval()
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    if len(batch) == 3:
                        if isinstance(batch[2], int) or batch[2].dim() == 1:
                            # Pair batch
                            img1, img2, labels = batch
                            img1 = img1.to(self.device)
                            img2 = img2.to(self.device)
                            labels = labels.to(self.device).float()
                            
                            emb1 = self.model(img1)
                            emb2 = self.model(img2)
                            loss = self.loss_fn(emb1, emb2, labels)
                        else:
                            # Triplet batch
                            anchor, positive, negative = batch
                            anchor = anchor.to(self.device)
                            positive = positive.to(self.device)
                            negative = negative.to(self.device)
                            
                            emb_a = self.model(anchor)
                            emb_p = self.model(positive)
                            emb_n = self.model(negative)
                            loss = self.loss_fn(emb_a, emb_p, emb_n)
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            return total_loss / num_batches if num_batches > 0 else 0.0
        
        def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            num_epochs: Optional[int] = None
        ) -> Dict[str, List[float]]:
            """
            Full training loop.
            
            Args:
                train_loader: Training data loader
                val_loader: Optional validation data loader
                num_epochs: Number of epochs (uses config if None)
                
            Returns:
                Training history
            """
            num_epochs = num_epochs or self.config.num_epochs
            
            train_fn = (
                self.train_epoch_triplet 
                if self.config.loss_type == LossType.TRIPLET 
                else self.train_epoch_contrastive
            )
            
            for epoch in range(num_epochs):
                # Training
                train_loss = train_fn(train_loader)
                self.history["train_loss"].append(train_loss)
                
                # Validation
                if val_loader is not None:
                    val_loss = self.validate(val_loader)
                    self.history["val_loss"].append(val_loss)
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
                
                # Update learning rate
                self.scheduler.step()
            
            return self.history


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cnn_model(
    config: Optional[CNNConfig] = None,
    backbone: str = "custom",
    pretrained: bool = False,
    **kwargs
) -> Union['FingerprintCNN', 'ResNetEmbedding']:
    """
    Factory function to create CNN models.
    
    Args:
        config: Model configuration
        backbone: Model backbone ("custom", "resnet18", "resnet34", "resnet50")
        pretrained: Use pretrained weights for ResNet
        **kwargs: Additional arguments
        
    Returns:
        CNN model instance
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for CNN models")
    
    config = config or CNNConfig()
    
    if backbone == "custom":
        return FingerprintCNN(config)
    else:
        return ResNetEmbedding(
            embedding_dim=config.embedding_dim,
            backbone=backbone,
            pretrained=pretrained,
            dropout=config.dropout
        )
