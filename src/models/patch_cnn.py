"""
Patch-based CNN for fingerprint recognition.

This module implements a patch-based CNN approach that extracts
local features around minutiae points. This hybrid approach combines
classical minutiae detection with deep learning feature extraction.

Mathematical Framework:
----------------------
Instead of processing the entire fingerprint image, this approach:
1. Extracts small patches centered at minutiae locations
2. Learns embeddings for each patch using a CNN
3. Aggregates patch embeddings for fingerprint-level matching

Advantages:
- More robust to global distortions
- Focus on discriminative regions (around minutiae)
- Better generalization across sensors
- Explicit incorporation of domain knowledge

Reference:
- Tang, Y., Gao, F., & Feng, J. (2017).
  "FingerNet: An Unified Deep Network for Fingerprint Minutiae Extraction." IJCAI.
"""

import numpy as np
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

@dataclass
class PatchCNNConfig:
    """Configuration for patch-based CNN model."""
    # Patch configuration
    patch_size: int = 64
    num_patches_per_image: int = 20
    
    # Network architecture
    embedding_dim: int = 64
    dropout: float = 0.3
    use_batch_norm: bool = True
    
    # Convolutional layers for patch processing
    conv_layers: List[Tuple[int, int, int, int]] = field(default_factory=lambda: [
        (32, 3, 1, 1),   # -> 32x32 after pool
        (64, 3, 1, 1),   # -> 16x16 after pool
        (128, 3, 1, 1),  # -> 8x8 after pool
    ])
    pool_size: int = 2
    
    # Aggregation method: "attention", "mean", "max", "netVLAD"
    aggregation: str = "attention"
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 32
    num_epochs: int = 100


# =============================================================================
# PATCH EXTRACTION
# =============================================================================

def extract_minutia_patch(
    image: np.ndarray,
    x: int,
    y: int,
    angle: float,
    patch_size: int = 64,
    rotate_to_align: bool = True
) -> Optional[np.ndarray]:
    """
    Extract a patch centered at a minutia location.
    
    The patch is optionally rotated to align with the minutia
    orientation, providing rotation invariance.
    
    Args:
        image: Fingerprint image
        x: Minutia x coordinate
        y: Minutia y coordinate
        angle: Minutia angle (radians)
        patch_size: Size of the extracted patch
        rotate_to_align: Whether to rotate patch to align with minutia
        
    Returns:
        Extracted patch or None if out of bounds
    """
    from scipy import ndimage
    
    h, w = image.shape[:2]
    half_size = patch_size // 2
    
    # Expand bounds for rotation
    margin = int(half_size * 1.5) if rotate_to_align else half_size
    
    # Check bounds with margin
    if (x - margin < 0 or x + margin >= w or
        y - margin < 0 or y + margin >= h):
        return None
    
    # Extract larger patch for rotation
    if rotate_to_align:
        large_patch = image[
            y - margin:y + margin,
            x - margin:x + margin
        ].copy()
        
        # Rotate to align with minutia angle
        angle_deg = np.degrees(angle)
        rotated = ndimage.rotate(large_patch, -angle_deg, reshape=False, mode='reflect')
        
        # Crop to final size
        center = rotated.shape[0] // 2
        patch = rotated[
            center - half_size:center + half_size,
            center - half_size:center + half_size
        ]
    else:
        patch = image[
            y - half_size:y + half_size,
            x - half_size:x + half_size
        ].copy()
    
    return patch


def extract_all_minutia_patches(
    image: np.ndarray,
    minutiae: List,
    config: PatchCNNConfig
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract patches for all minutiae in a fingerprint.
    
    Args:
        image: Fingerprint image
        minutiae: List of Minutia objects
        config: Patch extraction configuration
        
    Returns:
        List of (minutia_index, patch) tuples
    """
    patches = []
    
    for idx, m in enumerate(minutiae):
        patch = extract_minutia_patch(
            image, m.x, m.y, m.angle,
            config.patch_size, rotate_to_align=True
        )
        
        if patch is not None:
            patches.append((idx, patch))
    
    # Limit to top patches if needed
    if len(patches) > config.num_patches_per_image:
        # Sort by quality if available, otherwise random
        patches = patches[:config.num_patches_per_image]
    
    return patches


# =============================================================================
# PATCH CNN ARCHITECTURE
# =============================================================================

if TORCH_AVAILABLE:
    class PatchEncoder(nn.Module):
        """
        CNN encoder for individual patches.
        
        Architecture:
        Conv layers -> Global Average Pool -> Embedding
        
        Designed to process small patches around minutiae
        and produce compact feature vectors.
        """
        
        def __init__(self, config: PatchCNNConfig):
            """
            Initialize patch encoder.
            
            Args:
                config: Model configuration
            """
            super().__init__()
            self.config = config
            
            # Build convolutional layers
            layers = []
            in_channels = 1
            
            for out_channels, kernel_size, stride, padding in config.conv_layers:
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                ])
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.extend([
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(config.pool_size)
                ])
                in_channels = out_channels
            
            self.conv = nn.Sequential(*layers)
            
            # Global average pooling
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Embedding layer
            final_channels = config.conv_layers[-1][0]
            self.fc = nn.Sequential(
                nn.Linear(final_channels, config.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for single patch.
            
            Args:
                x: Input patch (batch, 1, H, W)
                
            Returns:
                Patch embedding (batch, embedding_dim)
            """
            features = self.conv(x)
            features = self.pool(features)
            features = features.view(features.size(0), -1)
            embedding = self.fc(features)
            return embedding
    
    
    class AttentionAggregator(nn.Module):
        """
        Attention-based aggregation of patch embeddings.
        
        Mathematical Formulation:
        -------------------------
        Given patch embeddings {e_1, ..., e_n}, compute:
        
        a_i = softmax(w^T * tanh(W * e_i))
        
        Final embedding: f = Σ_i a_i * e_i
        
        This allows the model to focus on the most discriminative
        patches while suppressing noisy ones.
        """
        
        def __init__(self, embedding_dim: int):
            """
            Initialize attention aggregator.
            
            Args:
                embedding_dim: Dimension of patch embeddings
            """
            super().__init__()
            
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 1)
            )
        
        def forward(
            self,
            embeddings: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Aggregate embeddings using attention.
            
            Args:
                embeddings: Patch embeddings (batch, num_patches, embedding_dim)
                mask: Optional mask for valid patches (batch, num_patches)
                
            Returns:
                Aggregated embedding (batch, embedding_dim)
            """
            # Compute attention scores
            scores = self.attention(embeddings).squeeze(-1)  # (batch, num_patches)
            
            # Apply mask
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            
            # Softmax
            weights = F.softmax(scores, dim=-1)  # (batch, num_patches)
            
            # Weighted sum
            weighted = weights.unsqueeze(-1) * embeddings
            aggregated = weighted.sum(dim=1)  # (batch, embedding_dim)
            
            return aggregated
    
    
    class NetVLAD(nn.Module):
        """
        NetVLAD aggregation for patch embeddings.
        
        Mathematical Formulation:
        -------------------------
        NetVLAD creates a fixed-size descriptor from variable
        number of local features using soft assignment to visual words.
        
        V(k) = Σ_i a_k(x_i) * (x_i - c_k)
        
        where:
        - c_k are cluster centers (visual words)
        - a_k(x_i) is soft assignment weight
        - V(k) is residual vector for cluster k
        
        Reference:
        Arandjelovic, R., et al. (2016). "NetVLAD: CNN architecture
        for weakly supervised place recognition." CVPR.
        """
        
        def __init__(
            self,
            embedding_dim: int,
            num_clusters: int = 8
        ):
            """
            Initialize NetVLAD.
            
            Args:
                embedding_dim: Dimension of patch embeddings
                num_clusters: Number of visual words
            """
            super().__init__()
            self.num_clusters = num_clusters
            self.embedding_dim = embedding_dim
            
            # Soft assignment weights
            self.conv = nn.Conv2d(
                embedding_dim, num_clusters,
                kernel_size=1, bias=False
            )
            
            # Cluster centers
            self.centers = nn.Parameter(
                torch.randn(num_clusters, embedding_dim)
            )
            nn.init.xavier_uniform_(self.centers)
        
        def forward(
            self,
            embeddings: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Aggregate embeddings using NetVLAD.
            
            Args:
                embeddings: Patch embeddings (batch, num_patches, embedding_dim)
                mask: Optional mask for valid patches
                
            Returns:
                VLAD descriptor (batch, num_clusters * embedding_dim)
            """
            batch_size, num_patches, _ = embeddings.shape
            
            # Reshape for conv1d (batch, embedding_dim, num_patches)
            x = embeddings.permute(0, 2, 1).unsqueeze(-1)
            
            # Soft assignment
            soft_assign = self.conv(x).squeeze(-1)  # (batch, num_clusters, num_patches)
            soft_assign = F.softmax(soft_assign, dim=1)
            
            # Compute residuals
            # (batch, num_patches, num_clusters, embedding_dim)
            residuals = embeddings.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(0)
            
            # Weight residuals by soft assignment
            # (batch, num_clusters, embedding_dim)
            soft_assign = soft_assign.permute(0, 2, 1)  # (batch, num_patches, num_clusters)
            
            if mask is not None:
                soft_assign = soft_assign * mask.unsqueeze(-1)
            
            vlad = torch.einsum('bnk,bnkd->bkd', soft_assign, residuals)
            
            # Intra-normalization
            vlad = F.normalize(vlad, p=2, dim=2)
            
            # Flatten and L2 normalize
            vlad = vlad.view(batch_size, -1)
            vlad = F.normalize(vlad, p=2, dim=1)
            
            return vlad
    
    
    class PatchCNN(nn.Module):
        """
        Complete patch-based CNN model for fingerprint recognition.
        
        Architecture:
        1. PatchEncoder: Extract embeddings from each patch
        2. Aggregator: Combine patch embeddings into global descriptor
        3. Final projection: Optional projection to target dimension
        
        This model processes a set of patches (extracted around minutiae)
        and produces a single fingerprint embedding.
        """
        
        def __init__(self, config: PatchCNNConfig):
            """
            Initialize patch CNN.
            
            Args:
                config: Model configuration
            """
            super().__init__()
            self.config = config
            
            # Patch encoder
            self.encoder = PatchEncoder(config)
            
            # Aggregator
            if config.aggregation == "attention":
                self.aggregator = AttentionAggregator(config.embedding_dim)
                output_dim = config.embedding_dim
            elif config.aggregation == "netVLAD":
                num_clusters = 8
                self.aggregator = NetVLAD(config.embedding_dim, num_clusters)
                output_dim = num_clusters * config.embedding_dim
            else:
                self.aggregator = None
                output_dim = config.embedding_dim
            
            # Final projection (if needed)
            if output_dim != config.embedding_dim:
                self.projection = nn.Linear(output_dim, config.embedding_dim)
            else:
                self.projection = None
        
        def encode_patches(
            self,
            patches: torch.Tensor
        ) -> torch.Tensor:
            """
            Encode a batch of patches.
            
            Args:
                patches: (batch * num_patches, 1, H, W)
                
            Returns:
                Embeddings (batch * num_patches, embedding_dim)
            """
            return self.encoder(patches)
        
        def forward(
            self,
            patches: torch.Tensor,
            num_patches: Optional[List[int]] = None
        ) -> torch.Tensor:
            """
            Forward pass for fingerprint.
            
            Args:
                patches: All patches concatenated (total_patches, 1, H, W)
                num_patches: List of patch counts per fingerprint in batch
                
            Returns:
                Fingerprint embeddings (batch, embedding_dim)
            """
            # Encode all patches
            patch_embeddings = self.encoder(patches)
            
            if num_patches is None:
                # Assume single fingerprint
                patch_embeddings = patch_embeddings.unsqueeze(0)
                num_patches = [patch_embeddings.size(1)]
            else:
                # Split embeddings by fingerprint
                batch_embeddings = []
                start = 0
                max_patches = max(num_patches)
                
                for n in num_patches:
                    emb = patch_embeddings[start:start+n]
                    
                    # Pad to max length
                    if n < max_patches:
                        pad = torch.zeros(
                            max_patches - n,
                            emb.size(1),
                            device=emb.device
                        )
                        emb = torch.cat([emb, pad], dim=0)
                    
                    batch_embeddings.append(emb)
                    start += n
                
                patch_embeddings = torch.stack(batch_embeddings)
            
            # Create mask
            batch_size, max_patches_batch, _ = patch_embeddings.shape
            mask = torch.zeros(batch_size, max_patches_batch, dtype=torch.bool,
                             device=patch_embeddings.device)
            for i, n in enumerate(num_patches):
                mask[i, :n] = True
            
            # Aggregate
            if self.config.aggregation == "mean":
                # Masked mean
                masked = patch_embeddings * mask.unsqueeze(-1)
                aggregated = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            elif self.config.aggregation == "max":
                # Masked max
                masked = patch_embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                aggregated, _ = masked.max(dim=1)
            else:
                aggregated = self.aggregator(patch_embeddings, mask)
            
            # Project if needed
            if self.projection is not None:
                aggregated = self.projection(aggregated)
            
            # L2 normalize
            aggregated = F.normalize(aggregated, p=2, dim=1)
            
            return aggregated
        
        def get_embedding_dim(self) -> int:
            """Return embedding dimension."""
            return self.config.embedding_dim


# =============================================================================
# MATCHER INTERFACE
# =============================================================================

class PatchCNNMatcher:
    """
    Patch-based CNN matcher for fingerprint verification.
    
    This matcher combines classical minutiae detection with
    deep learning feature extraction around minutiae points.
    
    Matching Algorithm:
    ------------------
    1. Extract minutiae from both fingerprints
    2. Extract patches around each minutia
    3. Encode patches using CNN
    4. Aggregate patch embeddings
    5. Compute cosine similarity
    """
    
    def __init__(
        self,
        config: Optional[PatchCNNConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize patch CNN matcher.
        
        Args:
            config: Model configuration
            model_path: Path to pretrained weights
            device: Computation device
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CNN models")
        
        self.config = config or PatchCNNConfig()
        self.device = torch.device(device)
        
        # Initialize model
        self.model = PatchCNN(self.config)
        self.model.to(self.device)
        
        # Load pretrained weights
        if model_path is not None:
            self.load(model_path)
        
        self.model.eval()
        
        # Minutiae extractor (lazy loading)
        self._extractor = None
    
    @property
    def name(self) -> str:
        return "PatchCNN"
    
    @property
    def extractor(self):
        """Get minutiae extractor (lazy loading)."""
        if self._extractor is None:
            from src.minutiae.minutiae_extraction import MinutiaeExtractor
            from src.minutiae.thinning import Thinner
            self._thinner = Thinner()
            self._extractor = MinutiaeExtractor()
        return self._extractor
    
    def load(self, path: str) -> None:
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def extract_patches_from_image(
        self,
        image: np.ndarray,
        minutiae: Optional[List] = None
    ) -> List[np.ndarray]:
        """
        Extract patches from fingerprint image.
        
        Args:
            image: Fingerprint image
            minutiae: Pre-extracted minutiae (optional)
            
        Returns:
            List of patch arrays
        """
        # Extract minutiae if not provided
        if minutiae is None:
            # Binarize image
            if image.max() > 1:
                binary = (image < 128).astype(np.uint8)
            else:
                binary = (image < 0.5).astype(np.uint8)
            
            skeleton = self._thinner.process(binary)
            minutiae = self.extractor.extract(skeleton)
        
        # Extract patches
        patches = extract_all_minutia_patches(image, minutiae, self.config)
        
        return [p for _, p in patches]
    
    def encode_patches(
        self,
        patches: List[np.ndarray]
    ) -> np.ndarray:
        """
        Encode patches to embedding.
        
        Args:
            patches: List of patch arrays
            
        Returns:
            Fingerprint embedding
        """
        if len(patches) == 0:
            return np.zeros(self.config.embedding_dim)
        
        # Preprocess patches
        patch_tensors = []
        for patch in patches:
            # Normalize
            if patch.max() > 1:
                patch = patch.astype(np.float32) / 255.0
            
            tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patch_tensors.append(tensor)
        
        # Stack and process
        batch = torch.cat(patch_tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(batch, [len(patches)])
        
        return embedding.cpu().numpy().flatten()
    
    def match(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        minutiae_a: Optional[List] = None,
        minutiae_b: Optional[List] = None
    ) -> float:
        """
        Compute similarity between two fingerprints.
        
        Args:
            sample_a: First fingerprint image
            sample_b: Second fingerprint image
            minutiae_a: Optional pre-extracted minutiae for first image
            minutiae_b: Optional pre-extracted minutiae for second image
            
        Returns:
            Similarity score in [0, 1]
        """
        # Extract patches
        patches_a = self.extract_patches_from_image(sample_a, minutiae_a)
        patches_b = self.extract_patches_from_image(sample_b, minutiae_b)
        
        if len(patches_a) == 0 or len(patches_b) == 0:
            return 0.0
        
        # Encode
        emb_a = self.encode_patches(patches_a)
        emb_b = self.encode_patches(patches_b)
        
        # Cosine similarity
        similarity = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8
        )
        
        # Map to [0, 1]
        similarity = (similarity + 1) / 2
        
        return float(similarity)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

if TORCH_AVAILABLE:
    class PatchPairDataset(Dataset):
        """
        Dataset for training patch-based CNN with contrastive loss.
        
        Each sample is a pair of fingerprints represented as sets of patches.
        """
        
        def __init__(
            self,
            pairs: List[Tuple[List[np.ndarray], List[np.ndarray], int]],
            config: PatchCNNConfig
        ):
            """
            Initialize dataset.
            
            Args:
                pairs: List of (patches1, patches2, label) tuples
                config: Model configuration
            """
            self.pairs = pairs
            self.config = config
        
        def __len__(self) -> int:
            return len(self.pairs)
        
        def __getitem__(
            self, idx: int
        ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
            """
            Get a pair of patch sets.
            
            Returns:
                Tuple of (patches1, patches2, label, num_patches1, num_patches2)
            """
            patches1, patches2, label = self.pairs[idx]
            
            # Convert to tensors
            def process_patches(patches):
                tensors = []
                for p in patches[:self.config.num_patches_per_image]:
                    if p.max() > 1:
                        p = p.astype(np.float32) / 255.0
                    t = torch.from_numpy(p).float().unsqueeze(0)
                    tensors.append(t)
                return torch.stack(tensors) if tensors else torch.zeros(1, 1, 64, 64)
            
            t1 = process_patches(patches1)
            t2 = process_patches(patches2)
            
            return t1, t2, label, len(patches1), len(patches2)
    
    
    class PatchCNNTrainer:
        """
        Trainer for patch-based CNN model.
        """
        
        def __init__(
            self,
            model: PatchCNN,
            config: PatchCNNConfig,
            device: str = "cpu"
        ):
            """
            Initialize trainer.
            
            Args:
                model: PatchCNN model
                config: Training configuration
                device: Computation device
            """
            self.model = model
            self.config = config
            self.device = torch.device(device)
            
            self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            from src.models.cnn_embedding import ContrastiveLoss
            self.loss_fn = ContrastiveLoss(margin=1.0)
            
            self.history = {"train_loss": [], "val_loss": []}
        
        def train_epoch(self, dataloader: DataLoader) -> float:
            """Train one epoch."""
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            for patches1, patches2, labels, n1, n2 in dataloader:
                batch_size = patches1.size(0)
                
                # Flatten patches for encoding
                all_patches1 = patches1.view(-1, 1, 
                    self.config.patch_size, self.config.patch_size).to(self.device)
                all_patches2 = patches2.view(-1, 1,
                    self.config.patch_size, self.config.patch_size).to(self.device)
                
                labels = labels.to(self.device).float()
                
                self.optimizer.zero_grad()
                
                # Encode each fingerprint
                emb1 = self.model(all_patches1, n1.tolist())
                emb2 = self.model(all_patches2, n2.tolist())
                
                # Compute loss
                loss = self.loss_fn(emb1, emb2, labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
        
        def train(
            self,
            train_loader: DataLoader,
            num_epochs: Optional[int] = None
        ) -> Dict[str, List[float]]:
            """Full training loop."""
            num_epochs = num_epochs or self.config.num_epochs
            
            for epoch in range(num_epochs):
                train_loss = self.train_epoch(train_loader)
                self.history["train_loss"].append(train_loss)
                print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")
            
            return self.history


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_patch_cnn(
    config: Optional[PatchCNNConfig] = None,
    aggregation: str = "attention"
) -> 'PatchCNN':
    """
    Factory function to create patch CNN model.
    
    Args:
        config: Model configuration
        aggregation: Aggregation method
        
    Returns:
        PatchCNN model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for patch CNN models")
    
    config = config or PatchCNNConfig()
    config.aggregation = aggregation
    
    return PatchCNN(config)
