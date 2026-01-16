"""
Experiment: Patch CNN Model Training and Evaluation

This experiment trains and evaluates patch-based CNN models that combine
classical minutiae detection with deep learning feature extraction.

Models:
- Patch CNN with attention aggregation
- Patch CNN with mean/max aggregation

Expected Results:
- Better performance than pure minutiae matching
- More robust to image quality variations
- EER typically 3-7% with sufficient training data
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.data.fingerprint_dataset import FingerprintDataset
from src.data.pair_generator import PairGenerator
from src.evaluation import VerificationEvaluator
from src.utils.logger import get_logger

# Import shared experiment utilities
sys.path.insert(0, str(project_root / "experiments"))
try:
    from exp_cnn_embedding import load_pairs_from_dataset
except ImportError:
    # Fallback if import fails
    def load_pairs_from_dataset(dataset, num_impostor_ratio=1.0):
        """Load evaluation pairs from dataset."""
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

# Check for PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from src.models.patch_cnn import (
        PatchCNNConfig,
        PatchCNNMatcher,
        PatchCNN,
        PatchCNNTrainer,
        create_patch_cnn,
        extract_minutia_patch,
    )
    from src.minutiae.minutiae_extraction import MinutiaeExtractor
    from src.minutiae.thinning import Thinner
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Patch CNN experiments disabled.")


class PatchPairDataset(Dataset):
    """
    Dataset for patch-based pair training.
    
    Generates pairs of fingerprints with their patches extracted
    around minutiae points.
    """
    
    def __init__(
        self,
        image_pairs: list,
        config: PatchCNNConfig,
    ):
        """
        Initialize dataset.
        
        Args:
            image_pairs: List of (path1, path2, label) tuples
            config: Patch CNN configuration
        """
        self.pairs = image_pairs
        self.config = config
        
        # Minutiae extractor (lazy loading)
        self._extractor = None
        self._thinner = None
    
    def get_extractor(self):
        """Get minutiae extractor (lazy loading)."""
        if self._extractor is None:
            self._thinner = Thinner()
            self._extractor = MinutiaeExtractor()
        return self._extractor, self._thinner
    
    def extract_patches_from_image(self, image: np.ndarray) -> list:
        """Extract patches from fingerprint image."""
        import cv2
        
        extractor, thinner = self.get_extractor()
        
        # Binarize
        if image.max() > 1:
            binary = (image < 128).astype(np.uint8)
        else:
            binary = (image < 0.5).astype(np.uint8)
        
        # Extract minutiae
        skeleton = thinner.process(binary)
        minutiae = extractor.extract(skeleton)
        
        # Extract patches
        patches = []
        for m in minutiae:
            patch = extract_minutia_patch(
                image, m.x, m.y, m.angle,
                self.config.patch_size, rotate_to_align=True
            )
            if patch is not None:
                patches.append(patch)
            
            if len(patches) >= self.config.num_patches_per_image:
                break
        
        return patches
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        """
        Get a pair of fingerprints with patches.
        
        Returns:
            Tuple of (patches1, patches2, label, n1, n2)
        """
        import cv2
        
        path1, path2, label = self.pairs[idx]
        
        # Load images
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        
        # Normalize
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Extract patches
        patches1 = self.extract_patches_from_image(img1)
        patches2 = self.extract_patches_from_image(img2)
        
        # Ensure we have patches
        max_patches = self.config.num_patches_per_image
        
        # Pad with zeros if needed
        while len(patches1) < max_patches:
            patches1.append(np.zeros((self.config.patch_size, self.config.patch_size)))
        while len(patches2) < max_patches:
            patches2.append(np.zeros((self.config.patch_size, self.config.patch_size)))
        
        # Truncate if too many
        patches1 = patches1[:max_patches]
        patches2 = patches2[:max_patches]
        
        # Convert to tensors
        patches1 = torch.stack([torch.from_numpy(p).float() for p in patches1])
        patches2 = torch.stack([torch.from_numpy(p).float() for p in patches2])
        
        n1 = len(patches1)
        n2 = len(patches2)
        
        return patches1, patches2, label, n1, n2


def prepare_training_data(
    dataset: FingerprintDataset,
    validation_split: float = 0.2
):
    """
    Prepare training and validation data.
    
    Args:
        dataset: FingerprintDataset instance
        validation_split: Fraction for validation
        
    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    # Get all subjects
    subjects = list(dataset.subject_ids)
    np.random.shuffle(subjects)
    
    # Split subjects
    num_val = int(len(subjects) * validation_split)
    val_subjects = set(subjects[:num_val])
    train_subjects = set(subjects[num_val:])
    
    # Generate pairs for each split
    train_pairs = []
    val_pairs = []
    
    # Genuine pairs
    for subject_id in train_subjects:
        samples = dataset.get_subject_samples(subject_id)
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                train_pairs.append((
                    str(samples[i].path),
                    str(samples[j].path),
                    0  # Genuine = 0 for contrastive loss
                ))
    
    for subject_id in val_subjects:
        samples = dataset.get_subject_samples(subject_id)
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                val_pairs.append((
                    str(samples[i].path),
                    str(samples[j].path),
                    0
                ))
    
    # Impostor pairs
    all_train_samples = [s for s in dataset.samples if s.subject_id in train_subjects]
    all_val_samples = [s for s in dataset.samples if s.subject_id in val_subjects]
    
    num_train_genuine = len(train_pairs)
    num_val_genuine = len(val_pairs)
    
    # Generate equal number of impostor pairs
    for _ in range(num_train_genuine):
        s1, s2 = np.random.choice(len(all_train_samples), 2, replace=False)
        if all_train_samples[s1].subject_id != all_train_samples[s2].subject_id:
            train_pairs.append((
                str(all_train_samples[s1].path),
                str(all_train_samples[s2].path),
                1  # Impostor = 1
            ))
    
    for _ in range(num_val_genuine):
        s1, s2 = np.random.choice(len(all_val_samples), 2, replace=False)
        if all_val_samples[s1].subject_id != all_val_samples[s2].subject_id:
            val_pairs.append((
                str(all_val_samples[s1].path),
                str(all_val_samples[s2].path),
                1
            ))
    
    return train_pairs, val_pairs


def train_patch_cnn_model(
    train_pairs: list,
    val_pairs: list,
    config: PatchCNNConfig,
    num_epochs: int = 50,
    device: str = "cpu",
    output_dir: Path = None
):
    """
    Train a patch CNN model.
    
    Args:
        train_pairs: Training pairs
        val_pairs: Validation pairs
        config: Model configuration
        num_epochs: Number of epochs
        device: Computation device
        output_dir: Where to save model
        
    Returns:
        Trained model
    """
    print(f"\nTraining Patch CNN Model")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = PatchPairDataset(train_pairs, config)
    val_dataset = PatchPairDataset(val_pairs, config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = create_patch_cnn(config, aggregation=config.aggregation)
    
    # Create trainer
    trainer = PatchCNNTrainer(model, config, device=device)
    
    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Save model
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "patch_cnn_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save to checkpoints directory
        checkpoint_dir = project_root / "models" / "checkpoints" / "patch_cnn"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "patch_cnn_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model also saved to: {checkpoint_path}")
    
    return model


def run_patch_cnn_experiment(
    data_dir: str = "data/raw/FVC2004/DB1_B",
    output_dir: str = "results/patch_cnn",
    train: bool = False,
    model_path: str = None,
    num_epochs: int = 50,
    device: str = "cpu"
):
    """Run the patch CNN experiment."""
    logger = get_logger("PatchCNN")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = FingerprintDataset(data_dir)
    logger.info(f"Loaded {len(dataset)} samples from {len(dataset.subject_ids)} subjects")
    
    # Configuration
    config = PatchCNNConfig(
        patch_size=64,
        num_patches_per_image=20,
        embedding_dim=64,
        aggregation="attention",
        learning_rate=0.001,
        batch_size=16,
        num_epochs=num_epochs
    )
    
    if train:
        # Prepare training data
        logger.info("Preparing training data...")
        train_pairs, val_pairs = prepare_training_data(dataset, validation_split=0.2)
        
        # Train model
        model = train_patch_cnn_model(
            train_pairs,
            val_pairs,
            config,
            num_epochs=num_epochs,
            device=device,
            output_dir=output_path
        )
        
        logger.info("Training complete!")
        
        # Use trained model for evaluation
        model_path = str(output_path / "patch_cnn_model.pth")
    
    # Evaluation
    if model_path and Path(model_path).exists():
        matcher = PatchCNNMatcher(config=config, model_path=model_path, device=device)
        genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
        
        # Evaluate
        logger.info("Running evaluation...")
        evaluator = VerificationEvaluator(matcher, verbose=True)
        results = evaluator.evaluate(genuine_pairs, impostor_pairs)
        
        # Generate report
        evaluator.generate_report(results, str(output_path))
        
        # Print results
        print("\n" + "="*60)
        print("PATCH CNN EXPERIMENT RESULTS")
        print("="*60)
        print(results)
        print("="*60)
    
    elif not train:
        logger.info("Skipping training (use --train to train a new model)")
        logger.info("Skipping evaluation (no model path provided)")
    
    logger.info(f"\nAll results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Patch CNN Experiment"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/FVC2004/DB1_B",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/patch_cnn",
        help="Output directory for results"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Computation device"
    )
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for Patch CNN experiments")
        print("Install with: pip install torch torchvision")
        sys.exit(1)
    
    run_patch_cnn_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train=args.train,
        model_path=args.model_path,
        num_epochs=args.epochs,
        device=args.device
    )


if __name__ == "__main__":
    main()
