"""
Experiment: CNN Embedding Model Training and Evaluation

This experiment trains and evaluates CNN-based fingerprint
embedding models on the FVC2004 dataset.

Models:
1. Custom CNN with global embedding
2. ResNet backbone with fine-tuning
3. Contrastive vs Triplet loss comparison

Expected Results:
- CNN models can learn discriminative features
- EER typically 2-5% with sufficient training data
- Requires careful training data augmentation
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
from src.evaluation import VerificationEvaluator, compare_matchers
from src.utils.logger import setup_logger

# Check for PyTorch
try:
    import torch
    from torch.utils.data import DataLoader
    from src.models import (
        CNNConfig, 
        CNNEmbeddingMatcher, 
        FingerprintCNN,
        ContrastiveLoss,
        TripletLoss,
        CNNTrainer,
        FingerprintPairDataset,
        FingerprintTripletDataset,
        LossType,
        create_cnn_model
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CNN experiments disabled.")


def load_pairs_from_dataset(
    dataset: FingerprintDataset,
    num_impostor_ratio: float = 1.0
):
    """Load genuine and impostor pairs from dataset."""
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
        Tuple of (train_pairs, val_pairs, train_paths, val_paths)
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
    
    # Impostor pairs (sample same number as genuine)
    all_train_samples = [s for s in dataset.samples if s.subject_id in train_subjects]
    all_val_samples = [s for s in dataset.samples if s.subject_id in val_subjects]
    
    num_train_genuine = len(train_pairs)
    num_val_genuine = len(val_pairs)
    
    for _ in range(num_train_genuine):
        s1, s2 = np.random.choice(len(all_train_samples), 2, replace=False)
        s1, s2 = all_train_samples[s1], all_train_samples[s2]
        if s1.subject_id != s2.subject_id:
            train_pairs.append((str(s1.path), str(s2.path), 1))
    
    for _ in range(num_val_genuine):
        s1, s2 = np.random.choice(len(all_val_samples), 2, replace=False)
        s1, s2 = all_val_samples[s1], all_val_samples[s2]
        if s1.subject_id != s2.subject_id:
            val_pairs.append((str(s1.path), str(s2.path), 1))
    
    np.random.shuffle(train_pairs)
    np.random.shuffle(val_pairs)
    
    return train_pairs, val_pairs


def train_cnn_model(
    train_pairs: list,
    val_pairs: list,
    config: CNNConfig,
    output_dir: str,
    num_epochs: int = 50,
    device: str = "cpu"
):
    """
    Train CNN embedding model.
    
    Args:
        train_pairs: Training pair list
        val_pairs: Validation pair list
        config: CNN configuration
        output_dir: Output directory
        num_epochs: Number of training epochs
        device: Computation device
        
    Returns:
        Trained model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for training")
    
    logger = setup_logger("cnn_training")
    
    # Create datasets
    train_dataset = FingerprintPairDataset(
        train_pairs,
        input_size=config.input_size
    )
    val_dataset = FingerprintPairDataset(
        val_pairs,
        input_size=config.input_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = FingerprintCNN(config)
    
    # Create trainer
    trainer = CNNTrainer(model, config, device)
    
    # Train
    logger.info(f"Training for {num_epochs} epochs")
    history = trainer.train(train_loader, val_loader, num_epochs)
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    import json
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    
    return model


def run_cnn_experiment(
    data_dir: str,
    output_dir: str,
    train: bool = False,
    model_path: str = None,
    num_epochs: int = 50,
    device: str = "cpu"
):
    """
    Run CNN embedding experiment.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
        train: Whether to train a new model
        model_path: Path to pretrained model
        num_epochs: Number of training epochs
        device: Computation device
    """
    if not TORCH_AVAILABLE:
        print("Error: PyTorch required for CNN experiments")
        return None
    
    logger = setup_logger("cnn_experiment")
    logger.info("Starting CNN embedding experiment")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    dataset = FingerprintDataset(data_dir, name="FVC2004")
    dataset.scan()
    
    logger.info(f"Found {len(dataset)} samples")
    
    # Configuration
    config = CNNConfig(
        input_size=(256, 256),
        embedding_dim=128,
        dropout=0.5,
        loss_type=LossType.CONTRASTIVE,
        contrastive_margin=1.0,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=num_epochs
    )
    
    # Train if requested
    if train:
        logger.info("Preparing training data")
        train_pairs, val_pairs = prepare_training_data(dataset)
        logger.info(f"Training pairs: {len(train_pairs)}, Validation pairs: {len(val_pairs)}")
        
        model = train_cnn_model(
            train_pairs, val_pairs, config,
            str(output_path / "training"),
            num_epochs, device
        )
        model_path = str(output_path / "training" / "cnn_model.pth")
    
    # Create matcher
    matcher = CNNEmbeddingMatcher(
        config=config,
        model_path=model_path,
        device=device
    )
    
    # Generate evaluation pairs
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    # Evaluate
    logger.info("Running evaluation")
    evaluator = VerificationEvaluator(matcher, verbose=True)
    result = evaluator.evaluate(genuine_pairs, impostor_pairs)
    
    # Generate report
    evaluator.generate_report(result, str(output_path))
    
    print("\n" + "=" * 60)
    print("CNN EMBEDDING EXPERIMENT RESULTS")
    print("=" * 60)
    print(result)
    
    return result


def compare_backbones(
    data_dir: str,
    output_dir: str,
    device: str = "cpu"
):
    """
    Compare different CNN backbones.
    
    Args:
        data_dir: Path to dataset directory
        output_dir: Path for output results
        device: Computation device
    """
    if not TORCH_AVAILABLE:
        print("Error: PyTorch required for CNN experiments")
        return None
    
    logger = setup_logger("backbone_comparison")
    logger.info("Comparing CNN backbones")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = FingerprintDataset(data_dir, name="FVC2004")
    dataset.scan()
    
    # Generate pairs
    genuine_pairs, impostor_pairs = load_pairs_from_dataset(dataset)
    
    # Create matchers with different backbones
    matchers = {}
    
    # Custom CNN
    custom_config = CNNConfig(embedding_dim=128)
    matchers["Custom_CNN"] = CNNEmbeddingMatcher(
        config=custom_config, device=device
    )
    
    # Note: ResNet matchers would need pretrained weights in practice
    # For demonstration, we use random initialization
    
    # Compare
    results = compare_matchers(
        matchers,
        genuine_pairs,
        impostor_pairs,
        output_dir=str(output_path),
        verbose=True
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CNN Embedding Experiment"
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
        default="results/cnn_embedding",
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
    parser.add_argument(
        "--compare_backbones",
        action="store_true",
        help="Compare different CNN backbones"
    )
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for CNN experiments")
        print("Install with: pip install torch torchvision")
        sys.exit(1)
    
    if args.compare_backbones:
        compare_backbones(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device
        )
    else:
        run_cnn_experiment(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train=args.train,
            model_path=args.model_path,
            num_epochs=args.epochs,
            device=args.device
        )


if __name__ == "__main__":
    main()
