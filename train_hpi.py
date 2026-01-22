"""
Training script - HPI Model (Hierarchical Prior Injection Net)

This is the renamed entry point for the model previously tracked under Strategy3/HPF-Net.

Usage:
    python train_hpi.py --config config/hpi_config.yaml
    python train_hpi.py --config config/hpi_config.yaml --resume results/hpi_models/latest_checkpoint.pth

Backward-compatibility:
    You can still resume from older checkpoints by passing their path to --resume.
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.hierarchical_prior_injection_net import (
    HierarchicalPriorInjectionNet,
    create_hierarchical_prior_injection_model
)
from src.data.hierarchical_prior_dataset import (
    HierarchicalPriorDataset,
    hierarchical_prior_collate_fn
)
from src.training.hierarchical_prior_trainer import (
    HierarchicalPriorTrainer,
    create_hierarchical_prior_trainer
)
from src.losses.angular_error_loss import AngularErrorLoss


def setup_logging(config: dict):
    log_config = config.get('logging', {})
    log_dir = Path(config.get('validation', {}).get('log_dir', 'results/hpi_logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'training.log'
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 80)
    logging.info("HPI Model - Training")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    data_config = config.get('data', {})

    logging.info("Creating datasets...")

    train_dataset = HierarchicalPriorDataset(
        data_dir=data_config['train_dir'],
        csf_path=data_config['csf_path'],
        mode='train',
        train_split_ratio=data_config.get('train_split_ratio', 0.8),
        random_seed=data_config.get('random_seed', 42),
        target_height=data_config.get('target_height', 132),
        normalize_input=data_config.get('normalize_input', True),
        use_augmentation=data_config.get('augmentation', {}).get('enable', False),
        cache_priors=data_config.get('cache_priors', True)
    )

    val_dataset = HierarchicalPriorDataset(
        data_dir=data_config['train_dir'],
        csf_path=data_config['csf_path'],
        mode='val',
        train_split_ratio=data_config.get('train_split_ratio', 0.8),
        random_seed=data_config.get('random_seed', 42),
        target_height=data_config.get('target_height', 132),
        normalize_input=data_config.get('normalize_input', True),
        use_augmentation=False,
        cache_priors=data_config.get('cache_priors', True)
    )

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: dict):
    training_config = config.get('training', {})
    dataloader_config = config.get('dataloader', {})

    batch_size = training_config.get('batch_size', 8)
    num_workers = dataloader_config.get('num_workers', 0)
    pin_memory = dataloader_config.get('pin_memory', True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=training_config.get('drop_last', True),
        collate_fn=hierarchical_prior_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=hierarchical_prior_collate_fn
    )

    logging.info(f"Train batches: {len(train_loader)}")
    logging.info(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def create_model(config: dict, device: str):
    logging.info("Creating HPI model (Hierarchical Prior Injection Network)...")

    model = create_hierarchical_prior_injection_model(config)
    model = model.to(device)

    model_info = model.get_model_info()
    logging.info("Model Information:")
    for key, value in model_info.items():
        if 'overhead' in key:
            logging.info(f"  {key}: {value:.2f}%")
        else:
            logging.info(f"  {key}: {value:,}")

    total_params = model_info['total_parameters']
    model_size_mb = total_params * 4 / (1024 * 1024)
    logging.info(f"  Model size: {model_size_mb:.2f} MB")

    return model


def create_loss_function(config: dict, train_dataset, device: str):
    loss_config = config.get('loss', {})
    task_loss_config = loss_config.get('task_loss', {})

    use_csf = task_loss_config.get('use_csf', True)

    if use_csf:
        logging.info("Creating loss function: AngularErrorLoss (with CSF transformation to RGB space)")
        csf_matrix = train_dataset.get_csf()
        loss_fn = AngularErrorLoss(
            epsilon=task_loss_config.get('epsilon', 1e-6),
            use_csf=True,
            csf_matrix=csf_matrix
        )
    else:
        logging.info("Creating loss function: AngularErrorLoss (direct 31-D spectral space)")
        loss_fn = AngularErrorLoss(
            epsilon=task_loss_config.get('epsilon', 1e-6),
            use_csf=False
        )

    loss_fn = loss_fn.to(device)
    return loss_fn


def save_config(config: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = save_dir / 'config.yaml'

    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logging.info(f"Configuration saved to: {config_save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train HPI model (Hierarchical Prior Injection Net)')
    parser.add_argument('--config', type=str, default='config/hpi_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--resume_weights_only', action='store_true',
                        help='Only load model weights, not optimizer state')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), overrides config')

    args = parser.parse_args()

    config = load_config(args.config)

    setup_logging(config)

    device_config = config.get('device', {})
    if args.device is not None:
        device = args.device
    elif device_config.get('use_cuda', True) and torch.cuda.is_available():
        device = f"cuda:{device_config.get('cuda_device', 0)}"
    else:
        device = 'cpu'

    logging.info(f"Using device: {device}")

    random_seed = config.get('data', {}).get('random_seed', 42)
    torch.manual_seed(random_seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(random_seed)
        if device_config.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True

    logging.info(f"Random seed: {random_seed}")

    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    model = create_model(config, device)

    loss_function = create_loss_function(config, train_dataset, device)

    trainer = create_hierarchical_prior_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        config=config,
        device=device
    )

    save_dir = Path(config.get('validation', {}).get('model_save_path', 'results/hpi_models'))
    save_config(config, save_dir)

    logging.info("\n" + "=" * 80)
    logging.info("Starting Training")
    logging.info("=" * 80)

    training_start_time = datetime.now()

    try:
        training_results = trainer.train(
            resume_from=args.resume,
            resume_weights_only=args.resume_weights_only
        )

        training_end_time = datetime.now()
        training_duration = training_end_time - training_start_time

        logging.info("\n" + "=" * 80)
        logging.info("Training Completed")
        logging.info("=" * 80)
        logging.info(f"Total training time: {training_duration}")
        logging.info(f"Best validation loss: {training_results['best_val_loss']:.6f}")
        logging.info(f"Best validation angular error: {training_results['best_val_angular_error']:.4f}°")
        logging.info(f"Best epoch: {training_results['best_epoch'] + 1}")
        logging.info(f"Total epochs trained: {training_results['total_epochs']}")
        logging.info(f"Early stopped: {training_results['early_stopped']}")

        logging.info(f"Next step: Evaluate on test set using best model")
        logging.info(f"Best model path: {save_dir / 'best_model.pth'}")

        return 0

    except KeyboardInterrupt:
        logging.info("\n" + "=" * 80)
        logging.info("Training interrupted by user")
        logging.info("=" * 80)

        logging.info("Saving current state...")
        current_epoch = trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0
        trainer.save_checkpoint(epoch=current_epoch, is_best=False)
        logging.info(f"Saved checkpoint at epoch {current_epoch}")
        logging.info("Current state saved. You can resume training with --resume flag.")

        return 1

    except Exception as e:
        logging.error("\n" + "=" * 80)
        logging.error("Training failed with exception")
        logging.error("=" * 80)
        logging.error(f"Error: {str(e)}", exc_info=True)

        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

