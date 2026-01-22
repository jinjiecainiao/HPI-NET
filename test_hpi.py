"""
Testing script - HPI Model (Hierarchical Prior Injection Net)

This is the renamed entry point for the model previously tracked under Strategy3/HPF-Net.

Usage:
    python test_hpi.py --config config/hpi_config.yaml --checkpoint results/hpi_models/best_model.pth

Backward-compatibility:
    You can still test older checkpoints by pointing --checkpoint to the old .pth path.
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
import numpy as np
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.hierarchical_prior_injection_net import create_hierarchical_prior_injection_model
from src.data.hierarchical_prior_dataset import HierarchicalPriorDataset, hierarchical_prior_collate_fn
from src.losses.angular_error_loss import AngularErrorLoss


def _resolve_output_dir(config: dict) -> Path:
    """Resolve output directory under HPI-NET by default."""
    project_dir = Path(__file__).parent

    cfg_dir = config.get('validation', {}).get('log_dir', None)
    if cfg_dir is None:
        return project_dir / 'results' / 'hpi_logs'

    cfg_dir_path = Path(cfg_dir)
    if cfg_dir_path.is_absolute():
        return cfg_dir_path

    # Force relative paths to be under HPI-NET
    return project_dir / cfg_dir_path


def setup_logging(config: dict) -> Path:
    log_dir = _resolve_output_dir(config)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'testing.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 80)
    logging.info("HPI Model - Testing on Test Set")
    logging.info("=" * 80)

    return log_dir


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_test_dataset(config: dict):
    data_config = config.get('data', {})

    logging.info("Creating test dataset...")

    test_dataset = HierarchicalPriorDataset(
        data_dir=data_config['test_dir'],
        csf_path=data_config['csf_path'],
        mode='test',
        target_height=data_config.get('target_height', 132),
        normalize_input=data_config.get('normalize_input', True),
        use_augmentation=False,
        cache_priors=data_config.get('cache_priors', True)
    )

    logging.info(f"Test dataset size: {len(test_dataset)}")

    return test_dataset


def create_test_dataloader(test_dataset, config: dict):
    training_config = config.get('training', {})
    dataloader_config = config.get('dataloader', {})

    batch_size = training_config.get('batch_size', 8)
    num_workers = dataloader_config.get('num_workers', 0)
    pin_memory = dataloader_config.get('pin_memory', True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=hierarchical_prior_collate_fn
    )

    logging.info(f"Test batches: {len(test_loader)}")

    return test_loader


def load_model(config: dict, checkpoint_path: str, device: str):
    logging.info(f"Loading model from: {checkpoint_path}")

    model = create_hierarchical_prior_injection_model(config)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
    if missing:
        logging.warning(f"Missing keys when loading state_dict (ignored with strict=False): {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys when loading state_dict (ignored with strict=False): {unexpected}")

    model_info = model.get_model_info()
    logging.info("Model Information:")
    for key, value in model_info.items():
        if 'overhead' in key:
            logging.info(f"  {key}: {value:.2f}%")
        else:
            logging.info(f"  {key}: {value:,}")

    if 'best_val_angular_error' in checkpoint:
        logging.info(f"Best validation angular error: {checkpoint['best_val_angular_error']:.4f}°")
    if 'best_epoch' in checkpoint:
        logging.info(f"Best epoch: {checkpoint['best_epoch'] + 1}")

    model.eval()
    return model


def compute_angular_error(pred: torch.Tensor, target: torch.Tensor, csf_matrix: torch.Tensor = None) -> torch.Tensor:
    if csf_matrix is not None:
        pred = torch.matmul(pred, csf_matrix.t())
        target = torch.matmul(target, csf_matrix.t())

    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)

    cos_sim = torch.sum(pred_norm * target_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    angular_error = torch.acos(cos_sim) * 180.0 / np.pi
    return angular_error


def compute_reproduction_angular_error(pred_illum: torch.Tensor,
                                       target_illum: torch.Tensor,
                                       csf_matrix: torch.Tensor = None,
                                       epsilon: float = 1e-8) -> torch.Tensor:
    if csf_matrix is not None:
        pred_rgb = torch.matmul(pred_illum, csf_matrix.t())
        target_rgb = torch.matmul(target_illum, csf_matrix.t())
    else:
        pred_rgb = pred_illum[:, :3]
        target_rgb = target_illum[:, :3]

    ratio = pred_rgb / (target_rgb + epsilon)

    uniform_white = torch.ones(3, dtype=pred_rgb.dtype, device=pred_rgb.device)
    dot_product = torch.sum(ratio * uniform_white, dim=1)

    ratio_norm = torch.norm(ratio, p=2, dim=1)
    ratio_norm = torch.clamp(ratio_norm, min=epsilon)

    sqrt_3 = np.sqrt(3.0)
    cos_sim = dot_product / (ratio_norm * sqrt_3)

    cos_sim = torch.clamp(cos_sim, -1.0 + epsilon, 1.0 - epsilon)

    angle_rad = torch.acos(cos_sim)
    reproduction_error = angle_rad * 180.0 / np.pi

    return reproduction_error


def evaluate_on_test_set(model, test_loader, test_dataset, device: str):
    logging.info("\n" + "=" * 80)
    logging.info("Evaluating on test set...")
    logging.info("=" * 80)

    model.eval()

    csf_matrix = test_dataset.get_csf().to(device)
    logging.info(f"Using CSF matrix for RGB space evaluation: {csf_matrix.shape}")

    all_recovery_errors = []
    all_reproduction_errors = []
    all_filenames = []

    ge2_impacts = []
    gw_impacts = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            image = batch['multispectral'].to(device)
            target = batch['illumination_gt'].to(device)
            ge2_prior = batch['ge2_prior'].to(device)
            gw_prior = batch['gw_prior'].to(device)
            filenames = batch['filename']

            results = model(
                image,
                ge2_prior=ge2_prior,
                gw_prior=gw_prior
            )

            prediction = results['illumination_pred']
            injection_info = results['injection_info']

            recovery_errors = compute_angular_error(prediction, target, csf_matrix)
            reproduction_errors = compute_reproduction_angular_error(prediction, target, csf_matrix)

            all_recovery_errors.extend(recovery_errors.cpu().numpy())
            all_reproduction_errors.extend(reproduction_errors.cpu().numpy())
            all_filenames.extend(filenames)

            if injection_info.get('ge2_injected', False):
                ge2_impacts.append(injection_info['ge2_impact'])
            if injection_info.get('gw_injected', False):
                gw_impacts.append(injection_info['gw_impact'])

            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    all_recovery_errors = np.array(all_recovery_errors)
    all_reproduction_errors = np.array(all_reproduction_errors)

    recovery_mean = np.mean(all_recovery_errors)
    recovery_median = np.median(all_recovery_errors)
    recovery_std = np.std(all_recovery_errors)
    recovery_min = np.min(all_recovery_errors)
    recovery_max = np.max(all_recovery_errors)
    recovery_p25 = np.percentile(all_recovery_errors, 25)
    recovery_p75 = np.percentile(all_recovery_errors, 75)
    recovery_p95 = np.percentile(all_recovery_errors, 95)

    reproduction_mean = np.mean(all_reproduction_errors)
    reproduction_median = np.median(all_reproduction_errors)
    reproduction_std = np.std(all_reproduction_errors)
    reproduction_min = np.min(all_reproduction_errors)
    reproduction_max = np.max(all_reproduction_errors)
    reproduction_p25 = np.percentile(all_reproduction_errors, 25)
    reproduction_p75 = np.percentile(all_reproduction_errors, 75)
    reproduction_p95 = np.percentile(all_reproduction_errors, 95)

    avg_ge2_impact = np.mean(ge2_impacts) if len(ge2_impacts) > 0 else 0.0
    avg_gw_impact = np.mean(gw_impacts) if len(gw_impacts) > 0 else 0.0

    logging.info("\n" + "=" * 80)
    logging.info("Test Set Results")
    logging.info("=" * 80)
    logging.info(f"Total samples: {len(all_recovery_errors)}")

    logging.info("\nRecovery Angular Error Statistics:")
    logging.info(f"  Mean:     {recovery_mean:.4f}°")
    logging.info(f"  Median:   {recovery_median:.4f}°")
    logging.info(f"  Std Dev:  {recovery_std:.4f}°")
    logging.info(f"  Min:      {recovery_min:.4f}°")
    logging.info(f"  Max:      {recovery_max:.4f}°")
    logging.info(f"  25th:     {recovery_p25:.4f}°")
    logging.info(f"  75th:     {recovery_p75:.4f}°")
    logging.info(f"  95th:     {recovery_p95:.4f}°")

    logging.info("\nReproduction Angular Error Statistics:")
    logging.info(f"  Mean:     {reproduction_mean:.4f}°")
    logging.info(f"  Median:   {reproduction_median:.4f}°")
    logging.info(f"  Std Dev:  {reproduction_std:.4f}°")
    logging.info(f"  Min:      {reproduction_min:.4f}°")
    logging.info(f"  Max:      {reproduction_max:.4f}°")
    logging.info(f"  25th:     {reproduction_p25:.4f}°")
    logging.info(f"  75th:     {reproduction_p75:.4f}°")
    logging.info(f"  95th:     {reproduction_p95:.4f}°")

    logging.info("\nPrior Injection Impact:")
    logging.info(f"  GE2 (shallow) impact: {avg_ge2_impact:.6f}")
    logging.info(f"  GW (deep) impact:     {avg_gw_impact:.6f}")

    results = {
        'recovery_angular_error': {
            'mean': float(recovery_mean),
            'median': float(recovery_median),
            'std': float(recovery_std),
            'min': float(recovery_min),
            'max': float(recovery_max),
            'percentile_25': float(recovery_p25),
            'percentile_75': float(recovery_p75),
            'percentile_95': float(recovery_p95),
        },
        'reproduction_angular_error': {
            'mean': float(reproduction_mean),
            'median': float(reproduction_median),
            'std': float(reproduction_std),
            'min': float(reproduction_min),
            'max': float(reproduction_max),
            'percentile_25': float(reproduction_p25),
            'percentile_75': float(reproduction_p75),
            'percentile_95': float(reproduction_p95),
        },
        'mean_angular_error': float(recovery_mean),
        'median_angular_error': float(recovery_median),
        'std_angular_error': float(recovery_std),
        'min_angular_error': float(recovery_min),
        'max_angular_error': float(recovery_max),
        'percentile_25': float(recovery_p25),
        'percentile_75': float(recovery_p75),
        'percentile_95': float(recovery_p95),
        'ge2_injection_impact': float(avg_ge2_impact),
        'gw_injection_impact': float(avg_gw_impact),
        'num_samples': len(all_recovery_errors),
        'recovery_errors': all_recovery_errors.tolist(),
        'reproduction_errors': all_reproduction_errors.tolist(),
        'filenames': all_filenames
    }

    return results


def save_results(results: dict, output_dir: Path):
    results_file = output_dir / 'test_results.json'

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Test HPI model (Hierarchical Prior Injection Net)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = setup_logging(config)

    device_config = config.get('device', {})
    if args.device is not None:
        device = args.device
    elif device_config.get('use_cuda', True) and torch.cuda.is_available():
        device = f"cuda:{device_config.get('cuda_device', 0)}"
    else:
        device = 'cpu'

    logging.info(f"Using device: {device}")

    test_dataset = create_test_dataset(config)
    test_loader = create_test_dataloader(test_dataset, config)

    model = load_model(config, args.checkpoint, device)

    results = evaluate_on_test_set(model, test_loader, test_dataset, device)

    save_results(results, output_dir)

    logging.info("\n" + "=" * 80)
    logging.info("Testing completed successfully!")
    logging.info("=" * 80)

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
