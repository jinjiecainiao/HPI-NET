"""
Create a right-side comparison figure (Input vs Ours) using the user's HPI-Net model.

- Loads the trained checkpoint at results/strategy3_v2_hpf_net_models/best_model.pth
- Picks a classic hard-case sample (default: Metal_halide_lamp_2500K_Scene_01)
- Generates the corrected image and a side-by-side figure with clean captions
- Also reports the recovery angular error (AE) for the sample

Usage:
  python SMPI-NET/scripts/create_visual_impact_pair.py \
    --scene Metal_halide_lamp_2500K_Scene_01 \
    --device auto \
    --output SMPI-NET/results/figures/visual_impact_pair.png

"""

import os
# Avoid OpenMP duplicate runtime error on some Windows/conda setups
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import torch
from typing import Tuple, Optional

# Resolve project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent.parent  # .../mutispectral_ResNet
SMPI_ROOT = SCRIPT_DIR.parent         # .../mutispectral_ResNet/SMPI-NET

# Add SMPI-NET/src to import path
import sys
sys.path.insert(0, str(SMPI_ROOT / 'src'))

from models.hierarchical_prior_injection_net import HierarchicalPriorInjectionNet


def load_camera_sensitivity(camera_file: Path) -> np.ndarray:
    data = sio.loadmat(str(camera_file))
    # Try common keys
    if 'CRF' in data:
        crf = data['CRF']
    else:
        # Fallback: pick the first non-meta key
        keys = [k for k in data.keys() if not k.startswith('__')]
        assert len(keys) > 0, f"No valid data found in {camera_file}"
        crf = data[keys[0]]
    # Keep first 31 bands if 33 provided
    if crf.shape[1] == 33:
        crf = crf[:, :31]
    assert crf.shape == (3, 31) or crf.shape == (31, 3)
    if crf.shape == (31, 3):
        crf = crf.T
    return crf.astype(np.float32)  # [3, 31]


def load_mat(mat_path: Path):
    mat = sio.loadmat(str(mat_path))
    # multispectral
    if 'tensor' in mat:
        ms = np.array(mat['tensor'], dtype=np.float32)
    elif 'img' in mat:
        ms = np.array(mat['img'], dtype=np.float32)
    else:
        # Fallback: first non-meta key
        keys = [k for k in mat.keys() if not k.startswith('__')]
        ms = np.array(mat[keys[0]], dtype=np.float32)
    # illumination gt
    illum = None
    for k in ['illumination', 'illum', 'light', 'gt', 'ground_truth']:
        if k in mat:
            illum = np.array(mat[k], dtype=np.float32).squeeze()
            break
    return ms, illum


def resize_ms(ms: np.ndarray, target_h: int = 132) -> np.ndarray:
    from scipy import ndimage
    H, W, C = ms.shape
    if H == target_h:
        return ms
    scale = target_h / float(H)
    new_w = int(round(W * scale))
    out = np.zeros((target_h, new_w, C), dtype=np.float32)
    for c in range(C):
        out[:, :, c] = ndimage.zoom(ms[:, :, c], (scale, scale), order=1)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0, None)
    return out


def compute_priors(ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # GE2 (Grey-Edge order=2, simple finite diff proxy)
    gx = np.abs(np.diff(ms, axis=1)).sum(axis=(0, 1))
    gy = np.abs(np.diff(ms, axis=0)).sum(axis=(0, 1))
    ge2 = gx + gy
    # GW (Grey-World)
    gw = ms.mean(axis=(0, 1))
    # Normalize
    ge2 = ge2 / (np.linalg.norm(ge2) + 1e-8)
    gw = gw / (np.linalg.norm(gw) + 1e-8)
    return ge2.astype(np.float32), gw.astype(np.float32)


def build_model(device: str) -> HierarchicalPriorInjectionNet:
    spectral_extractor_config = {
        'hidden_channels': [64, 128, 256],
        'dropout_rate': 0.4,
        'inject_ge2_after_conv1': True,
        'inject_gw_after_conv3': True,
    }
    resnet_regressor_config = {
        'input_dim': 256,
        'hidden_dims': [128, 256, 128],
        'num_residual_blocks': 3,
        'dropout_rate': 0.4,
        'l2_regularization': 1e-4,
    }
    model = HierarchicalPriorInjectionNet(
        input_channels=31,
        output_dim=31,
        spectral_extractor_config=spectral_extractor_config,
        resnet_regressor_config=resnet_regressor_config,
    )
    return model.to(device)


def load_checkpoint(model: HierarchicalPriorInjectionNet, ckpt_path: Path, device: str) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()


def estimate_illum(model, ms: np.ndarray, ge2: np.ndarray, gw: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        ms_t = torch.from_numpy(ms).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1,31,H,W]
        ge2_t = torch.from_numpy(ge2).unsqueeze(0).float().to(device)                 # [1,31]
        gw_t  = torch.from_numpy(gw).unsqueeze(0).float().to(device)                  # [1,31]
        out = model(ms_t, ge2_prior=ge2_t, gw_prior=gw_t)
        pred = out['illumination_pred'] if isinstance(out, dict) else out
        return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def correct_rgb(png_rgb: np.ndarray, illum_31: np.ndarray, crf_rgb31: np.ndarray) -> np.ndarray:
    rgb_illum = crf_rgb31 @ illum_31  # [3]
    rgb_illum = np.clip(rgb_illum, 1e-8, None)
    out = np.empty_like(png_rgb)
    for c in range(3):
        out[:, :, c] = png_rgb[:, :, c] / rgb_illum[c]
    out = np.clip(out, 0, None)
    m = out.max()
    if m > 1e-8:
        out = out / m
    return out


def recovery_angular_error(pred_31: np.ndarray, gt_31: np.ndarray, crf_rgb31: Optional[np.ndarray] = None) -> float:
    if crf_rgb31 is not None:
        p = (crf_rgb31 @ pred_31).astype(np.float32)
        g = (crf_rgb31 @ gt_31).astype(np.float32)
    else:
        p = pred_31.astype(np.float32)
        g = gt_31.astype(np.float32)
    p = p / (np.linalg.norm(p) + 1e-8)
    g = g / (np.linalg.norm(g) + 1e-8)
    cos = float(np.clip((p * g).sum(), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def make_figure(input_img: np.ndarray, corrected_img: np.ndarray, title_right: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].imshow(input_img)
    axes[0].set_title('Input (Hard Case)', fontsize=16, fontweight='bold', pad=12)
    axes[0].axis('off')

    axes[1].imshow(corrected_img)
    axes[1].set_title(title_right, fontsize=16, fontweight='bold', pad=12)
    axes[1].axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='Metal_halide_lamp_2500K_Scene_01',
                        help='Scene name without camera prefix, e.g., Daylight_Scene_03 or Metal_halide_lamp_2500K_Scene_01')
    parser.add_argument('--device', type=str, default='auto', help='cuda, cpu, or auto')
    parser.add_argument('--output', type=str, default=str(SMPI_ROOT / 'results' / 'figures' / 'visual_impact_pair.png'))
    parser.add_argument('--checkpoint', type=str, default=str(SMPI_ROOT / 'results' / 'strategy3_v2_hpf_net_models' / 'best_model.pth'))
    args = parser.parse_args()

    device = (
        f"cuda:{0}" if (args.device in ['auto', 'cuda'] and torch.cuda.is_available()) else ('cpu' if args.device in ['auto', 'cpu'] else args.device)
    )

    # Paths
    png_path = PROJ_ROOT / 'data' / 'dataset' / 'testing' / 'png' / f'Canon_1D_Mark_III_{args.scene}.png'
    mat_path = PROJ_ROOT / 'data' / 'dataset' / 'testing' / 'mat_norm' / f'{args.scene}.mat'
    camera_file = PROJ_ROOT / 'data' / 'Canon_1D_Mark_III.mat'
    ckpt_path = Path(args.checkpoint)

    assert png_path.exists(), f"PNG not found: {png_path}"
    assert mat_path.exists(), f"MAT not found: {mat_path}"
    assert camera_file.exists(), f"Camera file not found: {camera_file}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Load data
    png_img = np.array(Image.open(png_path)).astype(np.float32) / 255.0
    ms, illum_gt = load_mat(mat_path)
    ms = resize_ms(ms, target_h=132)
    ge2, gw = compute_priors(ms)

    # Load model and estimate illumination
    model = build_model(device)
    load_checkpoint(model, ckpt_path, device)
    illum_pred = estimate_illum(model, ms, ge2, gw, device)

    # Camera sensitivity
    crf = load_camera_sensitivity(camera_file)  # [3,31]

    # Correct
    corrected = correct_rgb(png_img, illum_pred, crf)

    # AE (recovery in RGB)
    ae_text = ''
    if illum_gt is not None and illum_gt.shape[0] == 31:
        ae = recovery_angular_error(illum_pred, illum_gt, crf)
        ae_text = f'Ours (HPI-Net)\nAE: {ae:.2f}°'
    else:
        ae_text = 'Ours (HPI-Net)'

    # Figure
    output_path = Path(args.output)
    make_figure(png_img, corrected, ae_text, output_path)
    print(f"Saved figure to: {output_path}")


if __name__ == '__main__':
    main()

