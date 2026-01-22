# HPI-Net: RGB Color Constancy from Multispectral Images via Hierarchical Physics-Injection

This repository contains the implementation for the paper **"RGB Color Constancy from Multispectral Images via Hierarchical Physics-Injection"**. We propose an end-to-end deep learning framework, **HPI-Net**, to recover/estimate scene illumination from **N-channel multispectral images**. The result is then projected into the RGB space via the camera spectral sensitivity function (CSF) for evaluation.

This project features two core contributions:

-   **Physics-based Spectral Augmentation**: Augments the training data 5-fold based on inverse rendering/re-rendering principles and physically measured illuminant spectrums.
-   **Hierarchical Physics-Injection**: Injects classical statistical priors as feature vectors into the network. Specifically, it injects the local texture prior `P_GE2` (2nd-order Gray-Edge) into shallow layers and the global color cast prior `P_GW` (Gray-World) into deep layers, combined with a CSF layer to regress the final RGB illumination.

## 1. Environment

-   Python: 3.8+ (3.9/3.10 recommended)
-   PyTorch: 2.0+ (CUDA acceleration supported)

### Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation

This project is trained and tested on the **NUS Multispectral Dataset**.
Download link: https://yorkucvil.github.io/projects/public_html/spectral_reconstruction/

## 3. Quick Start

### 3.1 Train

```bash
python train_hpi.py --config config/hpi_config.yaml
```

### 3.2 Test / Evaluation

```bash
python test_hpi.py --config config/hpi_config.yaml --checkpoint results/hpi_models/best_model.pth
```

---

## 4. Configuration

The core configuration is located in `config/hpi_config.yaml`.

## 5. Citation

If this work is helpful for your research, please consider citing our paper.

---

## 6. License

This project is released under the MIT License.
