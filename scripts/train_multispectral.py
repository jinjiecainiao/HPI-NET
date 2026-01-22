#!/usr/bin/env python3
"""
MultispectralResNet训练脚本
直接从多光谱图像进行RGB照度估计，不使用白点预处理
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练支持
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.multispectral_dataset import create_multispectral_dataloaders
from src.models.multispectral_resnet import create_multispectral_resnet, create_pure_spectral_resnet
from src.training.loss import create_loss_function
import scipy.io as sio
import numpy as np


def setup_logging(config: dict):
    """设置日志"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志目录
    log_file = log_config.get('log_file', 'results/multispectral_resnet_logs/training.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_csf_matrix(csf_path: str) -> torch.Tensor:
    """加载相机响应函数矩阵"""
    try:
        csf_data = sio.loadmat(csf_path)
        
        if 'CRF' in csf_data:
            csf_matrix = np.array(csf_data['CRF'], dtype=np.float32)
            if csf_matrix.shape == (3, 33):
                csf_matrix = csf_matrix[:, :31].T
            elif csf_matrix.shape == (3, 31):
                csf_matrix = csf_matrix.T
            elif csf_matrix.shape != (31, 3):
                raise ValueError(f"Unexpected CRF shape: {csf_matrix.shape}")
        else:
            # 尝试其他可能的键名
            possible_keys = ['csf', 'sensitivity', 'camera_sensitivity', 'response']
            csf_matrix = None
            
            for key in possible_keys:
                if key in csf_data:
                    csf_matrix = np.array(csf_data[key], dtype=np.float32)
                    break
            
            if csf_matrix is None:
                for key, value in csf_data.items():
                    if isinstance(value, np.ndarray) and not key.startswith('__'):
                        if value.shape == (31, 3):
                            csf_matrix = value.astype(np.float32)
                            break
                        elif value.shape == (3, 31):
                            csf_matrix = value.T.astype(np.float32)
                            break
                        elif value.shape == (3, 33):
                            csf_matrix = value[:, :31].T.astype(np.float32)
                            break
            
            if csf_matrix is None:
                raise ValueError(f"Could not find valid CSF matrix in {csf_path}")
            
            if csf_matrix.shape == (3, 31):
                csf_matrix = csf_matrix.T
        
        if csf_matrix.shape != (31, 3):
            raise ValueError(f"CSF matrix has invalid shape: {csf_matrix.shape}")
        
        logging.info(f"Loaded CSF matrix with shape {csf_matrix.shape}")
        return torch.from_numpy(csf_matrix)
        
    except Exception as e:
        logging.error(f"Failed to load CSF matrix from {csf_path}: {e}")
        logging.warning("Using default CSF matrix")
        return create_default_csf()


def create_default_csf() -> torch.Tensor:
    """创建默认的相机响应函数"""
    csf = np.zeros((31, 3), dtype=np.float32)
    csf[20:31, 0] = np.linspace(0.1, 1.0, 11)  # R
    csf[10:25, 1] = np.concatenate([np.linspace(0.1, 1.0, 8), np.linspace(1.0, 0.1, 7)])  # G
    csf[0:15, 2] = np.linspace(1.0, 0.1, 15)  # B
    return torch.from_numpy(csf)


def train_multispectral_resnet(config: dict, device: str, resume_from: str = None):
    """训练多光谱ResNet模型"""
    
    # 创建数据加载器
    data_config = config['data']
    train_dir = data_config['train_dir']
    test_dir = data_config['test_dir']
    csf_path = data_config['csf_path']
    
    # 转换为绝对路径
    if not os.path.isabs(train_dir):
        train_dir = project_root / train_dir
    if not os.path.isabs(test_dir):
        test_dir = project_root / test_dir
    if not os.path.isabs(csf_path):
        csf_path = project_root / csf_path
    
    logging.info(f"Loading training data from: {train_dir}")
    logging.info(f"Loading test data from: {test_dir}")
    
    # 获取数据加载器配置
    advanced_config = config.get('advanced', {})
    num_workers = advanced_config.get('num_workers', 4)
    persistent_workers = advanced_config.get('persistent_workers', False)
    prefetch_factor = advanced_config.get('prefetch_factor', 2)
    
    train_loader, val_loader, test_loader = create_multispectral_dataloaders(
        train_dir=str(train_dir),
        test_dir=str(test_dir),
        csf_path=str(csf_path),
        config=config,
        batch_size=config['training']['batch_size'],
        train_split_ratio=data_config['train_split_ratio'],
        num_workers=num_workers,
        random_seed=data_config['random_seed'],
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    logging.info(f"Data loaded successfully:")
    logging.info(f"  Training samples: {len(train_loader.dataset)}")
    logging.info(f"  Validation samples: {len(val_loader.dataset)}")
    logging.info(f"  Test samples: {len(test_loader.dataset)}")
    
    # 加载CSF矩阵
    csf_matrix = load_csf_matrix(str(csf_path))
    
    # 创建模型 - 支持多种模型类型
    model_type = config.get('model', {}).get('type', 'multispectral_resnet')
    
    if model_type == 'pure_spectral_resnet':
        model = create_pure_spectral_resnet(config)
        logging.info(f"Created PureSpectralResNet model (End-to-End, No WP)")
    else:
        model = create_multispectral_resnet(config)
        logging.info(f"Created MultispectralResNet model")
    
    model = model.to(device)
    
    # 显示模型信息
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        logging.info(f"Model Info:")
        for key, value in model_info.items():
            logging.info(f"  {key}: {value}")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 创建损失函数
    loss_function = create_loss_function(config, csf_matrix)
    loss_function = loss_function.to(device)
    
    # 创建优化器
    training_config = config['training']
    optimizer_name = training_config.get('optimizer', 'adamw').lower()
    learning_rate = training_config.get('learning_rate', 1e-4)
    weight_decay = training_config.get('weight_decay', 0.005)
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 混合精度训练支持
    use_mixed_precision = config.get('advanced', {}).get('mixed_precision', False) and 'cuda' in device
    scaler = GradScaler() if use_mixed_precision else None
    
    if use_mixed_precision:
        logging.info("🚀 Mixed Precision Training ENABLED (AMP)")
        logging.info("   Using automatic mixed precision for faster training")
    else:
        logging.info("Mixed Precision Training DISABLED (using FP32)")
    
    # 创建学习率调度器
    scheduler = None
    scheduler_config = training_config.get('scheduler', {})
    if scheduler_config.get('type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('num_epochs', 400),
            eta_min=scheduler_config.get('eta_min', 1e-5)
        )
    elif scheduler_config.get('type') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 100),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    
    # 训练参数
    num_epochs = training_config.get('num_epochs', 400)
    early_stopping_patience = training_config.get('early_stopping_patience', 80)
    grad_clip_norm = training_config.get('grad_clip_norm', 1.0)
    
    # 模型保存路径
    model_save_path = config['validation']['model_save_path']
    os.makedirs(model_save_path, exist_ok=True)
    
    # 断电恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 自动检查断点恢复 (优先级: 手动指定 > latest_checkpoint > best_model)
    checkpoint_path = None
    
    # 检查是否启用自动恢复
    auto_resume = config.get('advanced', {}).get('auto_resume', True)
    
    if resume_from:
        # 手动指定的恢复路径 (优先级最高)
        checkpoint_path = resume_from
        logging.info(f"Using manually specified checkpoint: {checkpoint_path}")
    elif auto_resume:
        # 自动查找检查点 (按优先级)
        auto_checkpoints = [
            os.path.join(model_save_path, 'latest_checkpoint.pth'),  # 最新训练状态
            os.path.join(model_save_path, 'best_model.pth'),        # 最佳模型
        ]
        
        for auto_checkpoint in auto_checkpoints:
            if os.path.exists(auto_checkpoint):
                checkpoint_path = auto_checkpoint
                logging.info(f"Auto-detected checkpoint: {checkpoint_path}")
                break
    else:
        logging.info("Auto-resume disabled in config, starting fresh training")
    
    # 加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复混合精度scaler状态
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logging.info("Loaded GradScaler state for mixed precision training")
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            
            logging.info(f"✅ Successfully resumed training from epoch {start_epoch}")
            logging.info(f"📊 Best validation loss so far: {best_val_loss:.4f}")
            logging.info(f"⏰ Early stopping patience: {patience_counter}/{early_stopping_patience}")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            logging.info("Starting training from scratch...")
    else:
        logging.info("🚀 No checkpoint found, starting fresh training from epoch 0")
    
    logging.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
    
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            ms_data = batch['multispectral'].to(device)  # [B, 31, H, W]
            ground_truth = batch['illumination_gt'].to(device)  # [B, 31]
            
            optimizer.zero_grad()
            
            # 混合精度前向传播和反向传播
            if scaler is not None:
                # 使用混合精度
                with autocast():
                    predictions = model(ms_data)  # [B, 31]
                    loss = loss_function(predictions, ground_truth)
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                # 混合精度梯度裁剪
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准FP32训练
                predictions = model(ms_data)  # [B, 31]
                loss = loss_function(predictions, ground_truth)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                
                optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logging.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                ms_data = batch['multispectral'].to(device)
                ground_truth = batch['illumination_gt'].to(device)
                
                # 验证时也使用混合精度加速
                if scaler is not None:
                    with autocast():
                        predictions = model(ms_data)
                        loss = loss_function(predictions, ground_truth)
                else:
                    predictions = model(ms_data)
                    loss = loss_function(predictions, ground_truth)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        
        # 记录日志
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch:3d}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}")
        
        # 早停和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config
            }
            # 保存混合精度scaler状态
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(model_save_path, 'best_model.pth'))
            
            logging.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # 保存最新检查点 (每个epoch都保存，用于断电恢复)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': config
        }
        # 保存混合精度scaler状态
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint, os.path.join(model_save_path, 'latest_checkpoint.pth'))
        
        # 早停检查
        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # 定期保存命名检查点 (可选，用于备份)
        if epoch % 20 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': config
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, os.path.join(model_save_path, f'checkpoint_epoch_{epoch}.pth'))
    
    logging.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # 保存最终配置
    with open(os.path.join(model_save_path, 'final_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train MultispectralResNet model')
    
    parser.add_argument('--config', type=str, default='config/multispectral_resnet_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置日志
    setup_logging(config)
    
    logging.info("="*60)
    logging.info("MultispectralResNet Training Started")
    logging.info("="*60)
    
    # 设置设备
    if args.device is not None:
        device = args.device
    elif config.get('device', {}).get('use_cuda', True) and torch.cuda.is_available():
        device = f"cuda:{config.get('device', {}).get('cuda_device', 0)}"
    else:
        device = 'cpu'
    
    logging.info(f"Using device: {device}")
    
    try:
        # 开始训练
        best_val_loss = train_multispectral_resnet(config, device, args.resume)
        
        logging.info("Training completed successfully!")
        logging.info(f"Best validation loss: {best_val_loss:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
