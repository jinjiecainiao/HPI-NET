"""
ResNet-WP训练器模块
实现完整的训练流程，包括训练、验证、早停和模型保存
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import json
from pathlib import Path
import time

from ..models.resnet1d import ResNet1D
from ..data.white_point import SpectralWhitePointExtractor
from .loss import RecoveryAngularErrorLoss, create_loss_function


class ResNetWPTrainer:
    """
    ResNet-WP训练器
    
    负责模型训练、验证、早停、模型保存和日志记录
    """
    
    def __init__(self,
                 model: ResNet1D,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_function: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cpu',
                 config: Dict = None):
        """
        初始化训练器
        
        Args:
            model: ResNet1D模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            loss_function: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            config: 训练配置
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # 训练配置
        self.num_epochs = self.config.get('training', {}).get('num_epochs', 500)
        self.early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        self.evaluate_every = self.config.get('validation', {}).get('evaluate_every', 10)
        
        # 白点特征提取器（不归一化，保持物理意义）
        self.wp_extractor = SpectralWhitePointExtractor(normalize=False)
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # NaN保护相关属性
        self.last_valid_state = None
        self.nan_recovery_count = 0
        self.max_nan_recoveries = 3
        self.enable_nan_protection = self.config.get('training', {}).get('enable_nan_protection', True)
        
        # 自适应学习率调整相关属性
        self.enable_adaptive_lr = self.config.get('training', {}).get('enable_adaptive_lr', False)
        self.grad_norm_threshold = float(self.config.get('training', {}).get('grad_norm_threshold', 5.0))
        self.grad_norm_history = []
        self.large_grad_count = 0
        self.lr_reduction_factor = float(self.config.get('training', {}).get('lr_reduction_factor', 0.8))
        self.min_lr = float(self.config.get('training', {}).get('min_lr', 1e-6))
        self.performance_window = 5  # 监控最近5轮的性能
        self.val_loss_history = []
        
        # 紧急恢复相关属性
        self.consecutive_nan_epochs = 0
        self.max_consecutive_nan = 3  # 连续3个NaN epoch后触发紧急恢复
        
        # 日志记录
        self.setup_logging()
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # TensorBoard
        log_dir = self.config.get('validation', {}).get('log_dir', 'results/logs')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        logging.info(f"Trainer initialized with device: {device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def setup_logging(self):
        """设置日志记录"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        
        # 创建日志目录
        log_file = log_config.get('log_file', 'results/logs/training.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def extract_white_point_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        从批次数据中获取白点特征
        
        Args:
            batch: 包含白点特征的批次
        
        Returns:
            白点特征 [B, 31]
        """
        # 数据已经在collate阶段提取了白点特征，直接返回
        wp_features = batch['multispectral']  # [B, 31] - 已经是白点特征
        
        return wp_features.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # 提取白点特征作为输入
            wp_features = self.extract_white_point_features(batch)  # [B, 31]
            ground_truth = batch['illumination_gt'].to(self.device)  # [B, 31]
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(wp_features)  # [B, 31]
            
            # 计算损失
            loss_result = self.loss_function(predictions, ground_truth)
            
            # 检查损失函数返回类型
            if isinstance(loss_result, dict):
                # 组合损失函数返回字典
                loss = loss_result['total_loss']
            else:
                # 单一损失函数返回标量张量
                loss = loss_result
            
            # 添加L2正则化
            l2_loss = self.model.get_l2_regularization_loss()
            total_loss_with_reg = loss + l2_loss
            
            # NaN检测和保护机制
            if self.enable_nan_protection and self._check_nan_in_computation(loss, total_loss_with_reg, predictions):
                logging.warning(f"NaN detected! Attempting recovery... (Count: {self.nan_recovery_count + 1})")
                
                if self._recover_from_nan():
                    continue  # 跳过这个batch，继续下一个
                else:
                    logging.error("NaN recovery failed. Stopping training.")
                    break
            
            # 保存当前有效状态（在反向传播前）
            if self.enable_nan_protection:
                self._save_valid_state()
            
            # 反向传播
            total_loss_with_reg.backward()
            
            # 检查梯度中的NaN
            if self.enable_nan_protection and self._check_nan_in_gradients():
                logging.warning("NaN detected in gradients! Skipping optimization step.")
                continue
            
            # 梯度裁剪（防止梯度爆炸）
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 极简NaN保护：仅在出现真正的数值错误时干预
            if self.enable_nan_protection and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                logging.warning(f"NaN/Inf gradient detected: {grad_norm:.4f}. Attempting recovery...")
                if self._recover_from_nan():
                    continue
                else:
                    logging.error("Failed to recover from NaN. Stopping training.")
                    break
            
            # 优化器步骤
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'L2': f'{l2_loss.item():.6f}'
            })
        
        # 防止除零错误：如果所有批次都被跳过
        if num_batches == 0:
            logging.warning("All batches were skipped due to large gradients. Triggering emergency recovery.")
            self._emergency_recovery()
            return {'train_loss': float('nan')}
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # 提取白点特征
                wp_features = self.extract_white_point_features(batch)
                ground_truth = batch['illumination_gt'].to(self.device)
                
                # 前向传播
                predictions = self.model(wp_features)
                
                # 计算损失
                loss_result = self.loss_function(predictions, ground_truth)
                
                # 检查损失函数返回类型
                if isinstance(loss_result, dict):
                    # 组合损失函数返回字典
                    loss = loss_result['total_loss']
                else:
                    # 单一损失函数返回标量张量
                    loss = loss_result
                
                total_loss += loss.item()
                
                # 收集预测结果
                all_predictions.append(predictions.cpu())
                all_ground_truth.append(ground_truth.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 计算详细指标
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        
        # 使用损失函数计算详细指标
        if hasattr(self.loss_function, 'compute_metrics'):
            detailed_metrics = self.loss_function.compute_metrics(
                all_predictions.to(self.device), 
                all_ground_truth.to(self.device)
            )
        else:
            detailed_metrics = {}
        
        metrics = {'val_loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        model_save_path = self.config.get('validation', {}).get('model_save_path', 'results/models')
        os.makedirs(model_save_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 添加最佳模型状态到检查点
        if self.best_model_state is not None:
            checkpoint['best_model_state'] = self.best_model_state
        
        # 保存最新检查点
        latest_path = os.path.join(model_save_path, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(model_save_path, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f"Best model saved at epoch {epoch} with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        
        Returns:
            开始epoch
        """
        if not os.path.exists(checkpoint_path):
            logging.warning(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', {
            'train_loss': [], 'val_loss': [], 'learning_rate': [], 'epoch_time': []
        })
        
        # 恢复最佳模型状态（关键修复！）
        self.best_model_state = checkpoint.get('best_model_state', None)
        if self.best_model_state is not None:
            logging.info(f"Best model state restored (val_loss: {self.best_val_loss:.4f})")
        else:
            logging.warning("No best model state found in checkpoint!")
        
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        
        return start_epoch
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, List]:
        """
        执行完整训练流程
        
        Args:
            resume_from_checkpoint: 恢复训练的检查点路径
        
        Returns:
            训练历史
        """
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
        
        logging.info(f"Starting training from epoch {start_epoch} to {self.num_epochs}")
        
        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 验证（每隔几个epoch或最后几个epoch）
            if epoch % self.evaluate_every == 0 or epoch >= self.num_epochs - 10:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {'val_loss': float('nan')}
            
            epoch_time = time.time() - epoch_start_time
            
            # 检查连续NaN epochs
            if np.isnan(train_metrics['train_loss']):
                self.consecutive_nan_epochs += 1
                if self.consecutive_nan_epochs >= self.max_consecutive_nan:
                    logging.error(f"Detected {self.consecutive_nan_epochs} consecutive NaN epochs. Triggering emergency recovery.")
                    self._emergency_recovery()
                    self.consecutive_nan_epochs = 0
            else:
                self.consecutive_nan_epochs = 0
            
            # 记录历史
            self.train_history['train_loss'].append(train_metrics['train_loss'])
            self.train_history['val_loss'].append(val_metrics['val_loss'])
            self.train_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.train_history['epoch_time'].append(epoch_time)
            
            # TensorBoard日志
            self.writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
            if not np.isnan(val_metrics['val_loss']):
                self.writer.add_scalar('Loss/Validation', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 记录详细验证指标
            for key, value in val_metrics.items():
                if key != 'val_loss' and not np.isnan(value):
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            # 早停检查和自适应学习率调整
            is_best = False
            if not np.isnan(val_metrics['val_loss']):
                # 记录验证损失历史
                if self.enable_adaptive_lr:
                    self.val_loss_history.append(val_metrics['val_loss'])
                    if len(self.val_loss_history) > self.performance_window:
                        self.val_loss_history.pop(0)
                    
                    # 检查性能趋势并调整学习率
                    self._check_performance_trend(epoch)
                
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    is_best = True
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.patience_counter += 1
            
            # 保存检查点
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # 日志输出
            log_msg = f"Epoch {epoch:3d}/{self.num_epochs}: "
            log_msg += f"Train Loss: {train_metrics['train_loss']:.4f}, "
            if not np.isnan(val_metrics['val_loss']):
                log_msg += f"Val Loss: {val_metrics['val_loss']:.4f}, "
                if 'mean_error' in val_metrics:
                    log_msg += f"Mean Error: {val_metrics['mean_error']:.2f}°, "
            log_msg += f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
            log_msg += f"Time: {epoch_time:.1f}s"
            
            # 添加动态阈值信息（每10轮显示一次）
            if self.enable_adaptive_lr and epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                dynamic_threshold = max(10.0 * (current_lr / 0.001) ** 0.5, 3.0)
                log_msg += f" [Grad threshold: {dynamic_threshold:.2f}]"
            
            if is_best:
                log_msg += " [BEST]"
            
            logging.info(log_msg)
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # 训练结束后加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logging.info("Loaded best model state")
        
        # 保存最终结果
        self.save_final_results()
        
        # 关闭TensorBoard
        self.writer.close()
        
        return self.train_history
    
    def save_final_results(self):
        """保存最终训练结果"""
        results_dir = self.config.get('validation', {}).get('model_save_path', 'results/models')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存训练历史
        history_path = os.path.join(results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # 将numpy类型转换为Python原生类型
            history_serializable = {}
            for key, values in self.train_history.items():
                history_serializable[key] = [float(v) if not np.isnan(v) else None for v in values]
            json.dump(history_serializable, f, indent=2)
        
        # 保存配置
        config_path = os.path.join(results_dir, 'final_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # 保存模型信息
        model_info = self.model.get_model_info()
        model_info['best_val_loss'] = float(self.best_val_loss)
        model_info['total_epochs'] = len(self.train_history['train_loss'])
        
        info_path = os.path.join(results_dir, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logging.info(f"Final results saved to {results_dir}")
    
    def _check_nan_in_computation(self, loss, total_loss, predictions):
        """检查计算过程中是否出现NaN"""
        return (torch.isnan(loss).any() or 
                torch.isnan(total_loss).any() or 
                torch.isnan(predictions).any() or
                torch.isinf(loss).any() or
                torch.isinf(total_loss).any() or
                torch.isinf(predictions).any())
    
    def _check_nan_in_gradients(self):
        """检查梯度中是否有NaN或inf"""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False
    
    def _save_valid_state(self):
        """保存当前有效的模型状态"""
        self.last_valid_state = {
            'model_state': {k: v.clone() for k, v in self.model.state_dict().items()},
            'optimizer_state': {k: v.clone() if isinstance(v, torch.Tensor) else v 
                              for k, v in self.optimizer.state_dict().items()}
        }
    
    def _recover_from_nan(self):
        """从NaN状态恢复"""
        if self.nan_recovery_count >= self.max_nan_recoveries:
            logging.error(f"Maximum NaN recovery attempts ({self.max_nan_recoveries}) reached.")
            return False
        
        if self.last_valid_state is None:
            logging.error("No valid state saved for recovery.")
            return False
        
        try:
            # 恢复模型状态
            self.model.load_state_dict(self.last_valid_state['model_state'])
            
            # 重建优化器状态
            self.optimizer.load_state_dict(self.last_valid_state['optimizer_state'])
            
            # 降低学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
                logging.info(f"Reduced learning rate to: {param_group['lr']:.2e}")
            
            self.nan_recovery_count += 1
            logging.info(f"Successfully recovered from NaN (attempt {self.nan_recovery_count})")
            return True
            
        except Exception as e:
            logging.error(f"Failed to recover from NaN: {e}")
            return False
    
    def _emergency_recovery(self):
        """紧急恢复机制：当训练完全失控时"""
        logging.error("=" * 60)
        logging.error("EMERGENCY RECOVERY TRIGGERED")
        logging.error("=" * 60)
        
        # 如果有最佳模型状态，回退到最佳状态
        if self.best_model_state is not None:
            logging.info("Restoring to best model state...")
            self.model.load_state_dict(self.best_model_state)
            
            # 重置学习率到一个安全值
            safe_lr = min(0.0001, self.optimizer.param_groups[0]['lr'] * 5.0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = safe_lr
            
            logging.info(f"Learning rate reset to safe value: {safe_lr:.2e}")
            
            # 重置优化器状态
            self.optimizer.state = {}
            logging.info("Optimizer state reset")
            
            # 重置自适应相关计数
            self.large_grad_count = 0
            self.nan_recovery_count = 0
            
            logging.info("Emergency recovery completed. Training will continue.")
        else:
            logging.error("No best model state available for recovery!")
    
    def _adaptive_learning_rate_adjustment(self, reason: str):
        """自适应学习率调整"""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
        
        if new_lr < current_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logging.info(f"[ADAPTIVE] LR adjustment triggered by {reason}")
            logging.info(f"[ADAPTIVE] Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")
            
            # 记录调整信息
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Adaptive/LR_Adjustment', new_lr, len(self.train_history['train_loss']))
        else:
            logging.info(f"[ADAPTIVE] Learning rate already at minimum: {self.min_lr:.2e}")
    
    def _check_performance_trend(self, epoch: int):
        """检查性能趋势并决定是否调整学习率"""
        if len(self.val_loss_history) < self.performance_window:
            return
        
        # 计算最近几轮的趋势
        recent_losses = self.val_loss_history[-self.performance_window:]
        
        # 检查是否有恶化趋势
        if len(recent_losses) >= 3:
            last_3 = recent_losses[-3:]
            if all(last_3[i] >= last_3[i-1] for i in range(1, len(last_3))):
                # 连续3轮性能恶化
                logging.warning(f"[ADAPTIVE] Performance degradation detected over last 3 validations")
                self._adaptive_learning_rate_adjustment("performance_degradation")
        
        # 检查梯度范数趋势
        if len(self.grad_norm_history) >= 20:
            recent_grad_norms = self.grad_norm_history[-20:]
            avg_grad_norm = sum(recent_grad_norms) / len(recent_grad_norms)
            
            if avg_grad_norm > self.grad_norm_threshold:
                logging.warning(f"[ADAPTIVE] High average gradient norm: {avg_grad_norm:.4f}")
                self._adaptive_learning_rate_adjustment("high_gradient_norm")
    
    def _log_adaptive_lr_stats(self, epoch: int):
        """记录自适应学习率相关统计信息"""
        if not self.enable_adaptive_lr:
            return
            
        if len(self.grad_norm_history) >= 10:
            recent_grad_norms = self.grad_norm_history[-10:]
            avg_grad_norm = sum(recent_grad_norms) / len(recent_grad_norms)
            max_grad_norm = max(recent_grad_norms)
            
            logging.debug(f"[ADAPTIVE] Gradient stats - Avg: {avg_grad_norm:.4f}, Max: {max_grad_norm:.4f}")
            
            if hasattr(self, 'writer'):
                self.writer.add_scalar('Gradient/Average_Norm', avg_grad_norm, epoch)
                self.writer.add_scalar('Gradient/Max_Norm', max_grad_norm, epoch)


def create_trainer_from_config(config_path: str,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              csf_matrix: torch.Tensor,
                              device: str = 'cpu') -> ResNetWPTrainer:
    """
    根据配置文件创建训练器
    
    Args:
        config_path: 配置文件路径
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        csf_matrix: 相机响应函数矩阵
        device: 计算设备
    
    Returns:
        训练器实例
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    from ..models.resnet1d import create_model
    model = create_model(config)
    
    # 创建损失函数
    loss_function = create_loss_function(config, csf_matrix)
    
    # 创建优化器
    training_config = config.get('training', {})
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 1e-4)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # 创建学习率调度器
    scheduler = None
    scheduler_config = training_config.get('scheduler', {})
    if scheduler_config.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 100),
            gamma=scheduler_config.get('gamma', 0.5)
        )
    elif scheduler_config.get('type') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('num_epochs', 500)
        )
    
    # 创建训练器
    trainer = ResNetWPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    return trainer


if __name__ == "__main__":
    # 测试训练器
    logging.basicConfig(level=logging.INFO)
    
    print("Trainer module test completed successfully!")
