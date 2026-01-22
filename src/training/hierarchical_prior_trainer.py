"""
层次化先验注入网络训练器
基于V3训练策略：固定学习率 + 轻量化模型 + 早停
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import json
from datetime import datetime


class HierarchicalPriorTrainer:
    """
    层次化先验注入网络训练器
    
    训练策略（借鉴V3成功经验）:
    1. 固定学习率（0.0001）
    2. 早停策略（patience=50）
    3. 梯度裁剪（norm=1.0）
    4. NaN保护
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_function: nn.Module,
                 config: dict,
                 device: str = 'cuda'):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            loss_function: 损失函数
            config: 配置字典
            device: 设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.config = config
        self.device = device
        
        # 训练参数
        training_config = config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 400)
        self.early_stopping_patience = training_config.get('early_stopping_patience', 50)
        self.gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
        self.enable_nan_protection = training_config.get('enable_nan_protection', True)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器（V3策略：不使用）
        self.scheduler = None
        if training_config.get('use_scheduler', False):
            self.scheduler = self._create_scheduler()
        
        # 保存路径
        validation_config = config.get('validation', {})
        self.model_save_path = Path(validation_config.get('model_save_path', 'results/hierarchical_prior_injection_models'))
        self.log_dir = Path(validation_config.get('log_dir', 'results/hierarchical_prior_injection_logs'))
        
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_angular_error = float('inf')
        self.best_epoch = 0
        self.resume_epoch = 0  # 用于断点恢复
        self.current_epoch = 0  # 当前正在训练的epoch
        self.epochs_without_improvement = 0
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_angular_error': [],
            'learning_rate': [],
            'ge2_injection_impact': [],
            'gw_injection_impact': []
        }
        
        logging.info("HierarchicalPriorTrainer initialized with V3-enhanced configuration:")
        logging.info(f"  - Epochs: {self.num_epochs}")
        logging.info(f"  - Early stopping patience: {self.early_stopping_patience}")
        logging.info(f"  - Gradient clip norm: {self.gradient_clip_norm}")
        logging.info(f"  - Use scheduler: {self.scheduler is not None}")
        logging.info(f"  - Model save path: {self.model_save_path}")
        logging.info(f"  - Log directory: {self.log_dir}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        training_config = self.config.get('training', {})
        
        lr = float(training_config.get('learning_rate', 0.0001))
        weight_decay = float(training_config.get('weight_decay', 0.0001))
        adam_eps = float(training_config.get('adam_eps', 1e-8))
        
        # 处理adam_betas（可能是列表）
        adam_betas = training_config.get('adam_betas', [0.9, 0.999])
        if isinstance(adam_betas, list):
            adam_betas = [float(b) for b in adam_betas]
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
            amsgrad=training_config.get('amsgrad', False)
        )
        
        logging.info(f"Optimizer: Adam(lr={lr}, weight_decay={weight_decay}, eps={adam_eps})")
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        training_config = self.config.get('training', {})
        scheduler_type = training_config.get('scheduler_type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(training_config.get('scheduler_factor', 0.5)),
                patience=int(training_config.get('scheduler_patience', 20)),
                min_lr=float(training_config.get('min_lr', 1e-6))
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logging.info(f"Scheduler: {scheduler_type}")
        return scheduler
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            image = batch['multispectral'].to(self.device)
            target = batch['illumination_gt'].to(self.device)
            ge2_prior = batch['ge2_prior'].to(self.device)
            gw_prior = batch['gw_prior'].to(self.device)
            
            # NaN检测
            if self.enable_nan_protection:
                if torch.isnan(image).any() or torch.isnan(target).any():
                    logging.warning(f"NaN detected in input data at batch {batch_idx}, skipping...")
                    continue
            
            # 前向传播
            self.optimizer.zero_grad()
            
            results = self.model(
                image,
                ge2_prior=ge2_prior,
                gw_prior=gw_prior
            )
            
            prediction = results['illumination_pred']
            
            # 计算损失
            loss_dict = self.loss_function(prediction, target)
            loss = loss_dict['total_loss']
            
            # 添加L2正则化
            l2_loss = self.model.get_l2_regularization_loss()
            loss = loss + l2_loss
            
            # NaN检测
            if self.enable_nan_protection and torch.isnan(loss):
                logging.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
            
            # 优化器步进
            self.optimizer.step()
            
            # 累积损失
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        return avg_epoch_loss
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        val_loss = 0.0
        val_angular_error = 0.0
        num_batches = 0
        
        # 注入影响统计
        ge2_impacts = []
        gw_impacts = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 获取数据
                image = batch['multispectral'].to(self.device)
                target = batch['illumination_gt'].to(self.device)
                ge2_prior = batch['ge2_prior'].to(self.device)
                gw_prior = batch['gw_prior'].to(self.device)
                
                # 前向传播
                results = self.model(
                    image,
                    ge2_prior=ge2_prior,
                    gw_prior=gw_prior
                )
                
                prediction = results['illumination_pred']
                injection_info = results['injection_info']
                
                # 计算损失
                loss_dict = self.loss_function(prediction, target)
                
                # 累积统计
                val_loss += loss_dict['total_loss'].item()
                val_angular_error += loss_dict['angular_error'].item()
                
                # 记录注入影响
                if injection_info.get('ge2_injected', False):
                    ge2_impacts.append(injection_info['ge2_impact'])
                if injection_info.get('gw_injected', False):
                    gw_impacts.append(injection_info['gw_impact'])
                
                num_batches += 1
        
        # 计算平均值
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        avg_angular_error = val_angular_error / num_batches if num_batches > 0 else float('inf')
        
        avg_ge2_impact = np.mean(ge2_impacts) if len(ge2_impacts) > 0 else 0.0
        avg_gw_impact = np.mean(gw_impacts) if len(gw_impacts) > 0 else 0.0
        
        return {
            'val_loss': avg_val_loss,
            'val_angular_error': avg_angular_error,
            'ge2_injection_impact': avg_ge2_impact,
            'gw_injection_impact': avg_gw_impact
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_angular_error': self.best_val_angular_error,
            'best_epoch': self.best_epoch,
            'train_history': self.train_history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = self.model_save_path / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_model_path = self.model_save_path / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            logging.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str, weights_only: bool = False):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if not weights_only:
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 加载训练状态
            self.resume_epoch = checkpoint.get('epoch', 0)  # 保存检查点的epoch
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_angular_error = checkpoint.get('best_val_angular_error', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            self.train_history = checkpoint.get('train_history', {})
            
            logging.info(f"Resumed from epoch {self.resume_epoch}")
            logging.info(f"Best val loss: {self.best_val_loss:.6f}")
            logging.info(f"Best angular error: {self.best_val_angular_error:.4f}°")
        else:
            logging.info("Loaded model weights only")
            self.resume_epoch = 0
    
    def train(self, resume_from: Optional[str] = None, resume_weights_only: bool = False) -> Dict:
        """
        主训练循环
        
        Args:
            resume_from: 恢复训练的检查点路径
            resume_weights_only: 仅加载权重，不加载优化器状态
        
        Returns:
            training_results: 训练结果字典
        """
        # 恢复训练
        start_epoch = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from, weights_only=resume_weights_only)
            if not resume_weights_only:
                # 从检查点的下一个epoch继续
                start_epoch = self.resume_epoch + 1
        
        logging.info("Starting hierarchical prior injection training...")
        logging.info(f"Training from epoch {start_epoch} to {self.num_epochs}")
        
        training_start_time = datetime.now()
        
        # 训练循环
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch  # 更新当前epoch，用于异常处理时保存
            epoch_start_time = datetime.now()
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_metrics['val_loss'])
            self.train_history['val_angular_error'].append(val_metrics['val_angular_error'])
            self.train_history['learning_rate'].append(current_lr)
            self.train_history['ge2_injection_impact'].append(val_metrics['ge2_injection_impact'])
            self.train_history['gw_injection_impact'].append(val_metrics['gw_injection_impact'])
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['val_loss'])
            
            # 检查是否为最佳模型
            is_best = val_metrics['val_loss'] < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_val_angular_error = val_metrics['val_angular_error']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # 保存检查点
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # 打印进度
            epoch_duration = datetime.now() - epoch_start_time
            logging.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Angular Error: {val_metrics['val_angular_error']:.4f}° | "
                f"GE2 Impact: {val_metrics['ge2_injection_impact']:.6f} | "
                f"GW Impact: {val_metrics['gw_injection_impact']:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Duration: {epoch_duration}"
            )
            
            # 早停检查
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                logging.info(f"Best epoch: {self.best_epoch+1}")
                break
        
        training_duration = datetime.now() - training_start_time
        
        # 保存训练历史
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 返回训练结果
        training_results = {
            'best_val_loss': self.best_val_loss,
            'best_val_angular_error': self.best_val_angular_error,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1,
            'training_duration': str(training_duration),
            'early_stopped': self.epochs_without_improvement >= self.early_stopping_patience
        }
        
        return training_results


def create_hierarchical_prior_trainer(model: nn.Module,
                                     train_loader: DataLoader,
                                     val_loader: DataLoader,
                                     loss_function: nn.Module,
                                     config: dict,
                                     device: str = 'cuda') -> HierarchicalPriorTrainer:
    """
    工厂函数：创建训练器
    """
    trainer = HierarchicalPriorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        config=config,
        device=device
    )
    
    return trainer

