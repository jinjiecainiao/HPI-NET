"""
Angular Error Loss Module
角误差损失函数

"""

import torch
import torch.nn as nn
import numpy as np


class AngularErrorLoss(nn.Module):
    """
    角误差损失函数
    
    计算预测光照与真实光照之间的角度误差（以度为单位）
    
    支持两种模式：
    - use_csf=False: 直接在31维光谱空间计算（简单但不准确）
    - use_csf=True: 通过CSF转换到RGB空间计算（物理意义正确，推荐）
    """
    
    def __init__(self, epsilon: float = 1e-8, use_csf: bool = False, csf_matrix: torch.Tensor = None):
        """
        初始化角误差损失
        
        Args:
            epsilon: 数值稳定性常数
            use_csf: 是否使用CSF转换到RGB空间
            csf_matrix: 相机灵敏度矩阵 [3, 31]，当use_csf=True时必须提供
        """
        super(AngularErrorLoss, self).__init__()
        self.epsilon = epsilon
        self.use_csf = use_csf
        
        if use_csf:
            if csf_matrix is None:
                raise ValueError("use_csf=True时必须提供csf_matrix")
            if csf_matrix.shape != (3, 31):
                raise ValueError(f"csf_matrix形状应为[3,31]，但得到{csf_matrix.shape}")
            # 注册为buffer（不参与训练但会随模型移动到GPU）
            self.register_buffer('csf_matrix', csf_matrix)
        else:
            self.csf_matrix = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        计算角误差损失
        
        Args:
            pred: 预测光照 [B, 31]
            target: 真实光照 [B, 31]
        
        Returns:
            loss_dict: 包含损失和误差的字典
        """
        if self.use_csf and self.csf_matrix is not None:
            # 模式1: 通过CSF转换到RGB空间（推荐）
            # pred: [B, 31] @ csf.T: [31, 3] = [B, 3]
            pred_rgb = torch.matmul(pred, self.csf_matrix.t())  # [B, 31] x [31, 3] = [B, 3]
            target_rgb = torch.matmul(target, self.csf_matrix.t())
            
            # 在RGB空间计算角误差
            angular_error_deg = self._compute_angular_error(pred_rgb, target_rgb)
            
            return {
                'total_loss': torch.mean(angular_error_deg),
                'angular_error': torch.mean(angular_error_deg),
                'recovery_angular_error': torch.mean(angular_error_deg),  # 恢复误差（RGB空间）
                'angular_error_rad': torch.mean(angular_error_deg) * np.pi / 180.0
            }
        else:
            # 模式2: 直接在31维光谱空间计算（简单但不够准确）
            angular_error_deg = self._compute_angular_error(pred, target)
            
            return {
                'total_loss': torch.mean(angular_error_deg),
                'angular_error': torch.mean(angular_error_deg),
                'spectral_angular_error': torch.mean(angular_error_deg),  # 光谱误差（31维空间）
                'angular_error_rad': torch.mean(angular_error_deg) * np.pi / 180.0
            }
    
    def _compute_angular_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算角误差（度）
        
        Args:
            pred: 预测向量 [B, D]
            target: 真实向量 [B, D]
        
        Returns:
            angular_error: 角误差（度）[B]
        """
        # 归一化
        pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + self.epsilon)
        target_norm = target / (torch.norm(target, dim=1, keepdim=True) + self.epsilon)
        
        # 计算余弦相似度
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + self.epsilon, 1.0 - self.epsilon)
        
        # 计算角度（弧度转度）
        angular_error_rad = torch.acos(cos_sim)
        angular_error_deg = angular_error_rad * 180.0 / np.pi
        
        return angular_error_deg


def compute_angular_error(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    计算角误差（度）- 工具函数
    
    Args:
        pred: 预测光照 [B, 31]
        target: 真实光照 [B, 31]
        epsilon: 数值稳定性常数
    
    Returns:
        angular_error: 角误差（度）[B]
    """
    # 归一化
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + epsilon)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + epsilon)
    
    # 计算余弦相似度
    cos_sim = torch.sum(pred_norm * target_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0 + epsilon, 1.0 - epsilon)
    
    # 计算角度（度）
    angular_error = torch.acos(cos_sim) * 180.0 / np.pi
    
    return angular_error


