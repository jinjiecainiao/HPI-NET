"""
自定义损失函数模块
实现Recovery Angular Error损失函数，用于多光谱光照估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging


class RecoveryAngularErrorLoss(nn.Module):
    """
    Recovery Angular Error损失函数
    
    计算预测光照和真实光照在RGB空间中的角度误差
    使用相机响应函数(CSF)将31维光谱转换为3维RGB
    """
    
    def __init__(self, 
                 csf_matrix: torch.Tensor,
                 reduction: str = 'mean',
                 epsilon: float = 1e-8,
                 use_cosine_similarity: bool = True):
        """
        初始化损失函数
        
        Args:
            csf_matrix: 相机响应函数矩阵 [31, 3]
            reduction: 损失缩减方式 ('mean', 'sum', 'none')
            epsilon: 数值稳定性常数
            use_cosine_similarity: 是否使用余弦相似度计算角度
        """
        super(RecoveryAngularErrorLoss, self).__init__()
        
        # 注册CSF矩阵为buffer（不参与梯度更新）
        if csf_matrix.shape != (31, 3):
            if csf_matrix.shape == (3, 31):
                csf_matrix = csf_matrix.T
            else:
                raise ValueError(f"CSF matrix shape should be (31, 3) or (3, 31), got {csf_matrix.shape}")
        
        self.register_buffer('csf_matrix', csf_matrix.float())
        self.reduction = reduction
        self.epsilon = epsilon
        self.use_cosine_similarity = use_cosine_similarity
    
    def spectral_to_rgb(self, spectral_data: torch.Tensor) -> torch.Tensor:
        """
        将31维光谱数据转换为3维RGB
        
        Args:
            spectral_data: 光谱数据 [..., 31]
        
        Returns:
            RGB数据 [..., 3]
        """
        # 矩阵乘法: [..., 31] x [31, 3] -> [..., 3]
        rgb_data = torch.matmul(spectral_data, self.csf_matrix)
        return rgb_data
    
    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        L2归一化向量（增强数值稳定性）
        
        Args:
            vectors: 输入向量 [..., D]
        
        Returns:
            归一化后的向量 [..., D]
        """
        norms = torch.norm(vectors, p=2, dim=-1, keepdim=True)
        # 使用更大的epsilon确保稳定性，并检查异常值
        norms_safe = torch.clamp(norms, min=1e-6)  # 防止接近零
        normalized = vectors / norms_safe
        
        # 额外检查：如果归一化后仍有异常值，替换为安全值
        normalized = torch.where(
            torch.isfinite(normalized),
            normalized,
            torch.zeros_like(normalized) + 1.0 / torch.sqrt(torch.tensor(vectors.shape[-1], dtype=vectors.dtype))
        )
        return normalized
    
    def compute_angular_error_cosine(self, 
                                   pred_rgb: torch.Tensor, 
                                   true_rgb: torch.Tensor) -> torch.Tensor:
        """
        使用余弦相似度计算角度误差（增强数值稳定性）
        
        Args:
            pred_rgb: 预测RGB [..., 3]
            true_rgb: 真实RGB [..., 3]
        
        Returns:
            角度误差（度） [...]
        """
        # 归一化
        pred_norm = self.normalize_vectors(pred_rgb)
        true_norm = self.normalize_vectors(true_rgb)
        
        # 计算余弦相似度
        cos_sim = torch.sum(pred_norm * true_norm, dim=-1)
        
        # 更严格的clamp：留出更大的安全边界
        # 使用1e-6而不是1e-8，避免arccos在边界处的数值不稳定
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # 计算角度（弧度）
        angle_rad = torch.acos(cos_sim)
        
        # 检查NaN/Inf
        if torch.isnan(angle_rad).any() or torch.isinf(angle_rad).any():
            logging.warning("NaN/Inf detected in angle calculation, using safe fallback")
            # 使用atan2作为备用方案（更稳定）
            cross_product = torch.cross(pred_norm, true_norm, dim=-1)
            cross_norm = torch.norm(cross_product, p=2, dim=-1)
            dot_product = torch.sum(pred_norm * true_norm, dim=-1)
            angle_rad = torch.atan2(cross_norm, dot_product)
        
        # 转换为度
        angle_deg = angle_rad * 180.0 / np.pi
        
        return angle_deg
    
    def compute_angular_error_cross(self, 
                                  pred_rgb: torch.Tensor, 
                                  true_rgb: torch.Tensor) -> torch.Tensor:
        """
        使用叉积计算角度误差（替代方法）
        
        Args:
            pred_rgb: 预测RGB [..., 3]
            true_rgb: 真实RGB [..., 3]
        
        Returns:
            角度误差（度） [...]
        """
        # 归一化
        pred_norm = self.normalize_vectors(pred_rgb)
        true_norm = self.normalize_vectors(true_rgb)
        
        # 计算叉积的模长
        cross_product = torch.cross(pred_norm, true_norm, dim=-1)
        cross_norm = torch.norm(cross_product, p=2, dim=-1)
        
        # 计算点积
        dot_product = torch.sum(pred_norm * true_norm, dim=-1)
        
        # 计算角度
        angle_rad = torch.atan2(cross_norm, dot_product)
        angle_deg = angle_rad * 180.0 / np.pi
        
        return angle_deg
    
    def forward(self, 
                pred_spectral: torch.Tensor, 
                true_spectral: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            pred_spectral: 预测的31维光谱 [B, 31]
            true_spectral: 真实的31维光谱 [B, 31]
        
        Returns:
            角度误差损失
        """
        # 转换到RGB空间
        pred_rgb = self.spectral_to_rgb(pred_spectral)  # [B, 3]
        true_rgb = self.spectral_to_rgb(true_spectral)  # [B, 3]
        
        # 计算角度误差
        if self.use_cosine_similarity:
            angular_errors = self.compute_angular_error_cosine(pred_rgb, true_rgb)
        else:
            angular_errors = self.compute_angular_error_cross(pred_rgb, true_rgb)
        
        # 应用缩减
        if self.reduction == 'mean':
            return torch.mean(angular_errors)
        elif self.reduction == 'sum':
            return torch.sum(angular_errors)
        elif self.reduction == 'none':
            return angular_errors
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def compute_metrics(self, 
                       pred_spectral: torch.Tensor, 
                       true_spectral: torch.Tensor) -> dict:
        """
        计算详细的评估指标
        
        Args:
            pred_spectral: 预测的31维光谱 [B, 31]
            true_spectral: 真实的31维光谱 [B, 31]
        
        Returns:
            包含各种指标的字典
        """
        with torch.no_grad():
            # 转换到RGB空间
            pred_rgb = self.spectral_to_rgb(pred_spectral)
            true_rgb = self.spectral_to_rgb(true_spectral)
            
            # 计算角度误差
            angular_errors = self.compute_angular_error_cosine(pred_rgb, true_rgb)
            
            # 计算统计指标
            metrics = {
                'mean_error': torch.mean(angular_errors).item(),
                'median_error': torch.median(angular_errors).item(),
                'std_error': torch.std(angular_errors).item(),
                'min_error': torch.min(angular_errors).item(),
                'max_error': torch.max(angular_errors).item(),
                'num_samples': angular_errors.shape[0]
            }
            
            # 计算百分位数
            percentiles = [25, 75, 90, 95, 99]
            for p in percentiles:
                percentile_value = torch.quantile(angular_errors, p / 100.0).item()
                metrics[f'p{p}_error'] = percentile_value
            
            # 计算小于特定阈值的样本比例
            thresholds = [1.0, 2.0, 3.0, 5.0, 10.0]
            for threshold in thresholds:
                ratio = (angular_errors < threshold).float().mean().item()
                metrics[f'ratio_below_{threshold}deg'] = ratio
            
            return metrics


class ReproductionAngularErrorLoss(nn.Module):
    """
    Reproduction Angular Error损失函数（公式9）
    
    计算RGB颜色向量与均匀白光(1,1,1)的角度误差
    e_rep(U, V) = arccos((U · (1,1,1)) / (||U|| * √3))
    
    这个指标衡量的是色彩再现质量，即光照估计对视觉颜色的影响
    """
    
    def __init__(self, 
                 csf_matrix: torch.Tensor,
                 reduction: str = 'mean',
                 epsilon: float = 1e-6,
                 use_cosine_similarity: bool = True):
        """
        初始化损失函数
        
        Args:
            csf_matrix: 相机响应函数矩阵 [31, 3]
            reduction: 损失缩减方式 ('mean', 'sum', 'none')
            epsilon: 数值稳定性常数
            use_cosine_similarity: 是否使用余弦相似度计算角度
        """
        super(ReproductionAngularErrorLoss, self).__init__()
        
        # 注册CSF矩阵为buffer
        if csf_matrix.shape != (31, 3):
            if csf_matrix.shape == (3, 31):
                csf_matrix = csf_matrix.T
            else:
                raise ValueError(f"CSF matrix shape should be (31, 3) or (3, 31), got {csf_matrix.shape}")
        
        self.register_buffer('csf_matrix', csf_matrix.float())
        self.reduction = reduction
        self.epsilon = epsilon
        self.use_cosine_similarity = use_cosine_similarity
        
        # 均匀白光向量 (1, 1, 1)，已归一化
        uniform_white = torch.ones(3, dtype=torch.float32) / np.sqrt(3.0)
        self.register_buffer('uniform_white', uniform_white)
    
    def spectral_to_rgb(self, spectral_data: torch.Tensor) -> torch.Tensor:
        """
        将31维光谱数据转换为3维RGB
        
        Args:
            spectral_data: 光谱数据 [..., 31]
        
        Returns:
            RGB数据 [..., 3]
        """
        rgb_data = torch.matmul(spectral_data, self.csf_matrix)
        return rgb_data
    
    def compute_reproduction_error(self, pred_rgb: torch.Tensor, true_rgb: torch.Tensor) -> torch.Tensor:
        """
        计算再现角度误差
        
        根据公式(9): e_rep(U, V) = arccos((U/V) · (1,1,1) / (||U/V|| * √3))
        其中：
        - U: 预测的RGB光照
        - V: 真实的RGB光照
        - U/V: 逐元素相除，得到色彩校正因子
        - (1,1,1): 均匀白光
        
        物理意义：
        - U/V 是色彩校正因子，理想情况下应该是 (1,1,1)
        - 该误差衡量预测光照导致的色偏程度
        
        Args:
            pred_rgb: 预测的RGB颜色向量 [..., 3]
            true_rgb: 真实的RGB颜色向量 [..., 3]
        
        Returns:
            再现角度误差（度） [...]
        """
        # 计算色彩校正因子: R = U/V（逐元素相除）
        ratio = pred_rgb / (true_rgb + self.epsilon)  # [..., 3]
        
        # 计算 R · (1,1,1)
        # 相当于 R₁ + R₂ + R₃
        uniform_white = torch.ones(3, dtype=pred_rgb.dtype, device=pred_rgb.device)
        dot_product = torch.sum(ratio * uniform_white, dim=-1)  # [...]
        
        # 计算 ||R||
        ratio_norm = torch.norm(ratio, p=2, dim=-1)  # [...]
        ratio_norm = torch.clamp(ratio_norm, min=self.epsilon)
        
        # 计算余弦值: (R · (1,1,1)) / (||R|| * √3)
        sqrt_3 = np.sqrt(3.0)
        cos_sim = dot_product / (ratio_norm * sqrt_3)
        
        # Clamp到有效范围
        cos_sim = torch.clamp(cos_sim, -1.0 + self.epsilon, 1.0 - self.epsilon)
        
        # 计算角度（弧度）
        angle_rad = torch.acos(cos_sim)
        
        # 转换为度
        angle_deg = angle_rad * 180.0 / np.pi
        
        return angle_deg
    
    def forward(self, 
                pred_spectral: torch.Tensor, 
                true_spectral: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        Args:
            pred_spectral: 预测的31维光谱 [B, 31]
            true_spectral: 真实的31维光谱 [B, 31]
        
        Returns:
            再现角度误差
        """
        # 转换到RGB空间
        pred_rgb = self.spectral_to_rgb(pred_spectral)  # [B, 3]
        true_rgb = self.spectral_to_rgb(true_spectral)  # [B, 3]
        
        # 计算再现角度误差
        # e_rep(U, V) = arccos((U/||U||) · (1,1,1) / (||V|| * √3))
        reproduction_errors = self.compute_reproduction_error(pred_rgb, true_rgb)  # [B]
        
        # 应用缩减
        if self.reduction == 'mean':
            return torch.mean(reproduction_errors)
        elif self.reduction == 'sum':
            return torch.sum(reproduction_errors)
        elif self.reduction == 'none':
            return reproduction_errors
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    结合Recovery Angular Error（公式8）和Reproduction Angular Error（公式9）
    可选辅助损失：光谱域MSE、平滑性损失
    """
    
    def __init__(self,
                 csf_matrix: torch.Tensor,
                 recovery_weight: float = 1.0,
                 reproduction_weight: float = 0.5,
                 spectral_weight: float = 0.0,
                 smoothness_weight: float = 0.0,
                 epsilon: float = 1e-6):
        """
        初始化组合损失
        
        Args:
            csf_matrix: 相机响应函数矩阵 [31, 3]
            recovery_weight: Recovery Angular Error权重（公式8）
            reproduction_weight: Reproduction Angular Error权重（公式9）
            spectral_weight: 光谱域MSE损失权重
            smoothness_weight: 平滑性损失权重
            epsilon: 数值稳定性常数
        """
        super(CombinedLoss, self).__init__()
        
        # Recovery Angular Error（公式8）
        self.recovery_loss_fn = RecoveryAngularErrorLoss(
            csf_matrix=csf_matrix,
            epsilon=epsilon
        )
        
        # Reproduction Angular Error（公式9）
        self.reproduction_loss_fn = ReproductionAngularErrorLoss(
            csf_matrix=csf_matrix,
            epsilon=epsilon
        )
        
        # 权重
        self.recovery_weight = recovery_weight
        self.reproduction_weight = reproduction_weight
        self.spectral_weight = spectral_weight
        self.smoothness_weight = smoothness_weight
    
    def spectral_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """光谱域MSE损失"""
        return F.mse_loss(pred, target)
    
    def smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """平滑性损失（相邻通道差异）"""
        # 计算相邻通道的差异
        diff = pred[:, 1:] - pred[:, :-1]  # [B, 30]
        return torch.mean(diff ** 2)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失
        
        Args:
            pred: 预测的光谱 [B, 31]
            target: 真实的光谱 [B, 31]
        
        Returns:
            总损失（标量）
        """
        total_loss = 0.0
        
        # Recovery Angular Error（公式8）- 主要损失
        # 衡量预测与真实光照的角度差异
        if self.recovery_weight > 0:
            recovery_loss = self.recovery_loss_fn(pred, target)
            total_loss = total_loss + self.recovery_weight * recovery_loss
        
        # Reproduction Angular Error（公式9）- 辅助损失
        # 衡量色彩校正因子（U/V）偏离白光的程度
        # e_rep(U, V) = arccos((U/V) · (1,1,1) / (||U/V|| * √3))
        # 物理意义：U/V应该接近(1,1,1)才能正确还原颜色
        if self.reproduction_weight > 0:
            reproduction_loss = self.reproduction_loss_fn(pred, target)
            total_loss = total_loss + self.reproduction_weight * reproduction_loss
        
        # 光谱域MSE损失（可选）
        if self.spectral_weight > 0:
            spectral_loss = self.spectral_mse_loss(pred, target)
            total_loss = total_loss + self.spectral_weight * spectral_loss
        
        # 平滑性损失（可选）
        if self.smoothness_weight > 0:
            smoothness = self.smoothness_loss(pred)
            total_loss = total_loss + self.smoothness_weight * smoothness
        
        return total_loss
    
    def compute_detailed_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        计算详细的各项损失（用于日志记录）
        
        Args:
            pred: 预测的光谱 [B, 31]
            target: 真实的光谱 [B, 31]
        
        Returns:
            包含各项损失的字典
        """
        losses = {}
        
        # Recovery Angular Error
        recovery_loss = self.recovery_loss_fn(pred, target)
        losses['recovery_loss'] = recovery_loss.item()
        
        # Reproduction Angular Error（需要预测值和真实值）
        reproduction_loss = self.reproduction_loss_fn(pred, target)
        losses['reproduction_loss'] = reproduction_loss.item()
        
        # 总损失
        total_loss = (self.recovery_weight * recovery_loss + 
                     self.reproduction_weight * reproduction_loss)
        
        # 可选损失
        if self.spectral_weight > 0:
            spectral_loss = self.spectral_mse_loss(pred, target)
            losses['spectral_loss'] = spectral_loss.item()
            total_loss = total_loss + self.spectral_weight * spectral_loss
        
        if self.smoothness_weight > 0:
            smoothness = self.smoothness_loss(pred)
            losses['smoothness_loss'] = smoothness.item()
            total_loss = total_loss + self.smoothness_weight * smoothness
        
        losses['total_loss'] = total_loss.item()
        
        return losses


def create_loss_function(config: dict, csf_matrix: torch.Tensor) -> nn.Module:
    """
    根据配置创建损失函数
    
    Args:
        config: 损失函数配置
        csf_matrix: 相机响应函数矩阵
    
    Returns:
        损失函数实例
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'recovery_angular_error')
    epsilon = float(loss_config.get('epsilon', 1e-6))
    
    if loss_type == 'recovery_angular_error':
        # 仅使用Recovery Angular Error（公式8）
        return RecoveryAngularErrorLoss(
            csf_matrix=csf_matrix,
            reduction=loss_config.get('reduction', 'mean'),
            epsilon=epsilon,
            use_cosine_similarity=loss_config.get('use_cosine_similarity', True)
        )
    
    elif loss_type == 'reproduction_angular_error':
        # 仅使用Reproduction Angular Error（公式9）
        return ReproductionAngularErrorLoss(
            csf_matrix=csf_matrix,
            reduction=loss_config.get('reduction', 'mean'),
            epsilon=epsilon
        )
    
    elif loss_type == 'combined':
        # 组合损失：Recovery + Reproduction（推荐）
        return CombinedLoss(
            csf_matrix=csf_matrix,
            recovery_weight=loss_config.get('recovery_weight', 1.0),
            reproduction_weight=loss_config.get('reproduction_weight', 0.5),
            spectral_weight=loss_config.get('spectral_weight', 0.0),
            smoothness_weight=loss_config.get('smoothness_weight', 0.0),
            epsilon=epsilon
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Available types: 'recovery_angular_error', "
                        f"'reproduction_angular_error', 'combined'")


if __name__ == "__main__":
    # 测试损失函数
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟CSF矩阵
    csf_matrix = torch.randn(31, 3)
    
    # 创建损失函数
    loss_fn = RecoveryAngularErrorLoss(csf_matrix)
    
    # 测试数据
    batch_size = 8
    pred_spectral = torch.randn(batch_size, 31)
    true_spectral = torch.randn(batch_size, 31)
    
    # 计算损失
    loss = loss_fn(pred_spectral, true_spectral)
    print(f"Recovery Angular Error Loss: {loss.item():.4f}°")
    
    # 计算详细指标
    metrics = loss_fn.compute_metrics(pred_spectral, true_spectral)
    print("\nDetailed Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试组合损失
    print("\nTesting Combined Loss:")
    combined_loss = CombinedLoss(csf_matrix)
    loss_dict = combined_loss(pred_spectral, true_spectral)
    
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("\nLoss function test completed successfully!")
