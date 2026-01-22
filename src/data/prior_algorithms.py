"""
经典光照估计算法实现模块
实现多种经典算法作为先验：WP, GW (Grey World), GE (Grey Edge)等
"""

import torch
import numpy as np
from typing import Union, List, Tuple
import logging


class ClassicalIlluminationEstimator:
    """
    经典光照估计算法集合
    提供多种经典算法的统一接口
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        初始化经典算法估计器
        
        Args:
            epsilon: 数值稳定性常数
        """
        self.epsilon = epsilon
    
    def white_patch(self, ms_data: torch.Tensor, percentile: float = 100.0) -> torch.Tensor:
        """
        White Patch (WP) 算法 - 最大值假设
        假设场景中存在白色区域，其反射率为1，因此最大响应即为光源
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
            percentile: 使用百分位数而非最大值 (100.0 = max)
        
        Returns:
            光照估计 [31]
        """
        if ms_data.dim() != 3:
            raise ValueError(f"Expected 3D input [H, W, C], got {ms_data.shape}")
        
        H, W, C = ms_data.shape
        pixels = ms_data.view(-1, C)  # [H*W, 31]
        
        if percentile >= 99.9:
            # 标准WP: 使用最大值
            illumination = torch.max(pixels, dim=0)[0]
        else:
            # 鲁棒WP: 使用百分位数
            illumination = torch.quantile(pixels, percentile / 100.0, dim=0)
        
        # 确保非负且有意义
        illumination = torch.clamp(illumination, min=self.epsilon)
        
        return illumination
    
    def grey_world(self, ms_data: torch.Tensor, norm: int = 1) -> torch.Tensor:
        """
        Grey World (GW) 算法 - 平均值假设
        假设场景的平均反射率为灰色，因此平均响应即为光源
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
            norm: 使用的范数 (1=L1平均, 2=L2平均)
        
        Returns:
            光照估计 [31]
        """
        if ms_data.dim() != 3:
            raise ValueError(f"Expected 3D input [H, W, C], got {ms_data.shape}")
        
        H, W, C = ms_data.shape
        pixels = ms_data.view(-1, C)  # [H*W, 31]
        
        if norm == 1:
            # 标准GW: 算术平均
            illumination = torch.mean(pixels, dim=0)
        elif norm == 2:
            # L2范数平均
            illumination = torch.sqrt(torch.mean(pixels ** 2, dim=0))
        else:
            # 通用Lp范数
            illumination = torch.pow(torch.mean(pixels ** norm, dim=0), 1.0 / norm)
        
        illumination = torch.clamp(illumination, min=self.epsilon)
        
        return illumination
    
    def grey_edge(self, ms_data: torch.Tensor, order: int = 1, norm: int = 1) -> torch.Tensor:
        """
        Grey Edge (GE) 算法 - 边缘统计假设
        假设场景边缘的平均导数为灰色
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
            order: 导数阶数 (1=一阶, 2=二阶)
            norm: 使用的范数 (1=L1平均, 2=L2平均)
        
        Returns:
            光照估计 [31]
        """
        if ms_data.dim() != 3:
            raise ValueError(f"Expected 3D input [H, W, C], got {ms_data.shape}")
        
        H, W, C = ms_data.shape
        
        # 计算导数（沿x和y方向）
        if order == 1:
            # 一阶导数
            dx = torch.diff(ms_data, dim=1, prepend=ms_data[:, :1, :])  # [H, W, 31]
            dy = torch.diff(ms_data, dim=0, prepend=ms_data[:1, :, :])  # [H, W, 31]
        elif order == 2:
            # 二阶导数
            dx1 = torch.diff(ms_data, dim=1, prepend=ms_data[:, :1, :])
            dx2 = torch.diff(dx1, dim=1, prepend=dx1[:, :1, :])
            dy1 = torch.diff(ms_data, dim=0, prepend=ms_data[:1, :, :])
            dy2 = torch.diff(dy1, dim=0, prepend=dy1[:1, :, :])
            dx, dy = dx2, dy2
        else:
            raise ValueError(f"Unsupported derivative order: {order}")
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(dx ** 2 + dy ** 2 + self.epsilon)
        
        # 展平
        gradient_flat = gradient_magnitude.view(-1, C)  # [H*W, 31]
        
        # 应用范数统计
        if norm == 1:
            illumination = torch.mean(gradient_flat, dim=0)
        elif norm == 2:
            illumination = torch.sqrt(torch.mean(gradient_flat ** 2, dim=0))
        else:
            illumination = torch.pow(torch.mean(gradient_flat ** norm, dim=0), 1.0 / norm)
        
        illumination = torch.clamp(illumination, min=self.epsilon)
        
        return illumination
    
    def shades_of_grey(self, ms_data: torch.Tensor, p: float = 6.0) -> torch.Tensor:
        """
        Shades of Grey (SoG) 算法 - Minkowski范数
        WP和GW的统一框架，p控制偏向哪一端
        p→∞: 接近WP, p=1: 等于GW
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
            p: Minkowski范数参数
        
        Returns:
            光照估计 [31]
        """
        if ms_data.dim() != 3:
            raise ValueError(f"Expected 3D input [H, W, C], got {ms_data.shape}")
        
        H, W, C = ms_data.shape
        pixels = ms_data.view(-1, C)  # [H*W, 31]
        
        if p < 100:
            # 标准Minkowski范数
            illumination = torch.pow(torch.mean(pixels ** p, dim=0), 1.0 / p)
        else:
            # p很大时，近似为最大值
            illumination = torch.max(pixels, dim=0)[0]
        
        illumination = torch.clamp(illumination, min=self.epsilon)
        
        return illumination
    
    def compute_all_priors(self, ms_data: torch.Tensor) -> torch.Tensor:
        """
        计算所有先验算法的结果
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
        
        Returns:
            所有先验特征 [K, 31]，K为算法数量
        """
        priors = []
        
        # 1. White Patch (标准)
        priors.append(self.white_patch(ms_data, percentile=100.0))
        
        # 2. Grey World
        priors.append(self.grey_world(ms_data, norm=1))
        
        # 3. Grey Edge (一阶, L1)
        priors.append(self.grey_edge(ms_data, order=1, norm=1))
        
        # 4. Grey Edge (一阶, L2)
        priors.append(self.grey_edge(ms_data, order=1, norm=2))
        
        # 5. Shades of Grey (p=6)
        priors.append(self.shades_of_grey(ms_data, p=6.0))
        
        # 堆叠成 [K, 31]
        priors_tensor = torch.stack(priors, dim=0)
        
        return priors_tensor
    
    def compute_selected_priors(self, ms_data: torch.Tensor, 
                               selected_methods: List[str]) -> torch.Tensor:
        """
        计算指定的先验算法
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
            selected_methods: 方法名称列表，例如 ['WP', 'GW', 'GE1', 'GE2']
        
        Returns:
            选定的先验特征 [K, 31]
        """
        method_map = {
            'WP': lambda: self.white_patch(ms_data, percentile=100.0),
            'WP_robust': lambda: self.white_patch(ms_data, percentile=99.0),
            'GW': lambda: self.grey_world(ms_data, norm=1),
            'GW_L2': lambda: self.grey_world(ms_data, norm=2),
            'GE1': lambda: self.grey_edge(ms_data, order=1, norm=1),
            'GE1_L2': lambda: self.grey_edge(ms_data, order=1, norm=2),
            'GE2': lambda: self.grey_edge(ms_data, order=2, norm=1),
            'GE2_L2': lambda: self.grey_edge(ms_data, order=2, norm=2),
            'SoG': lambda: self.shades_of_grey(ms_data, p=6.0),
        }
        
        priors = []
        for method_name in selected_methods:
            if method_name in method_map:
                try:
                    prior = method_map[method_name]()
                    
                    # 只在出现NaN/Inf时才干预（最小干预原则）
                    if torch.isnan(prior).any() or torch.isinf(prior).any():
                        logging.warning(f"{method_name} produced NaN/Inf, replacing with safe values")
                        prior = torch.where(torch.isfinite(prior), prior, torch.tensor(0.1))
                    
                    # 只保证非负，不限制上限（让模型自由处理各种亮度）
                    prior = torch.clamp(prior, min=1e-8)
                    
                    priors.append(prior)
                except Exception as e:
                    logging.warning(f"Failed to compute {method_name}: {e}")
                    # 使用合理的默认值作为fallback（而不是全零）
                    priors.append(torch.ones(31) * 0.1)
            else:
                logging.warning(f"Unknown method: {method_name}, using default values")
                priors.append(torch.ones(31) * 0.1)
        
        if len(priors) == 0:
            raise ValueError("No valid priors computed")
        
        priors_tensor = torch.stack(priors, dim=0)
        
        # 最后再检查一次整个tensor
        if torch.isnan(priors_tensor).any() or torch.isinf(priors_tensor).any():
            logging.error("NaN/Inf found in final priors_tensor, applying emergency fix")
            priors_tensor = torch.where(
                torch.isfinite(priors_tensor), 
                priors_tensor, 
                torch.tensor(0.1)
            )
        
        return priors_tensor


def compute_batch_priors(batch_ms_data: torch.Tensor,
                        selected_methods: List[str] = None,
                        epsilon: float = 1e-8) -> torch.Tensor:
    """
    批量计算先验特征
    
    Args:
        batch_ms_data: 批次多光谱数据 [B, H, W, 31] 或 [B, 31, H, W]
        selected_methods: 方法名称列表，默认使用['WP', 'GW', 'GE1', 'GE2']
        epsilon: 数值稳定性常数
    
    Returns:
        批次先验特征 [B, K, 31]
    """
    if batch_ms_data.dim() != 4:
        raise ValueError(f"Expected 4D input [B, C, H, W] or [B, H, W, C], got {batch_ms_data.shape}")
    
    # 检测数据格式并转换为 [B, H, W, C]
    if batch_ms_data.shape[1] == 31:
        # [B, 31, H, W] -> [B, H, W, 31]
        batch_ms_data = batch_ms_data.permute(0, 2, 3, 1)
    
    if selected_methods is None:
        selected_methods = ['WP', 'GW', 'GE1', 'GE2']
    
    estimator = ClassicalIlluminationEstimator(epsilon=epsilon)
    
    batch_priors = []
    for i in range(batch_ms_data.shape[0]):
        ms_data = batch_ms_data[i]  # [H, W, 31]
        priors = estimator.compute_selected_priors(ms_data, selected_methods)  # [K, 31]
        batch_priors.append(priors)
    
    # 堆叠成 [B, K, 31]
    batch_priors_tensor = torch.stack(batch_priors, dim=0)
    
    return batch_priors_tensor


if __name__ == "__main__":
    # 测试经典算法
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Classical Illumination Estimation Algorithms")
    print("=" * 60)
    
    # 创建模拟多光谱数据
    H, W, C = 132, 176, 31
    
    # 模拟一个简单场景：均匀光照 + 不同反射率
    # 光照: 中波段更强
    true_illumination = torch.ones(31)
    true_illumination[10:20] *= 2.0  # 中波段强
    true_illumination[25:31] *= 1.5  # 长波段较强
    
    # 反射率: 随机生成，范围 [0.1, 0.9]
    reflectance = torch.rand(H, W, C) * 0.8 + 0.1
    
    # 添加一些高反射率区域（模拟白色表面）
    reflectance[50:70, 80:100, :] = torch.rand(20, 20, C) * 0.2 + 0.8
    
    # 生成观测数据: observation = reflectance * illumination
    ms_data = reflectance * true_illumination.unsqueeze(0).unsqueeze(0)
    
    # 创建估计器
    estimator = ClassicalIlluminationEstimator()
    
    print(f"\n📊 Input data shape: {ms_data.shape}")
    print(f"🎯 True illumination range: [{true_illumination.min():.3f}, {true_illumination.max():.3f}]")
    
    # 测试各个算法
    print("\n🔬 Testing individual algorithms:")
    print("-" * 60)
    
    # 辅助函数: 计算角误差
    def angular_error(pred, target, eps=1e-8):
        pred_norm = pred / (torch.norm(pred) + eps)
        target_norm = target / (torch.norm(target) + eps)
        cos_sim = torch.clamp(torch.dot(pred_norm, target_norm), -1.0, 1.0)
        angle = torch.acos(cos_sim)
        return torch.rad2deg(angle).item()
    
    # 1. White Patch
    wp = estimator.white_patch(ms_data)
    wp_error = angular_error(wp, true_illumination)
    print(f"1. White Patch:")
    print(f"   Range: [{wp.min():.3f}, {wp.max():.3f}]")
    print(f"   Angular Error: {wp_error:.2f}°")
    
    # 2. Grey World
    gw = estimator.grey_world(ms_data)
    gw_error = angular_error(gw, true_illumination)
    print(f"2. Grey World:")
    print(f"   Range: [{gw.min():.3f}, {gw.max():.3f}]")
    print(f"   Angular Error: {gw_error:.2f}°")
    
    # 3. Grey Edge (1st order)
    ge1 = estimator.grey_edge(ms_data, order=1, norm=1)
    ge1_error = angular_error(ge1, true_illumination)
    print(f"3. Grey Edge (1st order, L1):")
    print(f"   Range: [{ge1.min():.3f}, {ge1.max():.3f}]")
    print(f"   Angular Error: {ge1_error:.2f}°")
    
    # 4. Grey Edge (2nd order)
    ge2 = estimator.grey_edge(ms_data, order=2, norm=1)
    ge2_error = angular_error(ge2, true_illumination)
    print(f"4. Grey Edge (2nd order, L1):")
    print(f"   Range: [{ge2.min():.3f}, {ge2.max():.3f}]")
    print(f"   Angular Error: {ge2_error:.2f}°")
    
    # 5. Shades of Grey
    sog = estimator.shades_of_grey(ms_data, p=6.0)
    sog_error = angular_error(sog, true_illumination)
    print(f"5. Shades of Grey (p=6):")
    print(f"   Range: [{sog.min():.3f}, {sog.max():.3f}]")
    print(f"   Angular Error: {sog_error:.2f}°")
    
    # 测试批量计算
    print("\n🚀 Testing batch computation:")
    print("-" * 60)
    
    batch_size = 4
    batch_data = ms_data.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [B, H, W, 31]
    
    selected_methods = ['WP', 'GW', 'GE1', 'GE2']
    batch_priors = compute_batch_priors(batch_data, selected_methods)
    
    print(f"Batch priors shape: {batch_priors.shape}")
    print(f"Selected methods: {selected_methods}")
    print(f"Priors per sample: {len(selected_methods)}")
    
    # 显示第一个样本的先验
    print(f"\nFirst sample priors:")
    for i, method in enumerate(selected_methods):
        prior = batch_priors[0, i]
        error = angular_error(prior, true_illumination)
        print(f"  {method}: Range [{prior.min():.3f}, {prior.max():.3f}], Error: {error:.2f}°")
    
    print("\n✅ All tests completed successfully!")
    print("=" * 60)

