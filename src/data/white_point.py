"""
多光谱白点算法特征提取模块
实现多光谱域的白点算法，用于生成初始光照估计
"""

import torch
import numpy as np
from typing import Union, Tuple
import logging


class SpectralWhitePointExtractor:
    """
    多光谱白点特征提取器
    
    实现多光谱域的白点算法：
    1. 对每个光谱通道独立操作
    2. 找到每个通道的最大像素值
    3. 输出31维向量作为光照估计（不归一化，保持物理意义）
    """
    
    def __init__(self, 
                 normalize: bool = False,  # 默认不归一化
                 epsilon: float = 1e-8):
        """
        初始化白点提取器
        
        Args:
            normalize: 是否对输出进行L2归一化（通常应该为False）
            epsilon: 数值稳定性常数
        """
        self.normalize = normalize
        self.epsilon = epsilon
    
    def extract_features(self, 
                        multispectral_data: Union[torch.Tensor, np.ndarray],
                        method: str = 'max') -> torch.Tensor:
        """
        从多光谱数据中提取白点特征
        
        Args:
            multispectral_data: 多光谱数据 [H, W, 31] 或 [B, H, W, 31]
            method: 提取方法 ('max', 'percentile_99', 'mean_top_k')
        
        Returns:
            白点特征向量 [31] 或 [B, 31]
        """
        # 转换为torch张量
        if isinstance(multispectral_data, np.ndarray):
            ms_tensor = torch.from_numpy(multispectral_data).float()
        else:
            ms_tensor = multispectral_data.float()
        
        # 处理批次维度
        if ms_tensor.dim() == 3:
            # 单个样本 [H, W, 31]
            return self._extract_single(ms_tensor, method)
        elif ms_tensor.dim() == 4:
            # 批次样本 [B, H, W, 31]
            batch_features = []
            for i in range(ms_tensor.shape[0]):
                features = self._extract_single(ms_tensor[i], method)
                batch_features.append(features)
            return torch.stack(batch_features)
        else:
            raise ValueError(f"Invalid input shape: {ms_tensor.shape}")
    
    def _extract_single(self, 
                       ms_data: torch.Tensor, 
                       method: str) -> torch.Tensor:
        """
        从单个多光谱图像提取白点特征
        
        Args:
            ms_data: 多光谱数据 [H, W, 31]
            method: 提取方法
        
        Returns:
            白点特征向量 [31]
        """
        H, W, C = ms_data.shape
        assert C == 31, f"Expected 31 channels, got {C}"
        
        # 重塑为 [H*W, 31] 便于处理
        pixels = ms_data.view(-1, 31)  # [H*W, 31]
        
        if method == 'max':
            # 标准白点：每个通道的最大值
            wp_features = torch.max(pixels, dim=0)[0]  # [31]
            
        elif method == 'percentile_99':
            # 99百分位数（更鲁棒，避免噪声影响）
            wp_features = torch.quantile(pixels, 0.99, dim=0)  # [31]
            
        elif method == 'mean_top_k':
            # 前K个最大值的均值（K = 1% of pixels）
            k = max(1, int(0.01 * pixels.shape[0]))
            top_k_values, _ = torch.topk(pixels, k, dim=0)
            wp_features = torch.mean(top_k_values, dim=0)  # [31]
            
        elif method == 'robust_max':
            # 鲁棒最大值：排除极端异常值后的最大值
            # 先计算每个通道的95百分位数
            p95 = torch.quantile(pixels, 0.95, dim=0)
            # 将超过95百分位数的值限制为95百分位数值
            clipped_pixels = torch.clamp(pixels, max=p95.unsqueeze(0))
            wp_features = torch.max(clipped_pixels, dim=0)[0]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # L2归一化
        if self.normalize:
            norm = torch.norm(wp_features)
            if norm > self.epsilon:
                wp_features = wp_features / norm
        
        return wp_features
    
    def extract_batch_features(self, 
                             batch_data: torch.Tensor,
                             method: str = 'max') -> torch.Tensor:
        """
        批量提取白点特征（优化版本）
        
        Args:
            batch_data: 批次多光谱数据 [B, H, W, 31]
            method: 提取方法
        
        Returns:
            批次白点特征 [B, 31]
        """
        B, H, W, C = batch_data.shape
        assert C == 31, f"Expected 31 channels, got {C}"
        
        # 重塑为 [B, H*W, 31]
        pixels = batch_data.view(B, -1, 31)
        
        if method == 'max':
            wp_features = torch.max(pixels, dim=1)[0]  # [B, 31]
        elif method == 'percentile_99':
            wp_features = torch.quantile(pixels, 0.99, dim=1)  # [B, 31]
        elif method == 'mean_top_k':
            k = max(1, int(0.01 * pixels.shape[1]))
            top_k_values, _ = torch.topk(pixels, k, dim=1)
            wp_features = torch.mean(top_k_values, dim=1)  # [B, 31]
        else:
            # 对于其他方法，逐个处理
            return self.extract_features(batch_data, method)
        
        # 批量L2归一化
        if self.normalize:
            norms = torch.norm(wp_features, dim=1, keepdim=True)
            wp_features = wp_features / (norms + self.epsilon)
        
        return wp_features
    
    def compare_methods(self, 
                       multispectral_data: torch.Tensor,
                       ground_truth: torch.Tensor = None) -> dict:
        """
        比较不同白点提取方法的性能
        
        Args:
            multispectral_data: 多光谱数据 [H, W, 31]
            ground_truth: 地面真值光照 [31] (可选)
        
        Returns:
            包含不同方法结果的字典
        """
        methods = ['max', 'percentile_99', 'mean_top_k', 'robust_max']
        results = {}
        
        for method in methods:
            try:
                wp_features = self.extract_features(multispectral_data, method)
                results[method] = {
                    'features': wp_features,
                    'norm': torch.norm(wp_features).item()
                }
                
                # 如果有地面真值，计算角度误差
                if ground_truth is not None:
                    angle_error = self._compute_angular_error(wp_features, ground_truth)
                    results[method]['angular_error'] = angle_error
                    
            except Exception as e:
                logging.warning(f"Method {method} failed: {e}")
                results[method] = {'error': str(e)}
        
        return results
    
    def _compute_angular_error(self, 
                             pred: torch.Tensor, 
                             target: torch.Tensor) -> float:
        """计算两个向量间的角度误差（度）"""
        # 归一化
        pred_norm = pred / (torch.norm(pred) + self.epsilon)
        target_norm = target / (torch.norm(target) + self.epsilon)
        
        # 计算余弦相似度
        cos_sim = torch.clamp(torch.dot(pred_norm, target_norm), -1.0, 1.0)
        
        # 转换为角度（度）
        angle_rad = torch.acos(cos_sim)
        angle_deg = torch.rad2deg(angle_rad)
        
        return angle_deg.item()


def batch_white_point_extraction(dataloader,
                                extractor: SpectralWhitePointExtractor,
                                method: str = 'max',
                                device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    批量提取整个数据集的白点特征
    
    Args:
        dataloader: 数据加载器
        extractor: 白点提取器
        method: 提取方法
        device: 计算设备
    
    Returns:
        (wp_features, ground_truth): 白点特征和地面真值
    """
    all_wp_features = []
    all_ground_truth = []
    
    extractor_device = SpectralWhitePointExtractor(
        normalize=extractor.normalize,
        epsilon=extractor.epsilon
    )
    
    with torch.no_grad():
        for batch in dataloader:
            ms_data = batch['multispectral'].to(device)  # [B, H, W, 31]
            gt_illum = batch['illumination_gt'].to(device)  # [B, 31]
            
            # 提取白点特征
            wp_features = extractor_device.extract_batch_features(ms_data, method)
            
            all_wp_features.append(wp_features.cpu())
            all_ground_truth.append(gt_illum.cpu())
    
    # 拼接所有批次
    wp_features_all = torch.cat(all_wp_features, dim=0)
    ground_truth_all = torch.cat(all_ground_truth, dim=0)
    
    return wp_features_all, ground_truth_all


if __name__ == "__main__":
    # 测试白点提取器
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    H, W, C = 256, 256, 31
    ms_data = torch.rand(H, W, C) * 0.8  # 基础反射率
    
    # 添加一些"白点"（高反射率区域）
    ms_data[100:120, 100:120, :] = torch.rand(20, 20, C) * 0.3 + 0.7
    ms_data[200:210, 200:210, :] = torch.rand(10, 10, C) * 0.2 + 0.8
    
    # 模拟光照（某些通道更强）
    illumination = torch.ones(31)
    illumination[10:20] *= 2.0  # 中波段更强
    illumination[25:31] *= 1.5  # 长波段较强
    
    # 应用光照
    ms_data = ms_data * illumination.unsqueeze(0).unsqueeze(0)
    
    # 测试提取器
    extractor = SpectralWhitePointExtractor(normalize=True)
    
    # 比较不同方法
    results = extractor.compare_methods(ms_data, illumination / torch.norm(illumination))
    
    print("White Point Extraction Results:")
    for method, result in results.items():
        if 'error' not in result:
            print(f"{method}:")
            print(f"  Norm: {result['norm']:.4f}")
            if 'angular_error' in result:
                print(f"  Angular Error: {result['angular_error']:.2f}°")
        else:
            print(f"{method}: {result['error']}")
    
    # 测试批量处理
    batch_data = torch.stack([ms_data, ms_data * 0.9], dim=0)  # [2, H, W, 31]
    batch_wp = extractor.extract_batch_features(batch_data)
    print(f"\nBatch WP features shape: {batch_wp.shape}")
    print(f"Batch WP norms: {torch.norm(batch_wp, dim=1)}")
