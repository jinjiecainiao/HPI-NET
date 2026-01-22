"""
层次化先验注入网络 (Hierarchical Prior Injection Network)
核心思想：在网络的不同层次注入特性匹配的经典先验

设计理念：
1. 浅层注入GE2（局部特征）- 引导边缘和纹理特征学习
2. 深层注入GW（全局特征）- 引导全局颜色分布学习
3. 基于V3轻量化架构（0.9M参数）

研究假设：
层次化注入（局部先验→浅层，全局先验→深层）优于单一注入或后期融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import logging


class PriorInjectionModule(nn.Module):
    """
    先验注入模块 - 将31维先验向量注入到特征图中
    
    机制：
    1. 线性投影：31维 -> C维（匹配特征图通道数）
    2. 空间扩展：[B, C] -> [B, C, 1, 1]
    3. 特征融合：通过广播相加 [B, C, 1, 1] + [B, C, H, W]
    
    优势：
    - 参数高效（仅一个线性层）
    - 空间不变性（所有位置使用相同先验）
    - 物理意义明确（先验作为全局偏置）
    """
    
    def __init__(self, prior_dim: int = 31, feature_channels: int = 64):
        """
        初始化先验注入模块
        
        Args:
            prior_dim: 先验向量维度（31）
            feature_channels: 目标特征图通道数
        """
        super(PriorInjectionModule, self).__init__()
        
        self.prior_dim = prior_dim
        self.feature_channels = feature_channels
        
        # 线性投影层：31 -> C
        self.projection = nn.Linear(prior_dim, feature_channels)
        
        # 初始化为小值，避免初期对特征图产生过大影响
        nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
        nn.init.constant_(self.projection.bias, 0.0)
        
        logging.debug(f"PriorInjectionModule: {prior_dim} -> {feature_channels}")
    
    def forward(self, features: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        将先验注入到特征图中
        
        Args:
            features: 特征图 [B, C, H, W]
            prior: 先验向量 [B, 31]
        
        Returns:
            injected_features: 注入后的特征图 [B, C, H, W]
        """
        # 投影先验到特征空间
        prior_proj = self.projection(prior)  # [B, 31] -> [B, C]
        
        # 扩展到空间维度
        prior_spatial = prior_proj.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 通过广播相加注入特征
        injected_features = features + prior_spatial  # [B, C, H, W]
        
        return injected_features


class LightweightSpectralExtractor(nn.Module):
    """
    轻量级光谱特征提取器（基于V3架构）
    
    架构：
    - Conv1: 31 -> 64  (stride=2) + BN + ReLU + MaxPool
    - Conv2: 64 -> 128 (stride=2) + BN + ReLU + MaxPool
    - Conv3: 128 -> 256 (stride=2) + BN + ReLU + AdaptiveAvgPool
    
    创新点：支持在Conv1和Conv3后注入不同的先验
    """
    
    def __init__(self,
                 input_channels: int = 31,
                 hidden_channels: List[int] = None,
                 dropout_rate: float = 0.3,
                 inject_ge2_after_conv1: bool = True,
                 inject_gw_after_conv3: bool = True):
        """
        初始化光谱特征提取器
        
        Args:
            input_channels: 输入通道数（31）
            hidden_channels: 隐藏层通道数列表 [64, 128, 256]
            dropout_rate: Dropout比率
            inject_ge2_after_conv1: 是否在Conv1后注入GE2
            inject_gw_after_conv3: 是否在Conv3后注入GW
        """
        super(LightweightSpectralExtractor, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        
        self.hidden_channels = hidden_channels
        self.inject_ge2_after_conv1 = inject_ge2_after_conv1
        self.inject_gw_after_conv3 = inject_gw_after_conv3
        
        # Conv1: 31 -> 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5)  # 浅层使用较小dropout
        )
        
        # 【注入点1】GE2注入模块（浅层 - 局部特征）
        if inject_ge2_after_conv1:
            self.ge2_injection = PriorInjectionModule(
                prior_dim=31,
                feature_channels=hidden_channels[0]
            )
        
        # Conv2: 64 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.7)
        )
        
        # Conv3: 128 -> 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # 【注入点2】GW注入模块（深层 - 全局特征）
        if inject_gw_after_conv3:
            self.gw_injection = PriorInjectionModule(
                prior_dim=31,
                feature_channels=hidden_channels[2]
            )
        
        # 全局自适应平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特征维度
        self.feature_dim = hidden_channels[2]
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, 
                ge2_prior: Optional[torch.Tensor] = None,
                gw_prior: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播：层次化注入先验
        
        Args:
            x: 输入图像 [B, 31, H, W]
            ge2_prior: GE2先验 [B, 31]（浅层注入）
            gw_prior: GW先验 [B, 31]（深层注入）
        
        Returns:
            features: 提取的特征 [B, 256]
            injection_info: 注入信息字典
        """
        injection_info = {}
        
        # Stage 1: Conv1 + GE2注入（浅层 - 局部特征）
        x = self.conv1(x)  # [B, 64, H/4, W/4]
        
        if self.inject_ge2_after_conv1 and ge2_prior is not None:
            x_before = x.clone()
            x = self.ge2_injection(x, ge2_prior)
            injection_info['ge2_injected'] = True
            injection_info['ge2_impact'] = (x - x_before).abs().mean().item()
        else:
            injection_info['ge2_injected'] = False
        
        # Stage 2: Conv2
        x = self.conv2(x)  # [B, 128, H/16, W/16]
        
        # Stage 3: Conv3 + GW注入（深层 - 全局特征）
        x = self.conv3(x)  # [B, 256, H/32, W/32]
        
        if self.inject_gw_after_conv3 and gw_prior is not None:
            x_before = x.clone()
            x = self.gw_injection(x, gw_prior)
            injection_info['gw_injected'] = True
            injection_info['gw_impact'] = (x - x_before).abs().mean().item()
        else:
            injection_info['gw_injected'] = False
        
        # Global pooling
        x = self.global_pool(x)  # [B, 256, 1, 1]
        features = x.view(x.size(0), -1)  # [B, 256]
        
        return features, injection_info


class LightweightResNetRegressor(nn.Module):
    """
    轻量级ResNet回归头（基于V3架构）
    
    架构：
    - Input: 256-dim
    - Hidden: [128, 256, 128]
    - ResBlocks: 3 blocks
    - Output: 31-dim illuminant spectrum
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 output_dim: int = 31,
                 hidden_dims: List[int] = None,
                 num_residual_blocks: int = 3,
                 dropout_rate: float = 0.3,
                 l2_regularization: float = 5e-5):
        """
        初始化ResNet回归头
        
        Args:
            input_dim: 输入维度（256）
            output_dim: 输出维度（31）
            hidden_dims: 隐藏层维度列表 [128, 256, 128]
            num_residual_blocks: 残差块数量
            dropout_rate: Dropout比率
            l2_regularization: L2正则化强度
        """
        super(LightweightResNetRegressor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        self.l2_regularization = l2_regularization
        self.hidden_dims = hidden_dims
        
        # Input projection: 256 -> 128
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # First hidden layer: 128 -> 256
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks at 256-dim
        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            block = ResidualBlock1D(
                dim=hidden_dims[1],
                dropout_rate=dropout_rate
            )
            self.residual_blocks.append(block)
        
        # Second hidden layer: 256 -> 128
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Output projection: 128 -> 31
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus()  # 确保非负输出
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, 256]
        
        Returns:
            prediction: 光照估计 [B, 31]
        """
        # Input projection: 256 -> 128
        x = self.input_proj(x)  # [B, 128]
        
        # First hidden layer: 128 -> 256
        x = self.hidden_layer1(x)  # [B, 256]
        
        # Apply residual blocks at 256-dim
        for res_block in self.residual_blocks:
            x = res_block(x)  # [B, 256]
        
        # Second hidden layer: 256 -> 128
        x = self.hidden_layer2(x)  # [B, 128]
        
        # Output projection: 128 -> 31
        output = self.output_proj(x)  # [B, 31]
        
        return output
    
    def get_l2_loss(self) -> torch.Tensor:
        """计算L2正则化损失"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_regularization * l2_loss


class ResidualBlock1D(nn.Module):
    """1D残差块（用于全连接层）"""
    
    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super(ResidualBlock1D, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：x + F(x)"""
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.activation(out)
        return out


class HierarchicalPriorInjectionNet(nn.Module):
    """
    层次化先验注入网络 - 主模型
    
    核心创新：
    1. 浅层（Conv1后）注入GE2 - 引导局部纹理和边缘特征学习
    2. 深层（Conv3后）注入GW - 引导全局颜色分布学习
    3. 基于V3轻量化架构 - 总参数量 ~0.9M
    
    研究问题：
    Q1: 层次化注入是否优于端到端基线（V3）？
    Q2: 局部先验（GE2）是否应该注入浅层？
    Q3: 全局先验（GW）是否应该注入深层？
    Q4: 层次化注入是否优于单一注入？
    """
    
    def __init__(self,
                 input_channels: int = 31,
                 output_dim: int = 31,
                 spectral_extractor_config: dict = None,
                 resnet_regressor_config: dict = None):
        """
        初始化层次化先验注入网络
        
        Args:
            input_channels: 输入通道数（31）
            output_dim: 输出维度（31）
            spectral_extractor_config: 光谱提取器配置
            resnet_regressor_config: ResNet回归头配置
        """
        super(HierarchicalPriorInjectionNet, self).__init__()
        
        # 默认配置（基于V3）
        if spectral_extractor_config is None:
            spectral_extractor_config = {
                'hidden_channels': [64, 128, 256],
                'dropout_rate': 0.3,
                'inject_ge2_after_conv1': True,
                'inject_gw_after_conv3': True
            }
        
        if resnet_regressor_config is None:
            resnet_regressor_config = {
                'input_dim': 256,
                'hidden_dims': [128, 256, 128],
                'num_residual_blocks': 3,
                'dropout_rate': 0.3,
                'l2_regularization': 5e-5
            }
        
        # 光谱特征提取器（支持层次化注入）
        self.spectral_extractor = LightweightSpectralExtractor(
            input_channels=input_channels,
            **spectral_extractor_config
        )
        
        # ResNet回归头
        self.resnet_regressor = LightweightResNetRegressor(
            output_dim=output_dim,
            **resnet_regressor_config
        )
        
        logging.info("HierarchicalPriorInjectionNet initialized")
        logging.info(f"  - Spectral extractor: {spectral_extractor_config}")
        logging.info(f"  - ResNet regressor: {resnet_regressor_config}")
    
    def forward(self, 
                image: torch.Tensor,
                ge2_prior: Optional[torch.Tensor] = None,
                gw_prior: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播：层次化先验注入
        
        Args:
            image: 多光谱图像 [B, 31, H, W]
            ge2_prior: GE2先验（局部特征）[B, 31]
            gw_prior: GW先验（全局特征）[B, 31]
        
        Returns:
            results: 包含预测和注入信息的字典
        """
        # 特征提取 + 层次化注入
        features, injection_info = self.spectral_extractor(
            image, 
            ge2_prior=ge2_prior,
            gw_prior=gw_prior
        )  # [B, 256]
        
        # 回归预测
        prediction = self.resnet_regressor(features)  # [B, 31]
        
        # 返回结果
        results = {
            'illumination_pred': prediction,
            'features': features,
            'injection_info': injection_info
        }
        
        return results
    
    def get_model_info(self) -> Dict[str, int]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        extractor_params = sum(p.numel() for p in self.spectral_extractor.parameters())
        regressor_params = sum(p.numel() for p in self.resnet_regressor.parameters())
        
        # 注入模块参数
        injection_params = 0
        if hasattr(self.spectral_extractor, 'ge2_injection'):
            injection_params += sum(p.numel() for p in self.spectral_extractor.ge2_injection.parameters())
        if hasattr(self.spectral_extractor, 'gw_injection'):
            injection_params += sum(p.numel() for p in self.spectral_extractor.gw_injection.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'spectral_extractor_params': extractor_params,
            'resnet_regressor_params': regressor_params,
            'injection_module_params': injection_params,
            'injection_overhead': injection_params / total_params * 100  # 百分比
        }
    
    def get_l2_regularization_loss(self) -> torch.Tensor:
        """计算L2正则化损失"""
        return self.resnet_regressor.get_l2_loss()


def create_hierarchical_prior_injection_model(config: dict) -> HierarchicalPriorInjectionNet:
    """
    工厂函数：从配置创建层次化先验注入模型
    
    Args:
        config: 配置字典
    
    Returns:
        model: 层次化先验注入网络
    """
    model_config = config.get('model', {})
    
    # 光谱提取器配置
    spectral_config = model_config.get('spectral_extractor', {})
    
    # ResNet回归头配置
    resnet_config = model_config.get('resnet_regressor', {})
    
    model = HierarchicalPriorInjectionNet(
        input_channels=model_config.get('input_channels', 31),
        output_dim=model_config.get('output_dim', 31),
        spectral_extractor_config=spectral_config,
        resnet_regressor_config=resnet_config
    )
    
    return model


if __name__ == "__main__":
    """测试模型"""
    print("=" * 80)
    print("Testing Hierarchical Prior Injection Network")
    print("=" * 80)
    
    # 创建模型
    model = HierarchicalPriorInjectionNet()
    
    # 打印模型信息
    model_info = model.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        if 'overhead' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:,}")
    
    # 测试前向传播
    batch_size = 4
    image = torch.randn(batch_size, 31, 132, 176)
    ge2_prior = torch.randn(batch_size, 31)
    gw_prior = torch.randn(batch_size, 31)
    
    print("\nTesting forward pass:")
    print(f"  Input shape: {image.shape}")
    print(f"  GE2 prior shape: {ge2_prior.shape}")
    print(f"  GW prior shape: {gw_prior.shape}")
    
    results = model(image, ge2_prior=ge2_prior, gw_prior=gw_prior)
    
    print("\nOutput:")
    print(f"  Prediction shape: {results['illumination_pred'].shape}")
    print(f"  Features shape: {results['features'].shape}")
    print(f"  Injection info: {results['injection_info']}")
    
    print("\n✅ All tests passed!")
    print("=" * 80)

