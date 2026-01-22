"""
1D-ResNet回归模型
专门用于多光谱光照估计的残差网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging


class ResidualBlock1D(nn.Module):
    """
    1D残差块
    
    结构: Input -> Dense(hidden_dim, relu) -> BN -> Dropout -> Dense(output_dim) -> BN -> Add(input) -> ReLU
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int = None,
                 dropout_rate: float = 0.2,
                 activation: str = 'relu'):
        """
        初始化残差块
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度，默认与输入维度相同
            dropout_rate: Dropout比率
            activation: 激活函数类型
        """
        super(ResidualBlock1D, self).__init__()
        
        if output_dim is None:
            output_dim = input_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 主路径
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        # 跳跃连接（如果输入输出维度不同）
        self.shortcut = None
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)  # SiLU is the same as Swish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x
        
        # 主路径
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        # 跳跃连接
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        # 残差连接
        out += identity
        out = self.activation(out)
        
        return out


class ResNet1D(nn.Module):
    """
    1D-ResNet回归模型
    
    专门设计用于多光谱光照估计任务：
    - 输入: 31维白点特征向量
    - 输出: 31维优化后的光照估计
    """
    
    def __init__(self,
                 input_dim: int = 31,
                 hidden_dims: List[int] = [128, 128],
                 num_residual_blocks: int = 3,
                 dropout_rate: float = 0.2,
                 l2_regularization: float = 1e-5,
                 activation: str = 'relu',
                 output_activation: Optional[str] = None):
        """
        初始化ResNet1D模型
        
        Args:
            input_dim: 输入维度（31）
            hidden_dims: 每个残差块的隐藏层维度列表
            num_residual_blocks: 残差块数量
            dropout_rate: Dropout比率
            l2_regularization: L2正则化强度
            activation: 激活函数类型
            output_activation: 输出层激活函数（可选）
        """
        super(ResNet1D, self).__init__()
        
        self.input_dim = input_dim
        self.num_residual_blocks = num_residual_blocks
        self.l2_regularization = l2_regularization
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        
        for i in range(num_residual_blocks):
            # 选择隐藏层维度（循环使用hidden_dims）
            hidden_dim = hidden_dims[i % len(hidden_dims)]
            
            # 最后一个残差块输出回到input_dim
            output_dim = input_dim if i == num_residual_blocks - 1 else current_dim
            
            block = ResidualBlock1D(
                input_dim=current_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout_rate=dropout_rate,
                activation=activation
            )
            
            self.residual_blocks.append(block)
            current_dim = output_dim
        
        # 输出层（如果需要额外的变换）
        self.output_layer = None
        if current_dim != input_dim:
            self.output_layer = nn.Linear(current_dim, input_dim)
        
        # 输出激活函数
        self.output_activation = None
        if output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)
        elif output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, 31]
        
        Returns:
            优化后的光照估计 [B, 31]
        """
        # 输入层
        out = self.input_layer(x)
        
        # 残差块
        for block in self.residual_blocks:
            out = block(out)
        
        # 输出层
        if self.output_layer is not None:
            out = self.output_layer(out)
        
        # 输出激活函数
        if self.output_activation is not None:
            out = self.output_activation(out)
        
        return out
    
    def get_l2_regularization_loss(self) -> torch.Tensor:
        """计算L2正则化损失"""
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.norm(param, p=2) ** 2
        
        return self.l2_regularization * l2_loss
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResNet1D',
            'input_dim': self.input_dim,
            'num_residual_blocks': self.num_residual_blocks,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'l2_regularization': self.l2_regularization
        }


class ResNet1DEnsemble(nn.Module):
    """
    ResNet1D集成模型
    
    使用多个ResNet1D模型进行集成预测，提高鲁棒性
    """
    
    def __init__(self,
                 num_models: int = 3,
                 model_configs: List[dict] = None,
                 ensemble_method: str = 'mean'):
        """
        初始化集成模型
        
        Args:
            num_models: 集成模型数量
            model_configs: 每个模型的配置列表
            ensemble_method: 集成方法 ('mean', 'weighted_mean')
        """
        super(ResNet1DEnsemble, self).__init__()
        
        self.num_models = num_models
        self.ensemble_method = ensemble_method
        
        # 创建子模型
        self.models = nn.ModuleList()
        
        if model_configs is None:
            # 使用默认配置创建多样化的模型
            default_configs = [
                {'hidden_dims': [128, 128], 'num_residual_blocks': 2, 'dropout_rate': 0.1},
                {'hidden_dims': [256, 128], 'num_residual_blocks': 3, 'dropout_rate': 0.2},
                {'hidden_dims': [128, 256], 'num_residual_blocks': 4, 'dropout_rate': 0.3}
            ]
            model_configs = default_configs[:num_models]
        
        for i, config in enumerate(model_configs):
            model = ResNet1D(**config)
            self.models.append(model)
        
        # 权重（用于加权平均）
        if ensemble_method == 'weighted_mean':
            self.weights = nn.Parameter(torch.ones(num_models))
        else:
            self.register_buffer('weights', torch.ones(num_models) / num_models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """集成前向传播"""
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # 堆叠预测结果 [num_models, B, 31]
        predictions = torch.stack(predictions, dim=0)
        
        if self.ensemble_method == 'mean':
            # 简单平均
            output = torch.mean(predictions, dim=0)
        elif self.ensemble_method == 'weighted_mean':
            # 加权平均
            weights = F.softmax(self.weights, dim=0)
            weights = weights.view(-1, 1, 1)  # [num_models, 1, 1]
            output = torch.sum(predictions * weights, dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return output
    
    def get_individual_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """获取各个模型的单独预测"""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        return torch.stack(predictions, dim=0)  # [num_models, B, 31]


def create_model(config: dict) -> ResNet1D:
    """
    根据配置创建模型
    
    Args:
        config: 模型配置字典
    
    Returns:
        ResNet1D模型实例
    """
    model_config = config.get('model', {})
    
    return ResNet1D(
        input_dim=model_config.get('input_dim', 31),
        hidden_dims=model_config.get('hidden_dims', [128, 128]),
        num_residual_blocks=model_config.get('num_residual_blocks', 3),
        dropout_rate=model_config.get('dropout_rate', 0.2),
        l2_regularization=model_config.get('l2_regularization', 1e-5),
        activation=model_config.get('activation', 'relu'),
        output_activation=model_config.get('output_activation', None)
    )


if __name__ == "__main__":
    # 测试模型
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = ResNet1D(
        input_dim=31,
        hidden_dims=[128, 256],
        num_residual_blocks=3,
        dropout_rate=0.2
    )
    
    # 打印模型信息
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 8
    input_tensor = torch.randn(batch_size, 31)
    
    print(f"\nInput shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Output shape: {output.shape}")
        
        # 计算L2正则化损失
        l2_loss = model.get_l2_regularization_loss()
        print(f"L2 regularization loss: {l2_loss.item():.6f}")
    
    # 测试集成模型
    print("\nTesting Ensemble Model:")
    ensemble = ResNet1DEnsemble(num_models=3, ensemble_method='weighted_mean')
    
    with torch.no_grad():
        ensemble_output = ensemble(input_tensor)
        individual_preds = ensemble.get_individual_predictions(input_tensor)
        
        print(f"Ensemble output shape: {ensemble_output.shape}")
        print(f"Individual predictions shape: {individual_preds.shape}")
        
        # 检查权重
        if hasattr(ensemble, 'weights'):
            weights = F.softmax(ensemble.weights, dim=0)
            print(f"Ensemble weights: {weights.detach().numpy()}")
    
    print("\nModel test completed successfully!")
