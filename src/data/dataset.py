"""
NUS多光谱数据集加载模块
实现NUS数据集的加载、预处理和数据划分功能
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict, List
import scipy.io as sio
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

class NUSMultispectralDataset(Dataset):
    """
    NUS多光谱数据集类
    
    加载31通道多光谱.mat文件，提供地面真值光照估计用于训练和评估
    严格遵循原论文的数据划分：训练40张，测试24张
    """
    
    def __init__(self,
                 data_dir: str,
                 csf_path: str,
                 mode: str = 'train',
                 train_split_ratio: float = 0.8,
                 random_seed: int = 42):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            csf_path: 相机响应函数文件路径
            mode: 'train', 'val', 或 'test'
            train_split_ratio: 训练集比例（从训练数据中划分验证集）
            random_seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.csf_path = csf_path
        self.mode = mode
        self.random_seed = random_seed
        
        # 加载相机响应函数
        self.csf = self._load_csf()
        
        # 获取文件列表并划分数据集
        self.file_paths = self._get_file_paths()
        
        # 根据原论文划分：训练40张，测试24张
        if 'training' in str(data_dir):
            # 训练数据：从40张中进一步划分训练/验证
            if mode in ['train', 'val']:
                train_files, val_files = train_test_split(
                    self.file_paths, 
                    train_size=train_split_ratio,
                    random_state=random_seed
                )
                self.file_paths = train_files if mode == 'train' else val_files
        # 测试数据：直接使用24张测试图像
        
        logging.info(f"Loaded {len(self.file_paths)} samples for {mode} mode")
    
    def _load_csf(self) -> np.ndarray:
        """加载相机响应函数"""
        try:
            csf_data = sio.loadmat(self.csf_path)
            
            # 根据实际文件结构，CSF矩阵的键名是'CRF'
            if 'CRF' in csf_data:
                csf = np.array(csf_data['CRF'], dtype=np.float32)
                # 原始形状是(3, 33)，我们需要截取前31列并转置为(31, 3)
                if csf.shape == (3, 33):
                    csf = csf[:, :31].T  # 截取前31列并转置
                    return csf
                elif csf.shape == (31, 3):
                    return csf
                elif csf.shape == (3, 31):
                    return csf.T
            
            # 尝试其他可能的键名
            for key in ['csf', 'sensitivity', 'camera_sensitivity']:
                if key in csf_data:
                    csf = np.array(csf_data[key], dtype=np.float32)
                    if csf.shape == (31, 3):
                        return csf
                    elif csf.shape == (3, 31):
                        return csf.T
                    elif csf.shape[0] == 3 and csf.shape[1] >= 31:
                        return csf[:, :31].T
            
            # 如果找不到标准key，尝试找到合适形状的矩阵
            for key, value in csf_data.items():
                if isinstance(value, np.ndarray) and not key.startswith('__'):
                    if value.shape == (31, 3):
                        return value.astype(np.float32)
                    elif value.shape == (3, 31):
                        return value.T.astype(np.float32)
                    elif value.shape == (3, 33):
                        return value[:, :31].T.astype(np.float32)
            
            raise ValueError(f"Could not find CSF matrix in {self.csf_path}")
            
        except Exception as e:
            logging.error(f"Failed to load CSF from {self.csf_path}: {e}")
            # 使用默认CSF（如果找不到文件）
            return self._create_default_csf()
    
    def _create_default_csf(self) -> np.ndarray:
        """创建默认的相机响应函数（简化版）"""
        logging.warning("Using default CSF matrix")
        # 创建一个简化的CSF矩阵
        csf = np.zeros((31, 3), dtype=np.float32)
        # R通道：主要响应长波长
        csf[20:31, 0] = np.linspace(0.1, 1.0, 11)
        # G通道：主要响应中波长
        csf[10:25, 1] = np.concatenate([np.linspace(0.1, 1.0, 8), np.linspace(1.0, 0.1, 7)])
        # B通道：主要响应短波长
        csf[0:15, 2] = np.linspace(1.0, 0.1, 15)
        return csf
    
    def _get_file_paths(self) -> List[Path]:
        """获取所有.mat文件路径"""
        file_paths = []
        if self.data_dir.is_dir():
            for file_path in self.data_dir.glob("*.mat"):
                file_paths.append(file_path)
        
        file_paths.sort()  # 确保顺序一致
        return file_paths
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            包含以下键的字典:
            - 'multispectral': 多光谱数据 [H, W, 31] (用于白点特征提取)
            - 'illumination_gt': 地面真值光照 [31]
            - 'filename': 文件名
        """
        file_path = self.file_paths[idx]
        
        try:
            # 加载.mat文件
            mat_data = sio.loadmat(str(file_path))
            
            # 提取多光谱数据
            if 'tensor' in mat_data:
                ms_data = np.array(mat_data['tensor'], dtype=np.float32)
            elif 'img' in mat_data:
                ms_data = np.array(mat_data['img'], dtype=np.float32)
            else:
                raise ValueError(f"Could not find multispectral data in {file_path}")
            
            # 提取地面真值光照
            if 'illumination' in mat_data:
                illumination_gt = np.array(mat_data['illumination'], dtype=np.float32)
            elif 'illum' in mat_data:
                illumination_gt = np.array(mat_data['illum'], dtype=np.float32)
            else:
                raise ValueError(f"Could not find illumination data in {file_path}")
            
            # 验证数据形状
            if len(ms_data.shape) != 3 or ms_data.shape[2] != 31:
                raise ValueError(f"Invalid multispectral data shape: {ms_data.shape}")
            
            # 确保illumination_gt是1D数组
            if illumination_gt.ndim > 1:
                illumination_gt = illumination_gt.flatten()
            
            if illumination_gt.shape[0] != 31:
                raise ValueError(f"Invalid illumination shape: {illumination_gt.shape}")
            
            # 数据预处理：遵循原论文策略，缩放到高度132
            H_orig, W_orig = ms_data.shape[:2]
            target_height = 132
            target_width = int(W_orig * (target_height / H_orig))
            
            # 对每个通道分别进行resize
            resized_channels = []
            for c in range(31):
                # 使用双线性插值缩放
                from scipy import ndimage
                channel_data = ms_data[:, :, c]
                resized_channel = ndimage.zoom(
                    channel_data, 
                    (target_height / H_orig, target_width / W_orig), 
                    order=1  # 双线性插值
                )
                resized_channels.append(resized_channel)
            
            ms_data = np.stack(resized_channels, axis=2)  # [132, W_new, 31]
            
            # 数据清理：确保有限值
            ms_data = np.nan_to_num(ms_data, nan=0.0, posinf=1.0, neginf=0.0)
            illumination_gt = np.nan_to_num(illumination_gt, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 确保非负值
            ms_data = np.clip(ms_data, 0, None)
            illumination_gt = np.clip(illumination_gt, 0, None)
            
            # 保持原始数据范围，不进行像素级归一化
            # 这样白点算法提取的特征才能代表真实的光源分布
            
            # 重要：不在这里对光照向量进行归一化！
            # 归一化应该只在损失函数计算Recovery Angular Error时进行
            # 这里保持原始的光源光谱分布，让网络学习真实的物理量
            
            # 转换为torch张量
            ms_tensor = torch.from_numpy(ms_data).float()
            illumination_tensor = torch.from_numpy(illumination_gt).float()
            
            return {
                'multispectral': ms_tensor,
                'illumination_gt': illumination_tensor,
                'filename': file_path.name
            }
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            # 返回零填充的数据以避免训练中断
            return {
                'multispectral': torch.zeros(132, 132, 31),  # 使用固定尺寸
                'illumination_gt': torch.ones(31) / np.sqrt(31),  # 归一化的均匀光照
                'filename': file_path.name
            }
    
    def get_csf(self) -> torch.Tensor:
        """获取相机响应函数张量"""
        return torch.from_numpy(self.csf).float()


def custom_collate_fn(batch):
    """
    自定义collate函数，处理不同尺寸的多光谱图像
    
    由于我们的模型只需要白点特征（31维向量），我们可以在collate阶段
    直接提取白点特征，避免尺寸不匹配的问题
    """
    from .white_point import SpectralWhitePointExtractor
    
    # 初始化白点提取器（不归一化，保持物理意义）
    wp_extractor = SpectralWhitePointExtractor(normalize=False)
    
    # 分离batch中的不同元素
    multispectral_data = []
    illumination_gt = []
    filenames = []
    
    for sample in batch:
        ms_data = sample['multispectral']  # [H, W, 31]
        
        # 直接在collate阶段提取白点特征
        wp_features = wp_extractor.extract_features(ms_data, method='max')  # [31]
        
        multispectral_data.append(wp_features)
        illumination_gt.append(sample['illumination_gt'])
        filenames.append(sample['filename'])
    
    # 堆叠成批次张量
    multispectral_batch = torch.stack(multispectral_data)  # [B, 31]
    illumination_batch = torch.stack(illumination_gt)     # [B, 31]
    
    return {
        'multispectral': multispectral_batch,
        'illumination_gt': illumination_batch,
        'filename': filenames
    }


def create_dataloaders(train_dir: str,
                      test_dir: str,
                      csf_path: str,
                      batch_size: int = 8,
                      train_split_ratio: float = 0.8,
                      num_workers: int = 0,
                      random_seed: int = 42,
                      drop_last: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        train_dir: 训练数据目录
        test_dir: 测试数据目录
        csf_path: CSF文件路径
        batch_size: 批次大小
        train_split_ratio: 训练验证划分比例
        num_workers: 数据加载工作进程数
        random_seed: 随机种子
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # 创建数据集
    train_dataset = NUSMultispectralDataset(
        train_dir, csf_path, mode='train', 
        train_split_ratio=train_split_ratio, random_seed=random_seed
    )
    
    val_dataset = NUSMultispectralDataset(
        train_dir, csf_path, mode='val',
        train_split_ratio=train_split_ratio, random_seed=random_seed
    )
    
    test_dataset = NUSMultispectralDataset(
        test_dir, csf_path, mode='test',
        random_seed=random_seed
    )
    
    # 创建数据加载器，使用自定义collate函数
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn, drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn, drop_last=drop_last
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn, drop_last=False  # 测试时不丢弃样本
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集加载
    logging.basicConfig(level=logging.INFO)
    
    train_dir = "../data/dataset/training/mat_norm"
    test_dir = "../data/dataset/testing/mat_norm"
    csf_path = "../data/Canon_1D_Mark_III.mat"
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dir, test_dir, csf_path, batch_size=4
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # 测试加载一个批次
        for batch in train_loader:
            print(f"Multispectral shape: {batch['multispectral'].shape}")
            print(f"Illumination GT shape: {batch['illumination_gt'].shape}")
            break
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
