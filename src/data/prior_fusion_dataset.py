"""
动态先验融合数据集模块
预计算并缓存多个经典算法的先验特征
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import scipy.io as sio
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import pickle
import hashlib

from .prior_algorithms import ClassicalIlluminationEstimator


class PriorFusionDataset(Dataset):
    """
    动态先验融合数据集
    
    为每个样本预计算并缓存多个经典算法的先验特征：
    - 完整的多光谱图像 [C, H, W] 用于注意力头
    - 预计算的先验特征 [K, 31] 用于先验融合
    - 地面真值光照 [31]
    """
    
    def __init__(self,
                 data_dir: str,
                 csf_path: str,
                 mode: str = 'train',
                 train_split_ratio: float = 0.85,
                 random_seed: int = 42,
                 target_height: int = 132,
                 selected_priors: List[str] = None,
                 cache_dir: str = None,
                 use_cache: bool = True,
                 normalize_priors: bool = False):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            csf_path: 相机响应函数文件路径
            mode: 'train', 'val', 或 'test'
            train_split_ratio: 训练集比例
            random_seed: 随机种子
            target_height: 目标图像高度
            selected_priors: 选择的先验算法列表，默认['WP', 'GW', 'GE1', 'GE2']
            cache_dir: 缓存目录，默认为data_dir/prior_cache
            use_cache: 是否使用缓存
            normalize_priors: 是否归一化先验特征
        """
        self.data_dir = Path(data_dir)
        self.csf_path = csf_path
        self.mode = mode
        self.random_seed = random_seed
        self.target_height = target_height
        self.normalize_priors = normalize_priors
        
        # 默认使用4个经典算法
        if selected_priors is None:
            selected_priors = ['WP', 'GW', 'GE1', 'GE2']
        self.selected_priors = selected_priors
        self.num_priors = len(selected_priors)
        
        # 缓存设置
        self.use_cache = use_cache
        if cache_dir is None:
            self.cache_dir = self.data_dir.parent / 'prior_cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载相机响应函数
        self.csf = self._load_csf()
        
        # 获取文件列表并划分数据集
        self.file_paths = self._get_file_paths()
        
        # 根据模式划分数据集
        if 'training' in str(data_dir):
            if mode in ['train', 'val']:
                train_files, val_files = train_test_split(
                    self.file_paths,
                    train_size=train_split_ratio,
                    random_state=random_seed
                )
                self.file_paths = train_files if mode == 'train' else val_files
        
        logging.info(f"Loaded {len(self.file_paths)} samples for {mode} mode")
        logging.info(f"Selected priors: {self.selected_priors}")
        
        # 预计算或加载先验特征
        self.priors_cache = {}
        if self.use_cache:
            self._load_or_compute_priors()
    
    def _load_csf(self) -> np.ndarray:
        """加载相机响应函数"""
        try:
            csf_data = sio.loadmat(self.csf_path)
            
            # 尝试不同的键名
            for key in ['CRF', 'csf', 'sensitivity', 'camera_sensitivity']:
                if key in csf_data:
                    csf = np.array(csf_data[key], dtype=np.float32)
                    if csf.shape == (3, 33):
                        return csf[:, :31].T
                    elif csf.shape == (3, 31):
                        return csf.T
                    elif csf.shape == (31, 3):
                        return csf
            
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
            return self._create_default_csf()
    
    def _create_default_csf(self) -> np.ndarray:
        """创建默认的相机响应函数"""
        logging.warning("Using default CSF matrix")
        csf = np.zeros((31, 3), dtype=np.float32)
        csf[20:31, 0] = np.linspace(0.1, 1.0, 11)  # R
        csf[10:25, 1] = np.concatenate([np.linspace(0.1, 1.0, 8), np.linspace(1.0, 0.1, 7)])  # G
        csf[0:15, 2] = np.linspace(1.0, 0.1, 15)  # B
        return csf
    
    def _get_file_paths(self) -> List[Path]:
        """获取所有.mat文件路径"""
        file_paths = []
        if self.data_dir.is_dir():
            for file_path in self.data_dir.glob("*.mat"):
                file_paths.append(file_path)
        
        file_paths.sort()  # 确保顺序一致
        return file_paths
    
    def _get_cache_path(self) -> Path:
        """获取缓存文件路径"""
        # 生成缓存文件名（基于数据目录和先验配置的哈希）
        config_str = f"{self.data_dir}_{','.join(self.selected_priors)}_{self.normalize_priors}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_filename = f"priors_{self.mode}_{config_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def _load_or_compute_priors(self):
        """加载或计算先验特征"""
        cache_path = self._get_cache_path()
        
        if cache_path.exists():
            # 从缓存加载
            logging.info(f"Loading priors from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self.priors_cache = pickle.load(f)
                logging.info(f"Successfully loaded {len(self.priors_cache)} cached priors")
                return
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}, recomputing...")
        
        # 计算先验特征
        logging.info(f"Computing priors for {len(self.file_paths)} samples...")
        self.priors_cache = {}
        
        estimator = ClassicalIlluminationEstimator()
        
        for idx, file_path in enumerate(self.file_paths):
            try:
                # 加载多光谱数据
                ms_data = self._load_multispectral_data(file_path)  # [H, W, 31]
                
                # 计算先验
                ms_tensor = torch.from_numpy(ms_data).float()
                priors = estimator.compute_selected_priors(ms_tensor, self.selected_priors)  # [K, 31]
                
                # 可选：归一化
                if self.normalize_priors:
                    for i in range(priors.shape[0]):
                        norm = torch.norm(priors[i])
                        if norm > 1e-8:
                            priors[i] = priors[i] / norm
                
                # 缓存
                self.priors_cache[str(file_path)] = priors.numpy()
                
                if (idx + 1) % 10 == 0:
                    logging.info(f"  Processed {idx + 1}/{len(self.file_paths)} samples")
                    
            except Exception as e:
                logging.error(f"Failed to compute priors for {file_path}: {e}")
                # 使用零填充
                self.priors_cache[str(file_path)] = np.zeros((self.num_priors, 31), dtype=np.float32)
        
        # 保存到缓存
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.priors_cache, f)
            logging.info(f"Saved priors cache to: {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")
    
    def _load_multispectral_data(self, file_path: Path) -> np.ndarray:
        """加载并预处理多光谱数据"""
        mat_data = sio.loadmat(str(file_path))
        
        # 提取多光谱数据
        if 'tensor' in mat_data:
            ms_data = np.array(mat_data['tensor'], dtype=np.float32)
        elif 'img' in mat_data:
            ms_data = np.array(mat_data['img'], dtype=np.float32)
        else:
            raise ValueError(f"Could not find multispectral data in {file_path}")
        
        # 验证形状
        if len(ms_data.shape) != 3 or ms_data.shape[2] != 31:
            raise ValueError(f"Invalid multispectral data shape: {ms_data.shape}")
        
        # 调整尺寸
        H_orig, W_orig = ms_data.shape[:2]
        target_width = int(W_orig * (self.target_height / H_orig))
        
        # 对每个通道进行resize
        from scipy import ndimage
        resized_channels = []
        for c in range(31):
            channel_data = ms_data[:, :, c]
            resized_channel = ndimage.zoom(
                channel_data,
                (self.target_height / H_orig, target_width / W_orig),
                order=1  # 双线性插值
            )
            resized_channels.append(resized_channel)
        
        ms_data = np.stack(resized_channels, axis=2)  # [H, W, 31]
        
        # 数据清理
        ms_data = np.nan_to_num(ms_data, nan=0.0, posinf=1.0, neginf=0.0)
        ms_data = np.clip(ms_data, 0, None)
        
        return ms_data
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            包含以下键的字典:
            - 'multispectral': 多光谱图像 [31, H, W] (用于注意力头)
            - 'priors': 预计算的先验特征 [K, 31] (用于先验融合)
            - 'illumination_gt': 地面真值光照 [31]
            - 'filename': 文件名
        """
        file_path = self.file_paths[idx]
        
        try:
            # 加载.mat文件
            mat_data = sio.loadmat(str(file_path))
            
            # 加载多光谱数据
            ms_data = self._load_multispectral_data(file_path)  # [H, W, 31]
            
            # 提取地面真值光照
            if 'illumination' in mat_data:
                illumination_gt = np.array(mat_data['illumination'], dtype=np.float32)
            elif 'illum' in mat_data:
                illumination_gt = np.array(mat_data['illum'], dtype=np.float32)
            else:
                raise ValueError(f"Could not find illumination data in {file_path}")
            
            # 确保illumination_gt是1D数组
            if illumination_gt.ndim > 1:
                illumination_gt = illumination_gt.flatten()
            
            if illumination_gt.shape[0] != 31:
                raise ValueError(f"Invalid illumination shape: {illumination_gt.shape}")
            
            # 数据清理
            illumination_gt = np.nan_to_num(illumination_gt, nan=0.0, posinf=1.0, neginf=0.0)
            illumination_gt = np.clip(illumination_gt, 0, None)
            
            # 获取预计算的先验特征
            if self.use_cache and str(file_path) in self.priors_cache:
                priors = self.priors_cache[str(file_path)]  # [K, 31]
            else:
                # 实时计算
                estimator = ClassicalIlluminationEstimator()
                ms_tensor = torch.from_numpy(ms_data).float()
                priors_tensor = estimator.compute_selected_priors(ms_tensor, self.selected_priors)
                
                if self.normalize_priors:
                    for i in range(priors_tensor.shape[0]):
                        norm = torch.norm(priors_tensor[i])
                        if norm > 1e-8:
                            priors_tensor[i] = priors_tensor[i] / norm
                
                priors = priors_tensor.numpy()
            
            # 转换为torch张量
            # 多光谱图像：[H, W, 31] -> [31, H, W]
            ms_tensor = torch.from_numpy(ms_data).float().permute(2, 0, 1)
            priors_tensor = torch.from_numpy(priors).float()
            illumination_tensor = torch.from_numpy(illumination_gt).float()
            
            return {
                'multispectral': ms_tensor,      # [31, H, W]
                'priors': priors_tensor,         # [K, 31]
                'illumination_gt': illumination_tensor,  # [31]
                'filename': file_path.name
            }
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            # 返回零填充的数据
            return {
                'multispectral': torch.zeros(31, self.target_height, int(self.target_height * 1.33)),
                'priors': torch.zeros(self.num_priors, 31),
                'illumination_gt': torch.ones(31) / np.sqrt(31),
                'filename': file_path.name
            }
    
    def get_csf(self) -> torch.Tensor:
        """获取相机响应函数张量"""
        return torch.from_numpy(self.csf).float()


def prior_fusion_collate_fn(batch):
    """
    自定义collate函数，处理不同尺寸的多光谱图像
    """
    import torch.nn.functional as F
    
    # 获取批次中的最大尺寸
    max_h = max([item['multispectral'].shape[1] for item in batch])
    max_w = max([item['multispectral'].shape[2] for item in batch])
    
    padded_images = []
    priors_list = []
    illumination_gts = []
    filenames = []
    
    for item in batch:
        ms_tensor = item['multispectral']  # [C, H, W]
        current_h, current_w = ms_tensor.shape[1], ms_tensor.shape[2]
        
        # 计算需要padding的量
        pad_bottom = max_h - current_h
        pad_right = max_w - current_w
        
        if pad_bottom > 0 or pad_right > 0:
            # 使用replicate padding
            padded = F.pad(ms_tensor,
                          (0, pad_right, 0, pad_bottom),
                          mode='replicate')
        else:
            padded = ms_tensor
        
        padded_images.append(padded)
        priors_list.append(item['priors'])
        illumination_gts.append(item['illumination_gt'])
        filenames.append(item['filename'])
    
    return {
        'multispectral': torch.stack(padded_images),     # [B, 31, H, W]
        'priors': torch.stack(priors_list),              # [B, K, 31]
        'illumination_gt': torch.stack(illumination_gts), # [B, 31]
        'filename': filenames
    }


def create_prior_fusion_dataloaders(train_dir: str,
                                   test_dir: str,
                                   csf_path: str,
                                   batch_size: int = 8,
                                   train_split_ratio: float = 0.85,
                                   num_workers: int = 0,
                                   random_seed: int = 42,
                                   selected_priors: List[str] = None,
                                   use_cache: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建先验融合数据加载器
    
    Args:
        train_dir: 训练数据目录
        test_dir: 测试数据目录
        csf_path: CSF文件路径
        batch_size: 批次大小
        train_split_ratio: 训练验证划分比例
        num_workers: 数据加载工作进程数
        random_seed: 随机种子
        selected_priors: 选择的先验算法
        use_cache: 是否使用缓存
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # 创建数据集
    train_dataset = PriorFusionDataset(
        train_dir, csf_path, mode='train',
        train_split_ratio=train_split_ratio,
        random_seed=random_seed,
        selected_priors=selected_priors,
        use_cache=use_cache
    )
    
    val_dataset = PriorFusionDataset(
        train_dir, csf_path, mode='val',
        train_split_ratio=train_split_ratio,
        random_seed=random_seed,
        selected_priors=selected_priors,
        use_cache=use_cache
    )
    
    test_dataset = PriorFusionDataset(
        test_dir, csf_path, mode='test',
        random_seed=random_seed,
        selected_priors=selected_priors,
        use_cache=use_cache
    )
    
    # 创建数据加载器
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': prior_fusion_collate_fn
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    logging.basicConfig(level=logging.INFO)
    
    train_dir = "../data/dataset/training/mat_norm"
    test_dir = "../data/dataset/testing/mat_norm"
    csf_path = "../data/Canon_1D_Mark_III.mat"
    
    try:
        train_loader, val_loader, test_loader = create_prior_fusion_dataloaders(
            train_dir, test_dir, csf_path, batch_size=4
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # 测试加载一个批次
        for batch in train_loader:
            print(f"\nBatch contents:")
            print(f"  Multispectral shape: {batch['multispectral'].shape}")
            print(f"  Priors shape: {batch['priors'].shape}")
            print(f"  Illumination GT shape: {batch['illumination_gt'].shape}")
            print(f"  Filenames: {batch['filename']}")
            break
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

