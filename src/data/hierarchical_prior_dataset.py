"""
层次化先验数据集 (Hierarchical Prior Dataset)
支持同时计算GE2和GW先验的数据集类
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import scipy.io as sio

from src.data.prior_algorithms import ClassicalIlluminationEstimator


class HierarchicalPriorDataset(Dataset):
    """
    层次化先验数据集
    
    功能：
    1. 加载多光谱图像和真实光照
    2. 计算GE2先验（局部特征）
    3. 计算GW先验（全局特征）
    4. 支持场景级划分（训练/验证分离）
    5. 支持缓存加速训练
    """
    
    def __init__(self,
                 data_dir: str,
                 csf_path: str,
                 mode: str = 'train',
                 train_split_ratio: float = 0.85,
                 random_seed: int = 42,
                 target_height: int = 132,
                 normalize_input: bool = True,
                 use_augmentation: bool = False,
                 cache_priors: bool = True):
        """
        初始化层次化先验数据集
        
        Args:
            data_dir: 数据目录路径
            csf_path: CSF矩阵路径
            mode: 模式 ('train', 'val', 'test')
            train_split_ratio: 训练集比例
            random_seed: 随机种子
            target_height: 目标高度
            normalize_input: 是否归一化输入
            use_augmentation: 是否使用数据增强
            cache_priors: 是否缓存先验计算结果
        """
        self.data_dir = Path(data_dir)
        self.csf_path = Path(csf_path)
        self.mode = mode
        self.target_height = target_height
        self.normalize_input = normalize_input
        self.use_augmentation = use_augmentation
        self.cache_priors = cache_priors
        
        # 加载CSF矩阵
        self.csf_matrix = self._load_csf_matrix()
        
        # 初始化先验估计器
        self.prior_estimator = ClassicalIlluminationEstimator(epsilon=1e-8)
        
        # 加载数据文件列表
        self.data_files = self._load_data_files(mode, train_split_ratio, random_seed)
        
        # 先验缓存
        self.prior_cache = {} if cache_priors else None
        
        logging.info(f"HierarchicalPriorDataset initialized:")
        logging.info(f"  - Mode: {mode}")
        logging.info(f"  - Samples: {len(self.data_files)}")
        logging.info(f"  - Target height: {target_height}")
        logging.info(f"  - Cache priors: {cache_priors}")
    
    def _load_csf_matrix(self) -> torch.Tensor:
        """加载CSF矩阵"""
        try:
            csf_data = sio.loadmat(str(self.csf_path))
            
            # 查找CSF矩阵
            possible_keys = ['csf', 'CSF', 'camera_sensitivity', 'sensitivity']
            csf_matrix = None
            
            for key in possible_keys:
                if key in csf_data:
                    csf_matrix = csf_data[key]
                    break
            
            if csf_matrix is None:
                data_keys = [k for k in csf_data.keys() if not k.startswith('__')]
                if data_keys:
                    csf_matrix = csf_data[data_keys[0]]
            
            if csf_matrix is None:
                raise ValueError(f"Could not find CSF matrix in {self.csf_path}")
            
            # 转换为torch张量
            csf_tensor = torch.from_numpy(csf_matrix).float()
            
            # 处理不同的CSF矩阵形状
            if csf_tensor.shape == (3, 31):
                csf_tensor = csf_tensor.T
            elif csf_tensor.shape == (31, 3):
                pass
            elif csf_tensor.shape == (3, 33):
                csf_tensor = csf_tensor[:, :31].T
            elif csf_tensor.shape == (33, 3):
                csf_tensor = csf_tensor[:31, :]
            else:
                raise ValueError(f"Unsupported CSF shape: {csf_tensor.shape}")
            
            logging.info(f"CSF matrix loaded: {csf_tensor.shape}")
            return csf_tensor
            
        except Exception as e:
            logging.error(f"Failed to load CSF matrix: {e}")
            raise
    
    def _load_data_files(self, mode: str, train_split_ratio: float, random_seed: int) -> list:
        """加载数据文件列表（支持场景级划分）"""
        # 获取所有.mat文件
        all_files = sorted(list(self.data_dir.glob('*.mat')))
        
        if len(all_files) == 0:
            raise ValueError(f"No .mat files found in {self.data_dir}")
        
        if mode == 'test':
            # 测试集：使用所有文件
            return all_files
        else:
            # 训练/验证集：按场景划分
            # 提取场景名称（去除增强后缀和光照后缀）
            scene_to_files = {}
            for file_path in all_files:
                filename = file_path.stem
                
                # 提取基础场景名（去除所有增强和光照后缀）
                # 例如：
                # - "Daylight_Scene_01_fliphorizontal_aug1_imp" -> "Daylight_Scene_01"
                # - "Daylight_Scene_01_Illum_01" -> "Daylight_Scene_01"
                # - "Daylight_Scene_01" -> "Daylight_Scene_01"
                
                if '_aug' in filename:
                    # 增强样本：去除 "_xxx_augN_imp" 后缀
                    # Daylight_Scene_01_fliphorizontal_aug1_imp -> Daylight_Scene_01
                    scene_name = filename.split('_aug')[0].rsplit('_', 1)[0]
                elif 'Illum' in filename:
                    # 多光照样本：去除 "_Illum_XX" 后缀
                    illum_idx = filename.rfind('_Illum_')
                    scene_name = filename[:illum_idx]
                else:
                    # 原始样本
                    scene_name = filename
                
                if scene_name not in scene_to_files:
                    scene_to_files[scene_name] = []
                scene_to_files[scene_name].append(file_path)
            
            # 按场景划分
            scene_names = sorted(scene_to_files.keys())
            np.random.seed(random_seed)
            np.random.shuffle(scene_names)
            
            split_idx = int(len(scene_names) * train_split_ratio)
            train_scenes = scene_names[:split_idx]
            val_scenes = scene_names[split_idx:]
            
            # 收集对应文件
            if mode == 'train':
                selected_files = []
                for scene in train_scenes:
                    selected_files.extend(scene_to_files[scene])
                logging.info(f"Train: {len(train_scenes)} scenes, {len(selected_files)} samples")
            else:  # mode == 'val'
                selected_files = []
                for scene in val_scenes:
                    selected_files.extend(scene_to_files[scene])
                logging.info(f"Val: {len(val_scenes)} scenes, {len(selected_files)} samples")
            
            return sorted(selected_files)
    
    def _load_mat_file(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载.mat文件
        
        Returns:
            multispectral: [H, W, 31]
            illumination_gt: [31]
        """
        try:
            mat_data = sio.loadmat(str(file_path))
            
            # 查找多光谱数据
            ms_keys = ['multispectral', 'ms_data', 'data', 'image']
            multispectral = None
            for key in ms_keys:
                if key in mat_data:
                    multispectral = mat_data[key]
                    break
            
            if multispectral is None:
                # 尝试第一个非元数据键
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if data_keys:
                    multispectral = mat_data[data_keys[0]]
            
            # 查找真实光照
            illum_keys = ['illumination', 'illuminant', 'gt', 'ground_truth', 'light']
            illumination_gt = None
            for key in illum_keys:
                if key in mat_data:
                    illumination_gt = mat_data[key]
                    break
            
            if multispectral is None or illumination_gt is None:
                raise ValueError(f"Could not find required data in {file_path}")
            
            # 转换为torch张量
            multispectral = torch.from_numpy(multispectral).float()
            illumination_gt = torch.from_numpy(illumination_gt).float().squeeze()
            
            # 确保形状正确
            if multispectral.dim() == 2:
                # [HW, 31] -> [H, W, 31]
                # 需要从文件名或其他方式推断H, W
                raise ValueError(f"Unexpected multispectral shape: {multispectral.shape}")
            elif multispectral.dim() != 3:
                raise ValueError(f"Unexpected multispectral shape: {multispectral.shape}")
            
            # 确保通道数为31
            if multispectral.shape[2] != 31:
                if multispectral.shape[0] == 31:
                    # [31, H, W] -> [H, W, 31]
                    multispectral = multispectral.permute(1, 2, 0)
                else:
                    raise ValueError(f"Expected 31 channels, got {multispectral.shape}")
            
            # 确保光照向量为31维
            if illumination_gt.dim() != 1 or illumination_gt.size(0) != 31:
                raise ValueError(f"Expected illumination shape [31], got {illumination_gt.shape}")
            
            return multispectral, illumination_gt
            
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            raise
    
    def _resize_multispectral(self, ms_data: torch.Tensor) -> torch.Tensor:
        """
        调整多光谱图像大小
        
        Args:
            ms_data: [H, W, 31]
        
        Returns:
            resized: [target_H, W', 31]
        """
        H, W, C = ms_data.shape
        
        if H == self.target_height:
            return ms_data
        
        # 计算缩放比例
        scale = self.target_height / H
        new_width = int(W * scale)
        
        # 转换为 [31, H, W] 格式用于插值
        ms_transposed = ms_data.permute(2, 0, 1).unsqueeze(0)  # [1, 31, H, W]
        
        # 双线性插值
        import torch.nn.functional as F
        resized = F.interpolate(
            ms_transposed,
            size=(self.target_height, new_width),
            mode='bilinear',
            align_corners=False
        )
        
        # 转回 [H', W', 31]
        resized = resized.squeeze(0).permute(1, 2, 0)
        
        return resized
    
    def _compute_ge2_prior(self, ms_data: torch.Tensor, filename: str) -> torch.Tensor:
        """
        计算GE2先验（局部特征）
        
        Args:
            ms_data: [H, W, 31]
            filename: 文件名（用于缓存）
        
        Returns:
            ge2_prior: [31]
        """
        # 检查缓存
        if self.prior_cache is not None:
            cache_key = f"{filename}_ge2"
            if cache_key in self.prior_cache:
                return self.prior_cache[cache_key]
        
        # 计算GE2
        ge2_prior = self.prior_estimator.grey_edge(ms_data, order=2, norm=1)
        
        # 保存到缓存
        if self.prior_cache is not None:
            cache_key = f"{filename}_ge2"
            self.prior_cache[cache_key] = ge2_prior
        
        return ge2_prior
    
    def _compute_gw_prior(self, ms_data: torch.Tensor, filename: str) -> torch.Tensor:
        """
        计算GW先验（全局特征）
        
        Args:
            ms_data: [H, W, 31]
            filename: 文件名（用于缓存）
        
        Returns:
            gw_prior: [31]
        """
        # 检查缓存
        if self.prior_cache is not None:
            cache_key = f"{filename}_gw"
            if cache_key in self.prior_cache:
                return self.prior_cache[cache_key]
        
        # 计算GW
        gw_prior = self.prior_estimator.grey_world(ms_data, norm=1)
        
        # 保存到缓存
        if self.prior_cache is not None:
            cache_key = f"{filename}_gw"
            self.prior_cache[cache_key] = gw_prior
        
        return gw_prior
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            sample: {
                'multispectral': [31, H, W],
                'illumination_gt': [31],
                'ge2_prior': [31],
                'gw_prior': [31],
                'filename': str
            }
        """
        file_path = self.data_files[idx]
        filename = file_path.stem
        
        # 加载数据
        ms_data, illumination_gt = self._load_mat_file(file_path)  # [H, W, 31], [31]
        
        # 调整大小
        ms_data = self._resize_multispectral(ms_data)  # [target_H, W', 31]
        
        # 计算先验
        ge2_prior = self._compute_ge2_prior(ms_data, filename)  # [31]
        gw_prior = self._compute_gw_prior(ms_data, filename)    # [31]
        
        # 归一化输入
        if self.normalize_input:
            ms_data = ms_data / (ms_data.max() + 1e-8)
        
        # 转换为 [31, H, W] 格式（PyTorch标准）
        ms_data = ms_data.permute(2, 0, 1)  # [31, H, W]
        
        # 数据增强（训练时）
        if self.use_augmentation and self.mode == 'train':
            ms_data = self._apply_augmentation(ms_data)
        
        return {
            'multispectral': ms_data,
            'illumination_gt': illumination_gt,
            'ge2_prior': ge2_prior,
            'gw_prior': gw_prior,
            'filename': filename
        }
    
    def _apply_augmentation(self, ms_data: torch.Tensor) -> torch.Tensor:
        """应用数据增强（简单版本）"""
        # 随机水平翻转
        if torch.rand(1).item() > 0.5:
            ms_data = torch.flip(ms_data, dims=[2])
        
        # 随机亮度调整
        brightness_factor = 0.85 + torch.rand(1).item() * 0.3  # [0.85, 1.15]
        ms_data = ms_data * brightness_factor
        
        # 裁剪到[0, 1]
        ms_data = torch.clamp(ms_data, 0.0, 1.0)
        
        return ms_data
    
    def get_csf(self) -> torch.Tensor:
        """
        获取CSF矩阵
        
        Returns:
            csf_matrix: 相机灵敏度函数矩阵 [3, 31]
        """
        # CSF矩阵存储为 [31, 3]，需要转置为 [3, 31]
        if self.csf_matrix.shape == (31, 3):
            return self.csf_matrix.t()  # 转置为 [3, 31]
        elif self.csf_matrix.shape == (3, 31):
            return self.csf_matrix
        else:
            raise ValueError(f"CSF矩阵形状异常: {self.csf_matrix.shape}，应为[31,3]或[3,31]")


def hierarchical_prior_collate_fn(batch):
    """
    层次化先验数据集的collate函数
    处理不同尺寸的图像，使用padding对齐
    """
    import torch.nn.functional as F
    
    # 找到batch中的最大尺寸
    max_height = max(sample['multispectral'].shape[1] for sample in batch)
    max_width = max(sample['multispectral'].shape[2] for sample in batch)
    
    # 初始化列表
    multispectral_list = []
    illumination_gt_list = []
    ge2_prior_list = []
    gw_prior_list = []
    filename_list = []
    
    # 处理每个样本
    for sample in batch:
        ms_data = sample['multispectral']  # [31, H, W]
        
        # 计算需要的padding
        pad_height = max_height - ms_data.shape[1]
        pad_width = max_width - ms_data.shape[2]
        
        # 使用replicate padding（边缘复制）
        if pad_height > 0 or pad_width > 0:
            ms_data = F.pad(
                ms_data,
                (0, pad_width, 0, pad_height),
                mode='replicate'
            )
        
        multispectral_list.append(ms_data)
        illumination_gt_list.append(sample['illumination_gt'])
        ge2_prior_list.append(sample['ge2_prior'])
        gw_prior_list.append(sample['gw_prior'])
        filename_list.append(sample['filename'])
    
    # 堆叠成批次
    batch_dict = {
        'multispectral': torch.stack(multispectral_list),      # [B, 31, max_H, max_W]
        'illumination_gt': torch.stack(illumination_gt_list),  # [B, 31]
        'ge2_prior': torch.stack(ge2_prior_list),              # [B, 31]
        'gw_prior': torch.stack(gw_prior_list),                # [B, 31]
        'filename': filename_list
    }
    
    return batch_dict


if __name__ == "__main__":
    """测试数据集"""
    print("=" * 80)
    print("Testing Hierarchical Prior Dataset")
    print("=" * 80)
    
    # 测试参数
    data_dir = "../data/dataset/training/mat_norm"
    csf_path = "../data/Canon_1D_Mark_III.mat"
    
    # 创建数据集
    dataset = HierarchicalPriorDataset(
        data_dir=data_dir,
        csf_path=csf_path,
        mode='train',
        train_split_ratio=0.85,
        target_height=132,
        cache_priors=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Multispectral shape: {sample['multispectral'].shape}")
    print(f"  Illumination GT shape: {sample['illumination_gt'].shape}")
    print(f"  GE2 prior shape: {sample['ge2_prior'].shape}")
    print(f"  GW prior shape: {sample['gw_prior'].shape}")
    print(f"  Filename: {sample['filename']}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=hierarchical_prior_collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Multispectral shape: {batch['multispectral'].shape}")
    print(f"  Illumination GT shape: {batch['illumination_gt'].shape}")
    print(f"  GE2 prior shape: {batch['ge2_prior'].shape}")
    print(f"  GW prior shape: {batch['gw_prior'].shape}")
    print(f"  Filenames: {len(batch['filename'])}")
    
    print("\n✅ All tests passed!")
    print("=" * 80)

