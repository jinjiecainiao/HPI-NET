"""
多光谱数据集加载模块 - 直接处理原始多光谱图像
不使用白点预处理，让ResNet直接学习多光谱特征
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
from .scene_split_utils import split_by_scene, analyze_scene_split
import torchvision.transforms as transforms
import random


class MultispectralDataset(Dataset):
    """
    多光谱数据集类 - 直接处理原始多光谱图像
    不进行白点特征提取，保留完整的空间-光谱信息
    """
    
    def __init__(self,
                 data_dir: str,
                 csf_path: str,
                 mode: str = 'train',
                 train_split_ratio: float = 0.85,
                 random_seed: int = 42,
                 target_height: int = 132,
                 max_width: int = 400,
                 normalize_input: bool = True,
                 use_augmentation: bool = False,
                 augmentation_config: Dict = None,
                 preprocessing_strategy: str = "progressive",
                 use_scene_split: bool = False):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            csf_path: 相机响应函数文件路径
            mode: 'train', 'val', 或 'test'
            train_split_ratio: 训练集比例
            random_seed: 随机种子
            target_height: 目标图像高度
            normalize_input: 是否归一化输入
            use_augmentation: 是否使用数据增强
            augmentation_config: 数据增强配置
        """
        self.data_dir = Path(data_dir)
        self.csf_path = csf_path
        self.mode = mode
        self.random_seed = random_seed
        self.target_height = target_height
        self.max_width = max_width
        self.normalize_input = normalize_input
        self.use_augmentation = use_augmentation and mode == 'train'
        self.augmentation_config = augmentation_config or {}
        self.preprocessing_strategy = preprocessing_strategy
        self.use_scene_split = use_scene_split
        
        # 加载相机响应函数
        self.csf = self._load_csf()
        
        # 获取文件列表并划分数据集
        self.file_paths = self._get_file_paths()
        
        # 根据模式划分数据集
        if 'training' in str(data_dir):
            # 训练数据：从训练数据中进一步划分训练/验证
            if mode in ['train', 'val']:
                if use_scene_split:
                    # 按场景划分，避免数据泄露
                    logging.info(f"Using scene-based split to avoid data leakage")
                    train_files, val_files = split_by_scene(
                        self.file_paths,
                        train_ratio=train_split_ratio,
                        random_seed=random_seed
                    )
                    # 分析并记录划分统计
                    stats = analyze_scene_split(train_files, val_files)
                    logging.info(f"Scene split stats:")
                    logging.info(f"  Train: {stats['num_train_files']} files, {stats['num_train_scenes']} scenes")
                    logging.info(f"  Val: {stats['num_val_files']} files, {stats['num_val_scenes']} scenes")
                    if stats['has_overlap']:
                        logging.warning(f"⚠️ Data leakage detected! Overlap scenes: {stats['overlap_scenes']}")
                    else:
                        logging.info(f"✅ No data leakage - scenes are properly separated")
                else:
                    # 随机划分（原始方法，可能导致数据泄露）
                    logging.warning(f"Using random split - may cause data leakage if same scene appears in train and val")
                    train_files, val_files = train_test_split(
                        self.file_paths, 
                        train_size=train_split_ratio,
                        random_state=random_seed
                    )
                
                self.file_paths = train_files if mode == 'train' else val_files
        # 测试数据：直接使用测试图像
        
        logging.info(f"Loaded {len(self.file_paths)} samples for {mode} mode")
        
        # 设置数据增强
        if self.use_augmentation:
            self._setup_augmentation()
    
    def _load_csf(self) -> np.ndarray:
        """加载相机响应函数"""
        try:
            csf_data = sio.loadmat(self.csf_path)
            
            # 尝试不同的键名
            for key in ['CRF', 'csf', 'sensitivity', 'camera_sensitivity']:
                if key in csf_data:
                    csf = np.array(csf_data[key], dtype=np.float32)
                    if csf.shape == (3, 33):
                        return csf[:, :31].T  # 截取前31列并转置
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
    
    def _setup_augmentation(self):
        """设置数据增强"""
        aug_config = self.augmentation_config
        
        # 空间增强
        self.random_crop = aug_config.get('random_crop', False)
        self.crop_size = aug_config.get('crop_size', [120, 160])
        self.horizontal_flip = aug_config.get('horizontal_flip', False)
        self.vertical_flip = aug_config.get('vertical_flip', False)
        
        # 光谱增强
        self.spectral_noise_std = aug_config.get('spectral_noise_std', 0.01)
        self.spectral_shift_range = aug_config.get('spectral_shift_range', 0.02)
        
        # 照度增强
        self.illuminant_variation = aug_config.get('illuminant_variation', False)
        self.illuminant_noise_std = aug_config.get('illuminant_noise_std', 0.05)
    
    def _resize_multispectral(self, ms_data: np.ndarray) -> np.ndarray:
        """智能多光谱图像尺寸调整 - 深度学习优化版本"""
        H_orig, W_orig, C = ms_data.shape
        
        # 根据配置选择预处理策略
        if self.preprocessing_strategy == "progressive":
            return self._progressive_downsample(ms_data)
        elif self.preprocessing_strategy == "two_stage":
            return self._two_stage_downsample(ms_data)
        else:  # "standard"
            return self._standard_resize(ms_data)
    
    def _progressive_downsample(self, ms_data: np.ndarray) -> np.ndarray:
        """
        智能渐进式下采样 - 借鉴WP成功经验，提取代表性特征
        从1912×W×31巨大信息中提取关键光照信息
        """
        from scipy import ndimage
        H_orig, W_orig, C = ms_data.shape
        
        # 第一步：提取空间代表性区域 (借鉴WP思想)
        # WP成功的原因：找到最亮区域 = 光源信息最丰富的区域
        ms_data = self._extract_informative_regions(ms_data)
        
        # 第二步：多阶段下采样保留细节
        stages = []
        current_h = ms_data.shape[0]  # 更新后的高度
        target_h = self.target_height  # 132
        
        # 计算下采样阶段
        while current_h > target_h * 2:
            current_h = current_h // 2
            stages.append(current_h)
        stages.append(target_h)
        
        # 逐阶段下采样
        current_data = ms_data
        for stage_h in stages:
            stage_w = int(current_data.shape[1] * (stage_h / current_data.shape[0]))
            stage_w = min(stage_w, self.max_width)  # 限制最大宽度
            
            # 对每个通道进行高质量resize
            resized_channels = []
            for c in range(C):
                channel_data = current_data[:, :, c]
                # 使用双三次插值获得更好的质量
                resized_channel = ndimage.zoom(
                    channel_data,
                    (stage_h / current_data.shape[0], stage_w / current_data.shape[1]),
                    order=3,  # 双三次插值
                    prefilter=True  # 预滤波减少混叠
                )
                resized_channels.append(resized_channel)
            
            current_data = np.stack(resized_channels, axis=2)
        
        return current_data
    
    def _extract_informative_regions(self, ms_data: np.ndarray) -> np.ndarray:
        """
        提取信息丰富的区域 - 借鉴WP算法的成功思路
        WP成功的核心：找到最亮区域 = 光源信息最集中的区域
        """
        H, W, C = ms_data.shape
        
        # 策略1：基于亮度的区域提取 (类似WP但保留空间信息)
        # 计算每个像素的总亮度
        brightness = np.sum(ms_data, axis=2)  # [H, W]
        
        # 找到亮度前20%的区域 (比WP的单点max更丰富)
        brightness_threshold = np.percentile(brightness, 80)
        bright_mask = brightness >= brightness_threshold
        
        # 策略2：保留高对比度区域 (光照变化明显的区域)
        # 计算每个像素的光谱变化
        spectral_variance = np.var(ms_data, axis=2)  # [H, W]
        variance_threshold = np.percentile(spectral_variance, 75)
        variance_mask = spectral_variance >= variance_threshold
        
        # 策略3：结合亮度和对比度信息
        informative_mask = bright_mask | variance_mask
        
        # 如果信息区域太少，放宽条件
        if np.sum(informative_mask) < H * W * 0.3:
            brightness_threshold = np.percentile(brightness, 70)
            variance_threshold = np.percentile(spectral_variance, 60)
            bright_mask = brightness >= brightness_threshold
            variance_mask = spectral_variance >= variance_threshold
            informative_mask = bright_mask | variance_mask
        
        # 提取信息丰富的行
        informative_rows = np.any(informative_mask, axis=1)
        if np.sum(informative_rows) < H * 0.4:  # 至少保留40%的行
            # 如果信息行太少，保留亮度最高的行
            row_brightness = np.mean(brightness, axis=1)
            top_rows = np.argsort(row_brightness)[-int(H * 0.4):]
            informative_rows = np.zeros(H, dtype=bool)
            informative_rows[top_rows] = True
        
        # 提取选中的行，大幅减少数据量
        selected_ms_data = ms_data[informative_rows, :, :]
        
        # 进一步优化：如果还是太大，再次压缩
        new_H = selected_ms_data.shape[0]
        if new_H > 800:  # 如果还是太大
            # 均匀采样到合理大小
            step = new_H // 600
            selected_ms_data = selected_ms_data[::step, :, :]
        
        return selected_ms_data
    
    def _two_stage_downsample(self, ms_data: np.ndarray) -> np.ndarray:
        """双阶段下采样 - 平衡质量和效率"""
        from scipy import ndimage
        H_orig, W_orig, C = ms_data.shape
        
        # 第一阶段: 下采样到中等尺寸
        mid_h = max(H_orig // 3, self.target_height * 2)
        mid_w = int(W_orig * (mid_h / H_orig))
        mid_w = min(mid_w, self.max_width)
        
        # 第二阶段: 下采样到目标尺寸
        target_h = self.target_height
        target_w = int(mid_w * (target_h / mid_h))
        
        resized_channels = []
        for c in range(C):
            channel_data = ms_data[:, :, c]
            
            # 第一阶段: 使用高斯滤波+下采样
            mid_channel = ndimage.gaussian_filter(channel_data, sigma=1.0)
            mid_channel = ndimage.zoom(mid_channel, (mid_h / H_orig, mid_w / W_orig), order=1)
            
            # 第二阶段: 精细调整到目标尺寸
            final_channel = ndimage.zoom(mid_channel, (target_h / mid_h, target_w / mid_w), order=3)
            
            resized_channels.append(final_channel)
        
        return np.stack(resized_channels, axis=2)
    
    def _standard_resize(self, ms_data: np.ndarray) -> np.ndarray:
        """标准resize - 用于小图像"""
        from scipy import ndimage
        H_orig, W_orig, C = ms_data.shape
        
        target_h = self.target_height
        calculated_w = int(W_orig * (target_h / H_orig))
        target_w = min(calculated_w, self.max_width)
        
        resized_channels = []
        for c in range(C):
            channel_data = ms_data[:, :, c]
            resized_channel = ndimage.zoom(
                channel_data,
                (target_h / H_orig, target_w / W_orig),
                order=1  # 双线性插值
            )
            resized_channels.append(resized_channel)
        
        return np.stack(resized_channels, axis=2)
    
    def _resize_to_target(self, ms_data: np.ndarray, target_size: list) -> np.ndarray:
        """调整图像到目标尺寸"""
        H_orig, W_orig, C = ms_data.shape
        target_h, target_w = target_size
        
        # 对每个通道分别进行resize
        resized_channels = []
        for c in range(C):
            from scipy import ndimage
            channel_data = ms_data[:, :, c]
            resized_channel = ndimage.zoom(
                channel_data, 
                (target_h / H_orig, target_w / W_orig), 
                order=1  # 双线性插值
            )
            resized_channels.append(resized_channel)
        
        return np.stack(resized_channels, axis=2)
    
    def _apply_spatial_augmentation(self, ms_data: np.ndarray) -> np.ndarray:
        """应用空间数据增强"""
        H, W, C = ms_data.shape
        
        # 随机裁剪 - 确保裁剪后尺寸一致
        if self.random_crop and H > self.crop_size[0] and W > self.crop_size[1]:
            top = random.randint(0, H - self.crop_size[0])
            left = random.randint(0, W - self.crop_size[1])
            ms_data = ms_data[top:top+self.crop_size[0], left:left+self.crop_size[1], :]
        elif self.random_crop:
            # 如果图像小于裁剪尺寸，直接resize到裁剪尺寸
            ms_data = self._resize_to_target(ms_data, self.crop_size)
        
        # 水平翻转
        if self.horizontal_flip and random.random() > 0.5:
            ms_data = np.flip(ms_data, axis=1)
        
        # 垂直翻转
        if self.vertical_flip and random.random() > 0.5:
            ms_data = np.flip(ms_data, axis=0)
        
        return ms_data.copy()  # 确保内存连续
    
    def _apply_spectral_augmentation(self, ms_data: np.ndarray) -> np.ndarray:
        """应用光谱数据增强"""
        # 光谱噪声
        if self.spectral_noise_std > 0:
            noise = np.random.normal(0, self.spectral_noise_std, ms_data.shape)
            ms_data = ms_data + noise
        
        # 光谱偏移
        if self.spectral_shift_range > 0:
            shift = np.random.uniform(-self.spectral_shift_range, 
                                    self.spectral_shift_range, 
                                    ms_data.shape[-1])
            ms_data = ms_data * (1 + shift)
        
        return ms_data
    
    def _apply_illuminant_augmentation(self, illumination: np.ndarray) -> np.ndarray:
        """应用照度数据增强"""
        if self.illuminant_variation:
            # 添加照度噪声
            if self.illuminant_noise_std > 0:
                noise = np.random.normal(0, self.illuminant_noise_std, illumination.shape)
                illumination = illumination * (1 + noise)
        
        return illumination
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            包含以下键的字典:
            - 'multispectral': 多光谱数据 [31, H, W]
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
            
            # 调整图像尺寸
            ms_data = self._resize_multispectral(ms_data)
            
            # 数据清理：确保有限值
            ms_data = np.nan_to_num(ms_data, nan=0.0, posinf=1.0, neginf=0.0)
            illumination_gt = np.nan_to_num(illumination_gt, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 确保非负值
            ms_data = np.clip(ms_data, 0, None)
            illumination_gt = np.clip(illumination_gt, 0, None)
            
            # 数据增强
            if self.use_augmentation:
                ms_data = self._apply_spatial_augmentation(ms_data)
                ms_data = self._apply_spectral_augmentation(ms_data)
                illumination_gt = self._apply_illuminant_augmentation(illumination_gt)
            
            # 确保尺寸一致 - 如果没有进行随机裁剪或裁剪后尺寸不对，强制resize
            current_h, current_w = ms_data.shape[:2]
            if self.use_augmentation and hasattr(self, 'crop_size'):
                target_h, target_w = self.crop_size
                if current_h != target_h or current_w != target_w:
                    ms_data = self._resize_to_target(ms_data, self.crop_size)
            else:
                # 非增强模式：统一resize到目标高度，保持宽高比
                if current_h != self.target_height:
                    target_width = int(current_w * (self.target_height / current_h))
                    ms_data = self._resize_to_target(ms_data, [self.target_height, target_width])
            
            # 输入归一化
            if self.normalize_input:
                # 按通道归一化
                for c in range(ms_data.shape[2]):
                    channel_data = ms_data[:, :, c]
                    if channel_data.max() > channel_data.min():
                        ms_data[:, :, c] = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
            
            # 转换为torch张量并调整维度顺序 [H, W, C] -> [C, H, W]
            ms_tensor = torch.from_numpy(ms_data).float().permute(2, 0, 1)
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
                'multispectral': torch.zeros(31, self.target_height, 
                                           int(self.target_height * 1.33)),  # 4:3 比例
                'illumination_gt': torch.ones(31) / np.sqrt(31),  # 归一化的均匀光照
                'filename': file_path.name
            }
    
    def get_csf(self) -> torch.Tensor:
        """获取相机响应函数张量"""
        return torch.from_numpy(self.csf).float()


def multispectral_collate_fn(batch):
    """
    自定义collate函数，处理不同尺寸的多光谱图像
    使用边缘复制padding，更适合光谱数据（避免引入假的0边界）
    """
    import torch.nn.functional as F
    
    # 获取批次中的最大尺寸
    max_h = max([item['multispectral'].shape[1] for item in batch])
    max_w = max([item['multispectral'].shape[2] for item in batch])
    
    padded_images = []
    illumination_gts = []
    filenames = []
    
    for item in batch:
        ms_tensor = item['multispectral']  # [C, H, W]
        current_h, current_w = ms_tensor.shape[1], ms_tensor.shape[2]
        
        # 计算需要padding的量
        pad_bottom = max_h - current_h
        pad_right = max_w - current_w
        
        if pad_bottom > 0 or pad_right > 0:
            # 🔧 改进：使用replicate padding代替zero padding
            # mode='replicate': 复制边缘像素值，对光谱数据更自然
            # padding顺序: (left, right, top, bottom)
            padded = F.pad(ms_tensor, 
                          (0, pad_right, 0, pad_bottom), 
                          mode='replicate')
        else:
            padded = ms_tensor
        
        padded_images.append(padded)
        illumination_gts.append(item['illumination_gt'])
        filenames.append(item['filename'])
    
    return {
        'multispectral': torch.stack(padded_images),
        'illumination_gt': torch.stack(illumination_gts),
        'filename': filenames
    }


def create_multispectral_dataloaders(train_dir: str,
                                   test_dir: str,
                                   csf_path: str,
                                   config: Dict,
                                   batch_size: int = 6,
                                   train_split_ratio: float = 0.85,
                                   num_workers: int = 0,
                                   random_seed: int = 42,
                                   persistent_workers: bool = False,
                                   prefetch_factor: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建多光谱数据加载器
    
    Args:
        train_dir: 训练数据目录
        test_dir: 测试数据目录
        csf_path: CSF文件路径
        config: 配置字典
        batch_size: 批次大小
        train_split_ratio: 训练验证划分比例
        num_workers: 数据加载工作进程数
        random_seed: 随机种子
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    
    # 获取配置
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    target_height = data_config.get('target_height', 132)
    max_width = data_config.get('max_width', 400)
    normalize_input = data_config.get('normalize_input', True)
    preprocessing_strategy = data_config.get('preprocessing_strategy', 'progressive')
    use_augmentation = training_config.get('use_augmentation', False)
    augmentation_config = training_config.get('augmentation', {})
    
    # 创建数据集
    train_dataset = MultispectralDataset(
        train_dir, csf_path, mode='train', 
        train_split_ratio=train_split_ratio, 
        random_seed=random_seed,
        target_height=target_height,
        max_width=max_width,
        normalize_input=normalize_input,
        use_augmentation=use_augmentation,
        augmentation_config=augmentation_config,
        preprocessing_strategy=preprocessing_strategy
    )
    
    val_dataset = MultispectralDataset(
        train_dir, csf_path, mode='val',
        train_split_ratio=train_split_ratio, 
        random_seed=random_seed,
        target_height=target_height,
        max_width=max_width,
        normalize_input=normalize_input,
        use_augmentation=False,  # 验证时不使用数据增强
        preprocessing_strategy=preprocessing_strategy
    )
    
    test_dataset = MultispectralDataset(
        test_dir, csf_path, mode='test',
        random_seed=random_seed,
        target_height=target_height,
        max_width=max_width,
        normalize_input=normalize_input,
        use_augmentation=False,  # 测试时不使用数据增强
        preprocessing_strategy=preprocessing_strategy
    )
    
    # 创建数据加载器，使用自定义collate函数
    # persistent_workers和prefetch_factor仅在num_workers > 0时生效
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': multispectral_collate_fn
    }
    
    # 仅在有workers时添加persistent_workers和prefetch_factor
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = persistent_workers
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        drop_last=True,  # 训练时丢弃最后一个不完整的batch
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
    # 测试数据集加载
    logging.basicConfig(level=logging.INFO)
    
    train_dir = "../data/dataset/training/mat_norm"
    test_dir = "../data/dataset/testing/mat_norm"
    csf_path = "../data/Canon_1D_Mark_III.mat"
    
    config = {
        'data': {
            'target_height': 132,
            'normalize_input': True
        },
        'training': {
            'use_augmentation': True,
            'augmentation': {
                'random_crop': True,
                'crop_size': [120, 160],
                'horizontal_flip': True,
                'spectral_noise_std': 0.01
            }
        }
    }
    
    try:
        train_loader, val_loader, test_loader = create_multispectral_dataloaders(
            train_dir, test_dir, csf_path, config, batch_size=4
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # 测试加载一个批次
        for batch in train_loader:
            print(f"Multispectral shape: {batch['multispectral'].shape}")
            print(f"Illumination GT shape: {batch['illumination_gt'].shape}")
            print(f"Value ranges: MS [{batch['multispectral'].min():.3f}, {batch['multispectral'].max():.3f}]")
            break
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
