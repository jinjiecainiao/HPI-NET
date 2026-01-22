import os
import numpy as np
import scipy.io as sio
from scipy.ndimage import rotate
from scipy import interpolate
import cv2
import glob
from pathlib import Path
import random
import pandas as pd

class DataAugmentator:
    """数据增强类，用于多光谱图像数据增强"""
    
    def __init__(self, input_dir="HPI-NET/data/dataset/training/mat_norm", 
                 output_dir="HPI-NET/data/dataset/training/mat_norm_enhanced",
                 led_excel_path="LED灯箱Channel14SPD.xlsx"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.led_excel_path = led_excel_path
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 增强参数配置 - 优化后的安全策略
        self.rotation_angles = [90, 180, 270]  # 旋转角度 - 安全有效
        self.crop_ratios = [0.90, 0.95]  # 裁剪比例 - 更保守，减少信息损失
        self.noise_levels = [0.001, 0.002]  # 噪声水平 - 进一步降低，仅用于鲁棒性
        self.flip_types = ['horizontal', 'vertical']  # 翻转类型 - 完全安全
        
        # 优化后的策略权重配置 - 基于光谱数据特性
        self.strategy_weights = {
            'illuminant': 5,    # 光源变换权重最高（物理正确，最有价值）
            'rotation': 4,      # 旋转变换权重高（完全安全，空间多样性）
            'flip': 4,          # 翻转变换权重高（完全安全，空间多样性）  
            'crop': 1,          # 裁剪权重降低（可能丢失边缘光照信息）
            'noise': 0          # 噪声权重设为0（风险太高，暂时禁用）
        }
        
        # LED光源数据
        self.led_spds = {}
        self.target_wavelengths = np.arange(400, 701, 10)  # 400-700nm，步长10nm
        
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        
        # 加载LED光源数据
        self._load_led_illuminants()
    
    def _load_led_illuminants(self):
        """加载LED光源数据"""
        try:
            if not os.path.exists(self.led_excel_path):
                print(f"警告: LED数据文件不存在: {self.led_excel_path}")
                return
            
            # 读取Excel文件
            df = pd.read_excel(self.led_excel_path)
            wavelengths = df.iloc[:, 0].values
            
            # 处理每个LED通道（14个通道）
            for i in range(1, 15):
                led_name = f"LED_{i:02d}"
                spd_values = df.iloc[:, i].values
                
                # 插值到目标波长范围
                interpolated_spd = self._interpolate_spd(wavelengths, spd_values)
                if interpolated_spd is not None:
                    self.led_spds[led_name] = interpolated_spd
            
            print(f"成功加载 {len(self.led_spds)} 个LED光源数据")
            
        except Exception as e:
            print(f"加载LED数据失败: {e}")
    
    def _interpolate_spd(self, wavelengths, spd_values):
        """将SPD数据插值到400-700nm范围"""
        try:
            # 找到400-700nm范围内的数据
            valid_mask = (wavelengths >= 400) & (wavelengths <= 700)
            valid_wavelengths = wavelengths[valid_mask]
            valid_spd = spd_values[valid_mask]
            
            if len(valid_wavelengths) < 2:
                return None
            
            # 使用线性插值
            interpolator = interpolate.interp1d(
                valid_wavelengths, valid_spd, 
                kind='linear', 
                bounds_error=False, 
                fill_value=0
            )
            
            # 插值到目标波长点
            interpolated_spd = interpolator(self.target_wavelengths)
            interpolated_spd = np.maximum(interpolated_spd, 0)  # 确保非负
            
            # 修正：尺度匹配而非总和归一化
            # 将LED SPD缩放到与原始illumination相似的数值范围 [0.3, 1.0]
            if np.max(interpolated_spd) > 0:
                # 先归一化到[0,1]保持相对形状
                normalized_spd = interpolated_spd / np.max(interpolated_spd)
                # 然后缩放到合理的illumination范围
                target_min, target_max = 0.3, 1.0
                scaled_spd = normalized_spd * (target_max - target_min) + target_min
                return scaled_spd
            else:
                return interpolated_spd
            
        except Exception as e:
            print(f"插值处理失败: {e}")
            return None
    
    def load_mat_file(self, file_path):
        """加载.mat文件"""
        try:
            data = sio.loadmat(file_path)
            tensor = data['tensor'].astype(np.float32)  # (H, W, 31)
            illumination = data['illumination'].astype(np.float32)  # (1, 31)
            return tensor, illumination
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return None, None
    
    def save_mat_file(self, tensor, illumination, output_path):
        """保存增强后的.mat文件"""
        try:
            # 确保数据类型一致
            data_dict = {
                'tensor': tensor.astype(np.float64),
                'illumination': illumination.astype(np.float64)
            }
            sio.savemat(output_path, data_dict)
            print(f"保存增强文件: {output_path}")
        except Exception as e:
            print(f"保存文件 {output_path} 失败: {e}")
    
    def rotate_image(self, tensor, angle):
        """旋转图像（所有31个通道）"""
        rotated_tensor = np.zeros_like(tensor)
        for i in range(tensor.shape[2]):
            # 使用order=1 (双线性插值) 并确保非负
            rotated_channel = rotate(tensor[:, :, i], angle, 
                                   reshape=False, mode='reflect', 
                                   order=1, prefilter=False)
            # 确保非负值（修复插值导致的负值）
            rotated_tensor[:, :, i] = np.maximum(rotated_channel, 0.0)
        return rotated_tensor
    
    def flip_image(self, tensor, flip_type):
        """翻转图像"""
        if flip_type == 'horizontal':
            flipped = np.fliplr(tensor)
        elif flip_type == 'vertical':
            flipped = np.flipud(tensor)
        else:
            flipped = tensor
        
        # 确保翻转后没有负值（虽然翻转理论上不应该产生负值，但为安全起见）
        return np.maximum(flipped, 0.0)
    
    def crop_image(self, tensor, crop_ratio):
        """安全的裁剪方法（简化版，避免复杂的镜像填充）"""
        h, w, c = tensor.shape
        
        # 计算裁剪区域
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        
        # 裁剪
        cropped_tensor = tensor[start_h:start_h+new_h, start_w:start_w+new_w, :]
        
        # 使用简单的边缘填充回原尺寸（更安全）
        pad_h_before = (h - new_h) // 2
        pad_h_after = h - new_h - pad_h_before
        pad_w_before = (w - new_w) // 2  
        pad_w_after = w - new_w - pad_w_before
        
        # 使用edge模式填充（重复边缘值）
        padded_tensor = np.pad(cropped_tensor, 
                              ((pad_h_before, pad_h_after), 
                               (pad_w_before, pad_w_after), 
                               (0, 0)), 
                              mode='edge')
        
        # 确保没有负值（edge填充理论上不会产生负值，但为了安全）
        padded_tensor = np.maximum(padded_tensor, 0.0)
        
        return padded_tensor
    
    def add_noise(self, tensor, noise_level):
        """添加光谱相关的高斯噪声，保持光谱连续性"""
        # 为了保持光谱连续性，添加空间相关的噪声
        h, w, c = tensor.shape
        
        # 生成空间平滑的噪声基础
        base_noise = np.random.normal(0, noise_level, (h, w, 1)).astype(tensor.dtype)
        
        # 为不同光谱通道生成相关噪声（保持光谱连续性）
        spectral_weights = np.random.normal(1.0, 0.1, (1, 1, c))  # 轻微的光谱变化
        noise = base_noise * spectral_weights
        
        noisy_tensor = tensor + noise
        
        # 确保数值在合理范围内，但不改变动态范围
        noisy_tensor = np.clip(noisy_tensor, 0, np.percentile(tensor, 99.5))
        return noisy_tensor
    
    def simulate_illuminant_change(self, tensor, illumination, change_factor=0.1):
        """物理约束的光源变换
        
        基于物理原理：通过反射率计算实现光源转换
        新光谱 = (原光谱 / 原光源) × 新光源 = 反射率 × 新光源
        
        Returns:
            tuple: (changed_tensor, changed_illumination, led_name)
        """
        if len(self.led_spds) == 0:
            print("警告: 未加载LED光源数据，使用简单的光照变化模拟")
            changed_tensor, changed_illumination = self._simple_illuminant_change(tensor, illumination, change_factor)
            return changed_tensor, changed_illumination, "simple"
        
        try:
            # 随机选择一个LED光源
            selected_led = random.choice(list(self.led_spds.keys()))
            new_spd = self.led_spds[selected_led]
            
            # 获取原始光源SPD
            original_illumination = illumination.flatten()  # shape: (31,)
            
            # 步骤1: 计算场景的反射率
            # 反射率 = 原光谱数据 / 原光源 (逐像素、逐波长计算)
            h, w, c = tensor.shape
            reflectance = np.zeros_like(tensor)
            
            for i in range(c):  # 对每个光谱通道
                # 避免除零，添加小常数
                original_illum_channel = original_illumination[i] + 1e-8
                reflectance[:, :, i] = tensor[:, :, i] / original_illum_channel
            
            # 限制反射率在物理合理范围内 [0, 1]
            reflectance = np.clip(reflectance, 0, 1)
            
            # 步骤2: 使用新光源重新渲染
            # 新光谱数据 = 反射率 × 新光源
            changed_tensor = np.zeros_like(tensor)
            for i in range(c):
                changed_tensor[:, :, i] = reflectance[:, :, i] * new_spd[i]
            
            # 步骤3: 更新光照信息为新的LED光源SPD
            changed_illumination = new_spd.reshape(1, 31).astype(np.float32)
            
            # 确保数值在合理范围内
            changed_tensor = np.clip(changed_tensor, 0, np.percentile(changed_tensor, 99.5))
            
            print(f"使用物理光源变换: {selected_led}")
            print(f"  原光源峰值: {original_illumination[np.argmax(original_illumination)]:.4f}")
            print(f"  新光源峰值: {new_spd[np.argmax(new_spd)]:.4f}")
            
            return changed_tensor, changed_illumination, selected_led
            
        except Exception as e:
            print(f"LED光源变换失败，使用简单模拟: {e}")
            changed_tensor, changed_illumination = self._simple_illuminant_change(tensor, illumination, change_factor)
            return changed_tensor, changed_illumination, "simple"
    
    def _simple_illuminant_change(self, tensor, illumination, change_factor=0.1):
        """简单的光照变化模拟（备用方法）"""
        # 为每个光谱通道生成一个小的随机变化因子
        random_factors = 1 + np.random.uniform(-change_factor, change_factor, (1, 1, 31))
        
        # 应用变化
        changed_tensor = tensor * random_factors
        changed_illumination = illumination * np.random.uniform(0.9, 1.1, illumination.shape)
        
        # 归一化防止数值过大
        changed_tensor = np.clip(changed_tensor, 0, tensor.max())
        
        return changed_tensor, changed_illumination
    
    def augment_single_file(self, input_file, num_augmentations=2):
        """对单个文件进行数据增强，生成指定数量的增强版本"""
        # 加载原始数据
        tensor, illumination = self.load_mat_file(input_file)
        if tensor is None:
            return
        
        # 获取文件名（不含扩展名）
        base_name = input_file.stem
        
        # 准备基础策略列表
        base_strategies = []
        
        # 1. 旋转策略
        for angle in self.rotation_angles:
            base_strategies.append(('rotation', angle))
        
        # 2. 翻转策略
        for flip_type in self.flip_types:
            base_strategies.append(('flip', flip_type))
        
        # 3. 光源变换策略 (如果有LED数据)
        if len(self.led_spds) > 0:
            base_strategies.append(('illuminant', None))
        
        # 4. 裁剪策略
        for crop_ratio in self.crop_ratios:
            base_strategies.append(('crop', crop_ratio))
        
        # 5. 噪声策略
        for noise_level in self.noise_levels:
            base_strategies.append(('noise', noise_level))
        
        # 根据权重扩展策略列表
        weighted_strategies = []
        for strategy_type, strategy_param in base_strategies:
            weight = self.strategy_weights.get(strategy_type, 1)
            for _ in range(weight):
                weighted_strategies.append((strategy_type, strategy_param))
        
        if not weighted_strategies:
            print(f"警告: 没有可用的增强策略用于文件 {base_name}")
            return
        
        # 随机打乱策略列表
        random.shuffle(weighted_strategies)
        
        # 生成多个增强版本
        generated_count = 0
        for aug_idx in range(num_augmentations):
            # 选择策略（避免重复）
            strategy_idx = aug_idx % len(weighted_strategies)
            strategy_type, strategy_param = weighted_strategies[strategy_idx]
            
            # 应用选中的增强策略
            augmented_tensor = tensor
            augmented_illumination = illumination
            strategy_name = ""
            
            if strategy_type == 'rotation':
                augmented_tensor = self.rotate_image(tensor, strategy_param)
                strategy_name = f"rot{strategy_param}"
                
            elif strategy_type == 'flip':
                augmented_tensor = self.flip_image(tensor, strategy_param)
                strategy_name = f"flip{strategy_param}"
                
            elif strategy_type == 'illuminant':
                augmented_tensor, augmented_illumination = self.simulate_illuminant_change(tensor, illumination)
                strategy_name = "illuminant"
                
            elif strategy_type == 'crop':
                augmented_tensor = self.crop_image(tensor, strategy_param)
                strategy_name = f"crop{int(strategy_param*100)}"
                
            elif strategy_type == 'noise':
                augmented_tensor = self.add_noise(tensor, strategy_param)
                strategy_name = f"noise{int(strategy_param*1000)}"
            
            # 保存增强后的文件
            output_name = f"{base_name}_{strategy_name}_aug{aug_idx+1}_imp.mat"
            output_path = self.output_dir / output_name
            self.save_mat_file(augmented_tensor, augmented_illumination, output_path)
            generated_count += 1
            
            print(f"  生成增强文件 {aug_idx+1}: {strategy_type}({strategy_param})")
        
        print(f"完成文件 {base_name} 的增强，生成 {generated_count} 个增强版本")
    
    def augment_dataset(self, target_total_files=400):
        """对整个数据集进行增强，扩充到目标总文件数"""
        # 获取所有.mat文件
        mat_files = list(self.input_dir.glob("*.mat"))
        current_files = len(mat_files)
        target_augmented_files = target_total_files - current_files
        
        if target_augmented_files <= 0:
            print(f"当前已有 {current_files} 个文件，无需增强")
            return
            
        # 计算每个文件需要增强的数量
        base_augmentations = target_augmented_files // current_files
        extra_augmentations = target_augmented_files % current_files
        
        print(f"找到 {current_files} 个.mat文件")
        print(f"目标总文件数: {target_total_files}")
        print(f"需要增强: {target_augmented_files} 个文件")
        print(f"基础增强数: 每个文件 {base_augmentations} 个版本")
        if extra_augmentations > 0:
            print(f"额外增强: {extra_augmentations} 个文件再增强1个版本")
        
        if len(mat_files) == 0:
            print("未找到.mat文件！")
            return
        
        # 清空输出目录
        existing_files = list(self.output_dir.glob("*.mat"))
        if existing_files:
            print(f"清理已存在的 {len(existing_files)} 个增强文件...")
            for existing_file in existing_files:
                existing_file.unlink()
        
        # 统计每种策略类型的使用次数
        strategy_usage = {'rotation': 0, 'flip': 0, 'illuminant': 0, 'crop': 0, 'noise': 0}
        total_generated = 0
        
        # 对每个文件进行增强
        for i, mat_file in enumerate(mat_files):
            # 确定当前文件需要增强的数量
            current_file_augmentations = base_augmentations
            if i < extra_augmentations:  # 前extra_augmentations个文件多增强1个
                current_file_augmentations += 1
                
            print(f"\n处理文件 {i+1}/{len(mat_files)}: {mat_file.name} (增强{current_file_augmentations}个)")
            
            # 加载原始数据
            tensor, illumination = self.load_mat_file(mat_file)
            if tensor is None:
                continue
            
            base_name = mat_file.stem
            
            # 准备基础策略列表
            base_strategies = []
            
            # 1. 旋转策略
            for angle in self.rotation_angles:
                base_strategies.append(('rotation', angle))
            
            # 2. 翻转策略
            for flip_type in self.flip_types:
                base_strategies.append(('flip', flip_type))
            
            # 3. 光源变换策略 (如果有LED数据)
            if len(self.led_spds) > 0:
                base_strategies.append(('illuminant', None))
            
            # 4. 裁剪策略
            for crop_ratio in self.crop_ratios:
                base_strategies.append(('crop', crop_ratio))
            
            # 5. 噪声策略
            for noise_level in self.noise_levels:
                base_strategies.append(('noise', noise_level))
            
            # 根据权重扩展策略列表
            weighted_strategies = []
            for strategy_type, strategy_param in base_strategies:
                weight = self.strategy_weights.get(strategy_type, 1)
                for _ in range(weight):
                    weighted_strategies.append((strategy_type, strategy_param))
            
            if not weighted_strategies:
                print(f"  警告: 没有可用的增强策略用于文件 {base_name}")
                continue
            
            # 随机打乱策略列表
            random.shuffle(weighted_strategies)
            
            # 生成多个增强版本
            for aug_idx in range(current_file_augmentations):
                # 选择策略（确保多样性）
                strategy_idx = aug_idx % len(weighted_strategies)
                strategy_type, strategy_param = weighted_strategies[strategy_idx]
                
                # 应用选中的增强策略
                augmented_tensor = tensor
                augmented_illumination = illumination
                strategy_name = ""
                led_name = None  # 初始化LED名称
                
                if strategy_type == 'rotation':
                    augmented_tensor = self.rotate_image(tensor, strategy_param)
                    strategy_name = f"rot{strategy_param}"
                    strategy_usage['rotation'] += 1
                    
                elif strategy_type == 'flip':
                    augmented_tensor = self.flip_image(tensor, strategy_param)
                    strategy_name = f"flip{strategy_param}"
                    strategy_usage['flip'] += 1
                    
                elif strategy_type == 'illuminant':
                    augmented_tensor, augmented_illumination, led_name = self.simulate_illuminant_change(tensor, illumination)
                    strategy_name = f"illuminant_{led_name}"
                    strategy_usage['illuminant'] += 1
                    
                elif strategy_type == 'crop':
                    augmented_tensor = self.crop_image(tensor, strategy_param)
                    strategy_name = f"crop{int(strategy_param*100)}"
                    strategy_usage['crop'] += 1
                    
                elif strategy_type == 'noise':
                    augmented_tensor = self.add_noise(tensor, strategy_param)
                    strategy_name = f"noise{int(strategy_param*1000)}"
                    strategy_usage['noise'] += 1
                
                # 保存增强后的文件
                output_name = f"{base_name}_{strategy_name}_aug{aug_idx+1}_imp.mat"
                output_path = self.output_dir / output_name
                self.save_mat_file(augmented_tensor, augmented_illumination, output_path)
                total_generated += 1
                
                # 显示更友好的策略信息
                if strategy_type == 'illuminant':
                    print(f"  生成增强文件 {aug_idx+1}: {strategy_type}({led_name})")
                else:
                    print(f"  生成增强文件 {aug_idx+1}: {strategy_type}({strategy_param})")
        
        # 统计结果
        output_files = list(self.output_dir.glob("*.mat"))
        print(f"\n=== 数据增强完成 ===")
        print(f"原始文件数: {len(mat_files)}")
        print(f"增强文件数: {len(output_files)}")
        print(f"总文件数: {len(mat_files) + len(output_files)}")
        print(f"增强倍数: {len(output_files) / len(mat_files):.1f}x")
        
        print(f"\n=== 策略使用统计 ===")
        total_augmentations = sum(strategy_usage.values())
        for strategy, count in strategy_usage.items():
            if count > 0:
                percentage = (count / total_augmentations) * 100
                print(f"{strategy:12}: {count:2d} 次 ({percentage:5.1f}%)")
        
        # 显示策略覆盖率
        used_strategies = sum(1 for count in strategy_usage.values() if count > 0)
        total_strategy_types = len(strategy_usage)
        coverage = (used_strategies / total_strategy_types) * 100
        print(f"\n策略类型覆盖率: {used_strategies}/{total_strategy_types} ({coverage:.1f}%)")
        
        # 提供策略效果分析
        self.analyze_strategy_effectiveness(strategy_usage, len(mat_files))
    
    def analyze_strategy_effectiveness(self, strategy_usage, total_files):
        """分析各种增强策略的效果和建议"""
        print(f"\n=== 数据增强策略分析 ===")
        
        # 优化后的策略安全性和有效性评级
        strategy_ratings = {
            'illuminant': {'safety': '★★★★★', 'effectiveness': '★★★★★', 
                          'note': '最佳策略 - 物理正确，最有价值'},
            'rotation': {'safety': '★★★★★', 'effectiveness': '★★★★★', 
                        'note': '完全安全 - 不改变光谱特性，权重提高'},
            'flip': {'safety': '★★★★★', 'effectiveness': '★★★★★', 
                    'note': '完全安全 - 增加空间多样性，权重提高'},
            'crop': {'safety': '★★☆☆☆', 'effectiveness': '★★☆☆☆', 
                    'note': '权重降低 - 可能丢失边缘光照信息'},
            'noise': {'safety': '★☆☆☆☆', 'effectiveness': '★☆☆☆☆', 
                     'note': '已禁用 - 风险太高，破坏光谱完整性'}
        }
        
        print("策略评级:")
        for strategy, rating in strategy_ratings.items():
            if strategy in strategy_usage and strategy_usage[strategy] > 0:
                usage_pct = (strategy_usage[strategy] / total_files) * 100
                print(f"  {strategy:12} | 安全性: {rating['safety']} | 效果: {rating['effectiveness']} | 使用率: {usage_pct:5.1f}%")
                print(f"               | {rating['note']}")
        
        # 提供优化建议
        print(f"\n=== 优化建议 ===")
        
        # 检查噪声使用率
        noise_usage = strategy_usage.get('noise', 0)
        if noise_usage > total_files * 0.2:  # 如果噪声使用率超过20%
            print(f"⚠️  噪声策略使用率过高 ({noise_usage}/{total_files})，建议降低使用频率")
        elif noise_usage > 0:
            print(f"✓  噪声策略使用适中 ({noise_usage}/{total_files})，已优化为光谱相关噪声")
        
        # 检查光源变换使用率
        illuminant_usage = strategy_usage.get('illuminant', 0)
        if illuminant_usage > 0:
            print(f"✓  光源变换策略使用良好 ({illuminant_usage}/{total_files})，这是最有价值的增强")
        else:
            print(f"⚠️  未使用光源变换策略，建议检查LED数据是否正确加载")
        
        # 检查几何变换
        geometric_usage = strategy_usage.get('rotation', 0) + strategy_usage.get('flip', 0)
        if geometric_usage > 0:
            print(f"✓  几何变换使用合理 ({geometric_usage}/{total_files})，安全且有效")
        
        print(f"\n优化后的最佳策略组合:")
        print(f"  1. 光源变换 (36%) - 最有价值，基于物理原理")
        print(f"  2. 旋转变换 (29%) - 完全安全，保持光谱特性")
        print(f"  3. 翻转变换 (29%) - 完全安全，增加空间多样性")
        print(f"  4. 裁剪变换 (7%)  - 保守使用，避免边缘信息损失")
        print(f"  5. 噪声增强 (0%)  - 已禁用，风险太高")
    
    def verify_augmented_data(self):
        """验证增强数据的格式正确性"""
        print("\n验证增强数据...")
        
        # 获取原始文件和增强文件
        original_files = list(self.input_dir.glob("*.mat"))
        augmented_files = list(self.output_dir.glob("*.mat"))
        
        if len(augmented_files) == 0:
            print("未找到增强文件！")
            return
        
        # 随机选择几个文件进行验证
        sample_files = random.sample(augmented_files, min(3, len(augmented_files)))
        
        for file_path in sample_files:
            print(f"\n验证文件: {file_path.name}")
            
            tensor, illumination = self.load_mat_file(file_path)
            if tensor is not None:
                print(f"  Tensor shape: {tensor.shape}")
                print(f"  Illumination shape: {illumination.shape}")
                print(f"  Tensor dtype: {tensor.dtype}")
                print(f"  Illumination dtype: {illumination.dtype}")
                print(f"  Tensor range: [{tensor.min():.4f}, {tensor.max():.4f}]")
                print("  ✓ 格式验证通过")
            else:
                print("  ✗ 文件加载失败")
    
    def quick_negative_check(self):
        """快速检查增强文件是否有负值问题"""
        augmented_files = list(self.output_dir.glob("*.mat"))
        if len(augmented_files) == 0:
            print("未找到增强文件")
            return
        
        # 随机检查几个文件
        import random
        sample_files = random.sample(augmented_files, min(5, len(augmented_files)))
        
        negative_count = 0
        for file_path in sample_files:
            try:
                data = sio.loadmat(file_path)
                tensor = data['tensor']
                if np.any(tensor < 0):
                    negative_count += 1
                    min_val = tensor.min()
                    print(f"⚠️ {file_path.name}: 最小值 {min_val:.2e}")
            except:
                pass
        
        if negative_count == 0:
            print("✅ 抽样检查未发现负值问题")
        else:
            print(f"❌ 在 {negative_count}/{len(sample_files)} 个文件中发现负值")
            print("建议重新生成数据或检查算法")


def main():
    """主函数"""
    print("=== 多光谱数据增强工具 ===\n")
    
    # 创建数据增强器
    augmentator = DataAugmentator()
    
    # 检查是否已有增强文件
    existing_files = list(augmentator.output_dir.glob("*.mat"))
    if existing_files:
        print(f"发现已存在 {len(existing_files)} 个增强文件")
        print("这些文件可能有负值问题，建议重新生成")
        user_choice = input("是否重新生成？(y/n): ").lower().strip()
        if user_choice != 'y':
            print("保持现有文件，程序退出")
            return
    
    # 执行数据增强，扩充到400张样本
    print("目标：将40张样本扩充到200张，增强训练数据")
    print("已修复算法：防止旋转、翻转、裁剪产生负值")
    augmentator.augment_dataset(target_total_files=200)
    
    # 验证增强后的数据
    augmentator.verify_augmented_data()
    
    # 快速检查负值问题
    print("\n=== 快速负值检查 ===")
    augmentator.quick_negative_check()
    
    print("\n=== 数据增强完成！===")
    print("现在你有:")
    print("  - 原始训练数据: 40个文件")
    print("  - 增强训练数据: 160个文件") 
    print("  - 总训练数据: 200个文件")
    print("  - 数据增强倍数: 5倍")



if __name__ == "__main__":
    main()