"""
按场景划分数据集工具
避免数据泄露：同一场景的所有图像（原始+增强）必须在同一集合中
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict
import random


def extract_scene_name(filename: str) -> str:
    """
    从文件名提取场景名
    
    例子：
    - "Daylight_Scene_01.mat" -> "Daylight_Scene_01"
    - "Daylight_Scene_01_fliphorizontal_aug1_imp.mat" -> "Daylight_Scene_01"
    - "Metal_halide_lamp_2500K_Scene_01_crop85_aug1_imp.mat" -> "Metal_halide_lamp_2500K_Scene_01"
    """
    # 移除扩展名
    stem = Path(filename).stem
    
    # 匹配模式：{光源类型}_Scene_{数字} 
    # 之后可能有 _aug, _imp, _flip 等增强标记
    match = re.match(r'(.+_Scene_\d+)', stem)
    
    if match:
        return match.group(1)
    else:
        # 如果没有匹配到，返回整个stem（去除aug标记）
        # 这是备用方案
        return stem.split('_aug')[0].split('_flip')[0].split('_rot')[0].split('_crop')[0].split('_noise')[0].split('_illuminant')[0]


def group_files_by_scene(file_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    按场景对文件进行分组
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        字典，键为场景名，值为该场景的所有文件路径列表
    """
    scene_groups = {}
    
    for file_path in file_paths:
        scene_name = extract_scene_name(file_path.name)
        
        if scene_name not in scene_groups:
            scene_groups[scene_name] = []
        
        scene_groups[scene_name].append(file_path)
    
    return scene_groups


def split_by_scene(file_paths: List[Path],
                   train_ratio: float = 0.70,
                   random_seed: int = 42) -> Tuple[List[Path], List[Path]]:
    """
    按场景划分训练集和验证集
    
    确保同一场景的所有图像（原始+增强）都在同一集合中，避免数据泄露
    
    Args:
        file_paths: 所有文件路径列表
        train_ratio: 训练集比例
        random_seed: 随机种子
        
    Returns:
        (train_files, val_files)
    """
    # 按场景分组
    scene_groups = group_files_by_scene(file_paths)
    
    # 获取所有场景名并排序（确保可复现）
    scene_names = sorted(scene_groups.keys())
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 打乱场景顺序
    random.shuffle(scene_names)
    
    # 计算训练集场景数
    num_train_scenes = int(len(scene_names) * train_ratio)
    
    # 划分场景
    train_scenes = scene_names[:num_train_scenes]
    val_scenes = scene_names[num_train_scenes:]
    
    # 收集文件
    train_files = []
    val_files = []
    
    for scene in train_scenes:
        train_files.extend(scene_groups[scene])
    
    for scene in val_scenes:
        val_files.extend(scene_groups[scene])
    
    return train_files, val_files


def analyze_scene_split(train_files: List[Path], val_files: List[Path]) -> Dict:
    """
    分析场景划分的统计信息
    
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        
    Returns:
        统计信息字典
    """
    train_scenes = set(extract_scene_name(f.name) for f in train_files)
    val_scenes = set(extract_scene_name(f.name) for f in val_files)
    
    # 检查是否有场景重叠（数据泄露）
    overlap = train_scenes & val_scenes
    
    stats = {
        'num_train_files': len(train_files),
        'num_val_files': len(val_files),
        'num_train_scenes': len(train_scenes),
        'num_val_scenes': len(val_scenes),
        'has_overlap': len(overlap) > 0,
        'overlap_scenes': list(overlap) if overlap else [],
        'train_scenes': sorted(train_scenes),
        'val_scenes': sorted(val_scenes),
    }
    
    return stats


if __name__ == "__main__":
    # 测试代码
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试场景名提取
    test_filenames = [
        "Daylight_Scene_01.mat",
        "Daylight_Scene_01_fliphorizontal_aug1_imp.mat",
        "Metal_halide_lamp_2500K_Scene_01_crop85_aug1_imp.mat",
        "LED_E400_Scene_01_illuminant_LED_14_aug1_imp.mat",
    ]
    
    print("测试场景名提取：")
    for filename in test_filenames:
        scene = extract_scene_name(filename)
        print(f"  {filename} -> {scene}")
    
    # 测试数据目录
    data_dir = Path("../../data/dataset/training/mat_norm")
    if data_dir.exists():
        print("\n测试按场景划分：")
        files = sorted(data_dir.glob("*.mat"))
        train_files, val_files = split_by_scene(files, train_ratio=0.70, random_seed=42)
        
        stats = analyze_scene_split(train_files, val_files)
        print(f"\n统计信息：")
        print(f"  训练文件数: {stats['num_train_files']}")
        print(f"  验证文件数: {stats['num_val_files']}")
        print(f"  训练场景数: {stats['num_train_scenes']}")
        print(f"  验证场景数: {stats['num_val_scenes']}")
        print(f"  是否有场景重叠: {stats['has_overlap']}")
        
        if stats['has_overlap']:
            print(f"  ⚠️ 警告：发现数据泄露！重叠场景: {stats['overlap_scenes']}")
        else:
            print(f"  ✅ 无数据泄露")
        
        print(f"\n训练场景: {stats['train_scenes'][:5]}... ({len(stats['train_scenes'])} total)")
        print(f"验证场景: {stats['val_scenes'][:5]}... ({len(stats['val_scenes'])} total)")

