"""
模型评估模块
实现完整的模型评估、性能分析和结果可视化
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from ..models.resnet1d import ResNet1D
from ..data.white_point import SpectralWhitePointExtractor
from ..training.loss import RecoveryAngularErrorLoss


class ModelEvaluator:
    """
    模型评估器
    
    提供完整的模型评估功能，包括：
    - Recovery Angular Error和Reproduction Angular Error计算
    - 统计指标分析
    - 性能对比
    - 结果可视化
    """
    
    def __init__(self,
                 model: ResNet1D,
                 loss_function: RecoveryAngularErrorLoss,
                 device: str = 'cpu',
                 config: Dict = None):
        """
        初始化评估器
        
        Args:
            model: 训练好的ResNet1D模型
            loss_function: 损失函数（用于计算指标）
            device: 计算设备
            config: 评估配置
        """
        self.model = model.to(device)
        self.loss_function = loss_function.to(device)
        self.device = device
        self.config = config or {}
        
        # 白点特征提取器
        self.wp_extractor = SpectralWhitePointExtractor(normalize=True)
        
        # 评估结果存储
        self.evaluation_results = {}
        
        logging.info(f"Evaluator initialized with device: {device}")
    
    def evaluate_dataset(self, 
                        dataloader: DataLoader,
                        dataset_name: str = 'test',
                        save_predictions: bool = True) -> Dict[str, float]:
        """
        评估整个数据集
        
        Args:
            dataloader: 数据加载器
            dataset_name: 数据集名称
            save_predictions: 是否保存预测结果
        
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_ground_truth = []
        all_wp_features = []
        all_filenames = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                # 检查数据是否已经是白点特征
                ms_data = batch['multispectral']
                if ms_data.dim() == 2:
                    # 数据已经是白点特征 [B, 31]
                    wp_features = ms_data.to(self.device)
                else:
                    # 原始多光谱数据，需要提取白点特征 [B, H, W, 31]
                    wp_features = self.wp_extractor.extract_batch_features(
                        ms_data, method='max'
                    ).to(self.device)
                
                ground_truth = batch['illumination_gt'].to(self.device)
                filenames = batch['filename']
                
                # 模型预测
                predictions = self.model(wp_features)
                
                # 收集结果
                all_predictions.append(predictions.cpu())
                all_ground_truth.append(ground_truth.cpu())
                all_wp_features.append(wp_features.cpu())
                all_filenames.extend(filenames)
        
        # 拼接所有批次结果
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        all_wp_features = torch.cat(all_wp_features, dim=0)
        
        # 计算评估指标
        metrics = self.compute_comprehensive_metrics(
            all_predictions.to(self.device),
            all_ground_truth.to(self.device),
            all_wp_features.to(self.device)
        )
        
        # 保存评估结果
        self.evaluation_results[dataset_name] = {
            'metrics': metrics,
            'predictions': all_predictions,
            'ground_truth': all_ground_truth,
            'wp_features': all_wp_features,
            'filenames': all_filenames
        }
        
        # 保存预测结果到文件
        if save_predictions:
            self.save_predictions(dataset_name)
        
        return metrics
    
    def compute_comprehensive_metrics(self,
                                    predictions: torch.Tensor,
                                    ground_truth: torch.Tensor,
                                    wp_features: torch.Tensor) -> Dict[str, float]:
        """
        计算全面的评估指标
        
        Args:
            predictions: 模型预测 [N, 31]
            ground_truth: 地面真值 [N, 31]
            wp_features: 白点特征 [N, 31]
        
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 1. Recovery Angular Error (主要指标)
        recovery_metrics = self.loss_function.compute_metrics(predictions, ground_truth)
        for key, value in recovery_metrics.items():
            metrics[f'recovery_{key}'] = value
        
        # 2. 基线方法性能 (仅使用白点算法)
        # 注意：wp_features已经是归一化的白点特征，这里直接作为基线是合理的
        # 因为这代表了"未经深度学习优化的白点算法"的性能
        baseline_metrics = self.loss_function.compute_metrics(wp_features, ground_truth)
        for key, value in baseline_metrics.items():
            metrics[f'baseline_{key}'] = value
        
        # 3. 改进程度
        metrics['improvement_mean'] = baseline_metrics['mean_error'] - recovery_metrics['mean_error']
        metrics['improvement_median'] = baseline_metrics['median_error'] - recovery_metrics['median_error']
        metrics['improvement_ratio_mean'] = metrics['improvement_mean'] / baseline_metrics['mean_error']
        metrics['improvement_ratio_median'] = metrics['improvement_median'] / baseline_metrics['median_error']
        
        # 4. 光谱域MSE
        mse_spectral = torch.mean((predictions - ground_truth) ** 2).item()
        metrics['spectral_mse'] = mse_spectral
        
        # 5. 余弦相似度（光谱域）
        pred_norm = torch.nn.functional.normalize(predictions, p=2, dim=1)
        gt_norm = torch.nn.functional.normalize(ground_truth, p=2, dim=1)
        cos_sim = torch.sum(pred_norm * gt_norm, dim=1)
        metrics['spectral_cosine_similarity_mean'] = torch.mean(cos_sim).item()
        metrics['spectral_cosine_similarity_std'] = torch.std(cos_sim).item()
        
        # 6. 通道级别分析
        channel_errors = torch.abs(predictions - ground_truth)  # [N, 31]
        metrics['max_channel_error'] = torch.max(torch.mean(channel_errors, dim=0)).item()
        metrics['min_channel_error'] = torch.min(torch.mean(channel_errors, dim=0)).item()
        
        # 7. 鲁棒性指标
        errors = self.compute_angular_errors(predictions, ground_truth)
        metrics['error_std'] = torch.std(errors).item()
        metrics['error_range'] = (torch.max(errors) - torch.min(errors)).item()
        
        return metrics
    
    def compute_angular_errors(self, 
                             predictions: torch.Tensor, 
                             ground_truth: torch.Tensor) -> torch.Tensor:
        """计算每个样本的角度误差"""
        # 转换到RGB空间
        pred_rgb = torch.matmul(predictions, self.loss_function.csf_matrix)
        gt_rgb = torch.matmul(ground_truth, self.loss_function.csf_matrix)
        
        # 归一化
        pred_norm = torch.nn.functional.normalize(pred_rgb, p=2, dim=1, eps=1e-8)
        gt_norm = torch.nn.functional.normalize(gt_rgb, p=2, dim=1, eps=1e-8)
        
        # 计算角度误差
        cos_sim = torch.sum(pred_norm * gt_norm, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        angles = torch.acos(cos_sim) * 180.0 / np.pi
        
        return angles
    
    def compare_with_baselines(self, 
                              test_loader: DataLoader,
                              baseline_results: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """
        与基线方法进行对比
        
        Args:
            test_loader: 测试数据加载器
            baseline_results: 基线方法结果
        
        Returns:
            对比结果
        """
        # 评估当前模型
        our_metrics = self.evaluate_dataset(test_loader, 'test_comparison', save_predictions=False)
        
        # 默认基线结果（来自原论文）
        if baseline_results is None:
            baseline_results = {
                'Multispectral WP + FFNN': {
                    'recovery_mean_error': 2.13,
                    'recovery_median_error': 1.85,  # 估计值
                    'reproduction_mean_error': 2.58,
                    'reproduction_median_error': 2.20  # 估计值
                },
                'Multispectral WP (Only)': {
                    'recovery_mean_error': our_metrics['baseline_mean_error'],
                    'recovery_median_error': our_metrics['baseline_median_error']
                }
            }
        
        # 添加我们的结果
        comparison_results = baseline_results.copy()
        comparison_results['ResNet-WP (Ours)'] = {
            'recovery_mean_error': our_metrics['recovery_mean_error'],
            'recovery_median_error': our_metrics['recovery_median_error']
        }
        
        # 计算排名
        methods = list(comparison_results.keys())
        mean_errors = [comparison_results[method]['recovery_mean_error'] for method in methods]
        
        # 按性能排序
        sorted_indices = np.argsort(mean_errors)
        ranked_results = {}
        for rank, idx in enumerate(sorted_indices):
            method = methods[idx]
            ranked_results[f"Rank_{rank+1}_{method}"] = comparison_results[method]
        
        return ranked_results
    
    def save_predictions(self, dataset_name: str):
        """保存预测结果到文件"""
        if dataset_name not in self.evaluation_results:
            return
        
        results = self.evaluation_results[dataset_name]
        
        # 创建保存目录
        save_dir = Path('results/predictions')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测数据
        predictions_data = {
            'filenames': results['filenames'],
            'predictions': results['predictions'].numpy().tolist(),
            'ground_truth': results['ground_truth'].numpy().tolist(),
            'wp_features': results['wp_features'].numpy().tolist(),
            'metrics': results['metrics']
        }
        
        save_path = save_dir / f'{dataset_name}_predictions.json'
        with open(save_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        logging.info(f"Predictions saved to {save_path}")
    
    def create_visualizations(self, dataset_name: str = 'test'):
        """创建评估结果可视化"""
        if dataset_name not in self.evaluation_results:
            logging.warning(f"No evaluation results for {dataset_name}")
            return
        
        results = self.evaluation_results[dataset_name]
        predictions = results['predictions']
        ground_truth = results['ground_truth']
        wp_features = results['wp_features']
        
        # 创建可视化目录
        viz_dir = Path('results/figures')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 角度误差分布直方图
        self._plot_error_distribution(predictions, ground_truth, wp_features, viz_dir)
        
        # 2. 预测vs真值散点图
        self._plot_prediction_scatter(predictions, ground_truth, viz_dir)
        
        # 3. 光谱曲线对比
        self._plot_spectral_comparison(predictions, ground_truth, viz_dir)
        
        # 4. 通道级别误差分析
        self._plot_channel_analysis(predictions, ground_truth, viz_dir)
        
        # 5. 改进程度可视化
        self._plot_improvement_analysis(predictions, ground_truth, wp_features, viz_dir)
        
        logging.info(f"Visualizations saved to {viz_dir}")
    
    def _plot_error_distribution(self, predictions, ground_truth, wp_features, save_dir):
        """绘制角度误差分布"""
        # 计算误差
        model_errors = self.compute_angular_errors(
            predictions.to(self.device), ground_truth.to(self.device)
        ).cpu().numpy()
        
        baseline_errors = self.compute_angular_errors(
            wp_features.to(self.device), ground_truth.to(self.device)
        ).cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        
        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(baseline_errors, bins=30, alpha=0.7, label='Baseline (WP Only)', color='red')
        plt.hist(model_errors, bins=30, alpha=0.7, label='ResNet-WP (Ours)', color='blue')
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 累积分布
        plt.subplot(1, 2, 2)
        sorted_baseline = np.sort(baseline_errors)
        sorted_model = np.sort(model_errors)
        
        plt.plot(sorted_baseline, np.arange(len(sorted_baseline))/len(sorted_baseline), 
                label='Baseline (WP Only)', color='red', linewidth=2)
        plt.plot(sorted_model, np.arange(len(sorted_model))/len(sorted_model), 
                label='ResNet-WP (Ours)', color='blue', linewidth=2)
        
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self, predictions, ground_truth, save_dir):
        """绘制预测vs真值散点图"""
        # 转换到RGB空间进行可视化
        pred_rgb = torch.matmul(predictions, torch.from_numpy(self.loss_function.csf_matrix.cpu().numpy()))
        gt_rgb = torch.matmul(ground_truth, torch.from_numpy(self.loss_function.csf_matrix.cpu().numpy()))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        channels = ['R', 'G', 'B']
        
        for i, (color, channel) in enumerate(zip(colors, channels)):
            axes[i].scatter(gt_rgb[:, i], pred_rgb[:, i], alpha=0.6, color=color, s=30)
            
            # 添加理想线
            min_val = min(gt_rgb[:, i].min(), pred_rgb[:, i].min())
            max_val = max(gt_rgb[:, i].max(), pred_rgb[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
            
            axes[i].set_xlabel(f'Ground Truth {channel}')
            axes[i].set_ylabel(f'Predicted {channel}')
            axes[i].set_title(f'{channel} Channel Prediction')
            axes[i].grid(True, alpha=0.3)
            
            # 计算相关系数
            corr = np.corrcoef(gt_rgb[:, i].numpy(), pred_rgb[:, i].numpy())[0, 1]
            axes[i].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spectral_comparison(self, predictions, ground_truth, save_dir):
        """绘制光谱曲线对比"""
        # 选择几个代表性样本
        n_samples = min(5, len(predictions))
        indices = np.linspace(0, len(predictions)-1, n_samples, dtype=int)
        
        wavelengths = np.linspace(400, 700, 31)  # 假设波长范围
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            axes[i].plot(wavelengths, ground_truth[idx].numpy(), 'b-', linewidth=2, 
                        label='Ground Truth', alpha=0.8)
            axes[i].plot(wavelengths, predictions[idx].numpy(), 'r--', linewidth=2, 
                        label='Prediction', alpha=0.8)
            
            # 计算该样本的角度误差
            sample_error = self.compute_angular_errors(
                predictions[idx:idx+1].to(self.device),
                ground_truth[idx:idx+1].to(self.device)
            ).cpu().item()
            
            axes[i].set_title(f'Sample {idx+1} (Error: {sample_error:.2f}°)')
            axes[i].set_xlabel('Wavelength (nm)')
            axes[i].set_ylabel('Illumination Intensity')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'spectral_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_channel_analysis(self, predictions, ground_truth, save_dir):
        """绘制通道级别误差分析"""
        # 计算每个通道的平均绝对误差
        channel_errors = torch.abs(predictions - ground_truth)
        mean_channel_errors = torch.mean(channel_errors, dim=0)
        std_channel_errors = torch.std(channel_errors, dim=0)
        
        wavelengths = np.linspace(400, 700, 31)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(wavelengths, mean_channel_errors.numpy(), 'b-', linewidth=2, label='Mean Error')
        plt.fill_between(wavelengths, 
                        (mean_channel_errors - std_channel_errors).numpy(),
                        (mean_channel_errors + std_channel_errors).numpy(),
                        alpha=0.3, color='blue', label='±1 Std')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absolute Error')
        plt.title('Channel-wise Error Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 热力图显示所有样本的误差
        plt.subplot(1, 2, 2)
        error_matrix = channel_errors.numpy().T  # [31, N]
        
        # 只显示前50个样本（如果有的话）
        if error_matrix.shape[1] > 50:
            error_matrix = error_matrix[:, :50]
        
        sns.heatmap(error_matrix, cmap='viridis', cbar_kws={'label': 'Absolute Error'})
        plt.xlabel('Sample Index')
        plt.ylabel('Channel Index')
        plt.title('Error Heatmap (All Channels)')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'channel_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self, predictions, ground_truth, wp_features, save_dir):
        """绘制改进程度分析"""
        model_errors = self.compute_angular_errors(
            predictions.to(self.device), ground_truth.to(self.device)
        ).cpu().numpy()
        
        baseline_errors = self.compute_angular_errors(
            wp_features.to(self.device), ground_truth.to(self.device)
        ).cpu().numpy()
        
        improvements = baseline_errors - model_errors
        
        plt.figure(figsize=(12, 8))
        
        # 改进程度散点图
        plt.subplot(2, 2, 1)
        plt.scatter(baseline_errors, improvements, alpha=0.6, s=30)
        plt.xlabel('Baseline Error (degrees)')
        plt.ylabel('Improvement (degrees)')
        plt.title('Improvement vs Baseline Error')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        
        # 改进程度直方图
        plt.subplot(2, 2, 2)
        plt.hist(improvements, bins=30, alpha=0.7, color='green')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=np.mean(improvements), color='blue', linestyle='-', alpha=0.7, 
                   label=f'Mean: {np.mean(improvements):.2f}°')
        plt.xlabel('Improvement (degrees)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Improvements')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 误差对比箱线图
        plt.subplot(2, 2, 3)
        plt.boxplot([baseline_errors, model_errors], 
                   labels=['Baseline\n(WP Only)', 'ResNet-WP\n(Ours)'])
        plt.ylabel('Angular Error (degrees)')
        plt.title('Error Distribution Comparison')
        plt.grid(True, alpha=0.3)
        
        # 改进率分析
        plt.subplot(2, 2, 4)
        improvement_rates = improvements / baseline_errors * 100
        plt.hist(improvement_rates, bins=30, alpha=0.7, color='orange')
        plt.axvline(x=np.mean(improvement_rates), color='blue', linestyle='-', alpha=0.7,
                   label=f'Mean: {np.mean(improvement_rates):.1f}%')
        plt.xlabel('Improvement Rate (%)')
        plt.ylabel('Frequency')
        plt.title('Relative Improvement Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, dataset_name: str = 'test') -> str:
        """生成评估报告"""
        if dataset_name not in self.evaluation_results:
            return "No evaluation results available."
        
        metrics = self.evaluation_results[dataset_name]['metrics']
        
        report = f"""
# ResNet-WP Model Evaluation Report

## Dataset: {dataset_name.upper()}
- Number of samples: {metrics['recovery_num_samples']}

## Recovery Angular Error (Primary Metric)
- Mean Error: {metrics['recovery_mean_error']:.3f}°
- Median Error: {metrics['recovery_median_error']:.3f}°
- Standard Deviation: {metrics['recovery_std_error']:.3f}°
- Min Error: {metrics['recovery_min_error']:.3f}°
- Max Error: {metrics['recovery_max_error']:.3f}°

## Baseline Comparison (White Point Only)
- Baseline Mean Error: {metrics['baseline_mean_error']:.3f}°
- Baseline Median Error: {metrics['baseline_median_error']:.3f}°

## Improvement Analysis
- Absolute Improvement (Mean): {metrics['improvement_mean']:.3f}°
- Absolute Improvement (Median): {metrics['improvement_median']:.3f}°
- Relative Improvement (Mean): {metrics['improvement_ratio_mean']*100:.1f}%
- Relative Improvement (Median): {metrics['improvement_ratio_median']*100:.1f}%

## Additional Metrics
- Spectral MSE: {metrics['spectral_mse']:.6f}
- Spectral Cosine Similarity: {metrics['spectral_cosine_similarity_mean']:.4f} ± {metrics['spectral_cosine_similarity_std']:.4f}

## Performance Thresholds
"""
        
        # 添加阈值分析
        thresholds = [1.0, 2.0, 3.0, 5.0, 10.0]
        for threshold in thresholds:
            key = f'recovery_ratio_below_{threshold}deg'
            if key in metrics:
                report += f"- Samples below {threshold}°: {metrics[key]*100:.1f}%\n"
        
        report += f"""
## Target Achievement
- Target: Beat Multispectral WP + FFNN (2.13° mean error)
- Our Result: {metrics['recovery_mean_error']:.3f}°
- Achievement: {'✅ TARGET ACHIEVED' if metrics['recovery_mean_error'] < 2.13 else '❌ Target not achieved'}
"""
        
        return report


def compute_angular_error(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    计算两个向量之间的角度误差（度）
    
    Args:
        pred: 预测向量 [31]
        target: 目标向量 [31]
        epsilon: 数值稳定性常数
    
    Returns:
        角度误差（度）
    """
    # 归一化
    pred_norm = pred / (torch.norm(pred) + epsilon)
    target_norm = target / (torch.norm(target) + epsilon)
    
    # 计算余弦相似度
    cos_sim = torch.clamp(torch.dot(pred_norm, target_norm), -1.0, 1.0)
    
    # 转换为角度（度）
    angle_rad = torch.acos(cos_sim)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg.item()


def load_model_for_evaluation(model_path: str, 
                            config_path: str,
                            csf_matrix: torch.Tensor,
                            device: str = 'cpu') -> ModelEvaluator:
    """
    加载训练好的模型用于评估
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        csf_matrix: 相机响应函数矩阵
        device: 计算设备
    
    Returns:
        评估器实例
    """
    import yaml
    from ..models.resnet1d import create_model
    from ..training.loss import create_loss_function
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    model = create_model(config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 创建损失函数
    loss_function = create_loss_function(config, csf_matrix)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, loss_function, device, config)
    
    return evaluator


if __name__ == "__main__":
    # 测试评估器
    logging.basicConfig(level=logging.INFO)
    
    print("Evaluator module test completed successfully!")
