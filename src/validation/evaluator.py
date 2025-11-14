"""
统一的疾病检测评估器

特点：
1. 训练时：快速评估，不用bootstrap
2. 推理时：完整评估，带bootstrap CI
3. 保证训练和推理使用相同的评估逻辑
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

from .metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1_optimal,
    compute_confusion_matrix_metrics,
    aggregate_metrics
)


class DiseaseEvaluator:
    """
    疾病检测统一评估器

    用法示例：
        # 训练时（快速）
        evaluator = DiseaseEvaluator(
            pathology_classes=['Atelectasis', 'Cardiomegaly'],
            metrics=['auroc', 'auprc', 'f1'],
            use_bootstrap=False
        )

        # 推理时（完整）
        evaluator = DiseaseEvaluator(
            pathology_classes=['Atelectasis', 'Cardiomegaly'],
            metrics=['auroc', 'auprc', 'f1', 'precision', 'recall'],
            use_bootstrap=True,
            n_bootstrap=1000
        )

        # 评估
        results = evaluator.evaluate(predictions, labels)
    """

    def __init__(
        self,
        pathology_classes: List[str],
        metrics: List[str] = ["auroc", "auprc", "f1"],
        use_bootstrap: bool = False,
        n_bootstrap: int = 1000,
        bootstrap_ci: float = 0.95,
        threshold: float = 0.5,
        random_state: int = 42
    ):
        """
        Args:
            pathology_classes: 病理类别列表
            metrics: 要计算的指标列表
                支持: 'auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity'
            use_bootstrap: 是否使用bootstrap计算置信区间
            n_bootstrap: bootstrap采样次数
            bootstrap_ci: 置信区间水平 (default: 0.95 = 95% CI)
            threshold: 二分类阈值 (default: 0.5)
            random_state: 随机种子
        """
        self.pathology_classes = pathology_classes
        self.num_classes = len(pathology_classes)
        self.metrics = metrics
        self.use_bootstrap = use_bootstrap
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = bootstrap_ci
        self.threshold = threshold
        self.random_state = random_state

        # 验证metrics
        valid_metrics = {'auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity', 'accuracy'}
        for m in metrics:
            if m not in valid_metrics:
                raise ValueError(f"Unknown metric: {m}. Valid metrics: {valid_metrics}")

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        return_per_class: bool = True,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        评估预测结果

        Args:
            predictions: 预测分数 (N, num_classes), 范围[0, 1]
            labels: 真实标签 (N, num_classes), 0/1
            return_per_class: 是否返回每个类别的详细指标
            verbose: 是否打印详细信息

        Returns:
            评估结果字典，包含：
            - macro_auroc, macro_auprc, ... (宏平均)
            - 如果return_per_class=True:
                per_class: {
                    'Atelectasis': {'auroc': 0.85, 'auprc': 0.72, ...},
                    ...
                }
            - 如果use_bootstrap=True，还包括CI:
                macro_auroc_ci_lower, macro_auroc_ci_upper, ...
        """
        # 验证输入
        assert predictions.shape == labels.shape, \
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        assert predictions.shape[1] == self.num_classes, \
            f"Number of classes mismatch: {predictions.shape[1]} vs {self.num_classes}"

        if verbose:
            print(f"Evaluating {predictions.shape[0]} samples across {self.num_classes} classes")
            print(f"Metrics: {self.metrics}")
            print(f"Bootstrap: {self.use_bootstrap}")

        # 计算每个类别的指标
        per_class_metrics = {}
        skipped_classes = []  # Track classes with single-class issue

        for i, class_name in enumerate(self.pathology_classes):
            y_true = labels[:, i]
            y_score = predictions[:, i]

            # Check if only one class present
            if len(np.unique(y_true)) < 2:
                skipped_classes.append(class_name)

            class_metrics = self._evaluate_single_class(y_true, y_score)
            per_class_metrics[class_name] = class_metrics

        # Print summary if classes were skipped
        if skipped_classes and verbose:
            print(f"⚠️  {len(skipped_classes)}/{self.num_classes} classes skipped due to single-class samples: {skipped_classes[:3]}{'...' if len(skipped_classes) > 3 else ''}")

        # 聚合指标（macro average）
        results = aggregate_metrics(per_class_metrics, self.pathology_classes)

        # 如果需要，添加每个类别的详细信息
        if return_per_class:
            results['per_class'] = per_class_metrics

        # 如果使用bootstrap，计算置信区间
        if self.use_bootstrap:
            ci_results = self._compute_bootstrap_ci(predictions, labels, verbose=verbose)
            results.update(ci_results)

        return results

    def _evaluate_single_class(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Dict[str, float]:
        """
        评估单个类别

        Args:
            y_true: 真实标签 (N,)
            y_score: 预测分数 (N,)

        Returns:
            该类别的所有指标
        """
        metrics_dict = {}

        # AUROC
        if 'auroc' in self.metrics:
            metrics_dict['auroc'] = compute_auroc(y_true, y_score)

        # AUPRC
        if 'auprc' in self.metrics:
            metrics_dict['auprc'] = compute_auprc(y_true, y_score)

        # F1 (optimal threshold)
        if 'f1' in self.metrics:
            f1, _ = compute_f1_optimal(y_true, y_score)
            metrics_dict['f1'] = f1

        # 基于固定阈值的指标
        need_cm_metrics = any(m in self.metrics for m in ['precision', 'recall', 'specificity', 'accuracy'])

        if need_cm_metrics:
            y_pred = (y_score >= self.threshold).astype(int)
            cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)

            # 只添加请求的指标
            for metric_name in ['precision', 'recall', 'specificity', 'accuracy']:
                if metric_name in self.metrics:
                    metrics_dict[metric_name] = cm_metrics[metric_name]

        return metrics_dict

    def _compute_bootstrap_ci(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        使用bootstrap计算置信区间

        对macro metrics计算CI

        Args:
            predictions: (N, num_classes)
            labels: (N, num_classes)
            verbose: 是否显示进度

        Returns:
            包含CI的字典: {
                'macro_auroc_ci_lower': float,
                'macro_auroc_ci_upper': float,
                ...
            }
        """
        if verbose:
            print(f"Computing bootstrap CI with {self.n_bootstrap} samples...")

        n_samples = predictions.shape[0]
        np.random.seed(self.random_state)

        # 存储每次bootstrap的macro metrics
        bootstrap_results = {metric: [] for metric in self.metrics}

        for i in range(self.n_bootstrap):
            # Bootstrap采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_boot = predictions[indices]
            label_boot = labels[indices]

            # 计算每个类别的指标
            per_class_boot = {}
            for j, class_name in enumerate(self.pathology_classes):
                y_true = label_boot[:, j]
                y_score = pred_boot[:, j]
                class_metrics = self._evaluate_single_class(y_true, y_score)
                per_class_boot[class_name] = class_metrics

            # 聚合为macro
            macro_boot = aggregate_metrics(per_class_boot, self.pathology_classes)

            # 记录macro指标
            for metric in self.metrics:
                key = f'macro_{metric}'
                if key in macro_boot and not np.isnan(macro_boot[key]):
                    bootstrap_results[metric].append(macro_boot[key])

        # 计算置信区间
        ci_results = {}
        alpha = 1 - self.bootstrap_ci
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        for metric in self.metrics:
            values = bootstrap_results[metric]
            if values:
                ci_lower = np.percentile(values, lower_percentile)
                ci_upper = np.percentile(values, upper_percentile)
                ci_results[f'macro_{metric}_ci_lower'] = float(ci_lower)
                ci_results[f'macro_{metric}_ci_upper'] = float(ci_upper)
                ci_results[f'macro_{metric}_ci_width'] = float(ci_upper - ci_lower)
            else:
                ci_results[f'macro_{metric}_ci_lower'] = np.nan
                ci_results[f'macro_{metric}_ci_upper'] = np.nan
                ci_results[f'macro_{metric}_ci_width'] = np.nan

        if verbose:
            print("Bootstrap CI computed.")

        return ci_results

    def format_results(self, results: Dict, indent: int = 0) -> str:
        """
        格式化结果为可读的字符串

        Args:
            results: evaluate()返回的结果
            indent: 缩进级别

        Returns:
            格式化的字符串
        """
        lines = []
        prefix = "  " * indent

        # Macro指标
        lines.append(f"{prefix}Macro Metrics:")
        for metric in self.metrics:
            key = f'macro_{metric}'
            if key in results:
                value = results[key]
                line = f"{prefix}  {metric.upper()}: {value:.4f}"

                # 如果有CI，添加
                if self.use_bootstrap:
                    ci_lower_key = f'{key}_ci_lower'
                    ci_upper_key = f'{key}_ci_upper'
                    if ci_lower_key in results and ci_upper_key in results:
                        ci_lower = results[ci_lower_key]
                        ci_upper = results[ci_upper_key]
                        line += f" (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"

                lines.append(line)

        # Per-class指标
        if 'per_class' in results:
            lines.append(f"\n{prefix}Per-Class Metrics:")
            for class_name, class_metrics in results['per_class'].items():
                lines.append(f"{prefix}  {class_name}:")
                for metric_name, value in class_metrics.items():
                    if not np.isnan(value):
                        lines.append(f"{prefix}    {metric_name}: {value:.4f}")

        return "\n".join(lines)
