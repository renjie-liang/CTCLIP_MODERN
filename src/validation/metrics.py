"""
评估指标计算函数

包含疾病检测的各种评估指标：
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- F1 Score (optimal threshold)
- Precision, Recall, Specificity
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
import warnings

# Suppress repeated warnings for single-class issues
warnings.filterwarnings('ignore', message='Only one class present in y_true')


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算AUROC (Area Under ROC Curve)

    Args:
        y_true: 真实标签 (N,) 0/1
        y_score: 预测分数 (N,) 0-1之间

    Returns:
        AUROC值，如果无法计算返回np.nan
    """
    try:
        # 检查是否只有一个类别
        if len(np.unique(y_true)) < 2:
            warnings.warn("Only one class present in y_true. AUROC is undefined.")
            return np.nan

        return roc_auc_score(y_true, y_score)
    except Exception as e:
        warnings.warn(f"Error computing AUROC: {e}")
        return np.nan


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算AUPRC (Area Under Precision-Recall Curve)

    也称为Average Precision (AP)

    Args:
        y_true: 真实标签 (N,) 0/1
        y_score: 预测分数 (N,) 0-1之间

    Returns:
        AUPRC值，如果无法计算返回np.nan
    """
    try:
        if len(np.unique(y_true)) < 2:
            warnings.warn("Only one class present in y_true. AUPRC is undefined.")
            return np.nan

        return average_precision_score(y_true, y_score)
    except Exception as e:
        warnings.warn(f"Error computing AUPRC: {e}")
        return np.nan


def compute_f1_optimal(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    计算最优阈值下的F1 Score

    遍历PR曲线上的所有阈值，找到F1最大的点

    Args:
        y_true: 真实标签 (N,) 0/1
        y_score: 预测分数 (N,) 0-1之间

    Returns:
        (optimal_f1, optimal_threshold)
    """
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan

        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # 计算每个阈值的F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        # 找最大F1
        optimal_idx = np.argmax(f1_scores)
        optimal_f1 = f1_scores[optimal_idx]

        # precision_recall_curve返回的thresholds比precision少一个
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 1.0

        return float(optimal_f1), float(optimal_threshold)
    except Exception as e:
        warnings.warn(f"Error computing optimal F1: {e}")
        return np.nan, np.nan


def compute_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    基于混淆矩阵计算各种指标

    Args:
        y_true: 真实标签 (N,) 0/1
        y_pred: 预测标签 (N,) 0/1 (已二值化)

    Returns:
        dict包含:
        - precision: 精确率 TP / (TP + FP)
        - recall: 召回率 TP / (TP + FN)  (sensitivity)
        - specificity: 特异性 TN / (TN + FP)
        - f1: F1 Score
        - accuracy: 准确率
    """
    try:
        if len(np.unique(y_true)) < 2:
            return {
                'precision': np.nan,
                'recall': np.nan,
                'specificity': np.nan,
                'f1': np.nan,
                'accuracy': np.nan
            }

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # 计算各项指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    except Exception as e:
        warnings.warn(f"Error computing confusion matrix metrics: {e}")
        return {
            'precision': np.nan,
            'recall': np.nan,
            'specificity': np.nan,
            'f1': np.nan,
            'accuracy': np.nan
        }


def compute_youden_index(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    计算Youden's Index (J = sensitivity + specificity - 1)

    找到最大化J的阈值

    Args:
        y_true: 真实标签 (N,) 0/1
        y_score: 预测分数 (N,) 0-1之间

    Returns:
        (optimal_youden_index, optimal_threshold)
    """
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Youden's Index = TPR - FPR = sensitivity - (1 - specificity) = sensitivity + specificity - 1
        youden_index = tpr - fpr

        # 找最大值
        optimal_idx = np.argmax(youden_index)
        optimal_j = youden_index[optimal_idx]
        optimal_threshold = thresholds[optimal_idx]

        return float(optimal_j), float(optimal_threshold)
    except Exception as e:
        warnings.warn(f"Error computing Youden index: {e}")
        return np.nan, np.nan


def aggregate_metrics(
    per_class_metrics: Dict[str, Dict[str, float]],
    classes: list
) -> Dict[str, float]:
    """
    聚合每个类别的指标

    计算macro average (每个类别权重相同)

    Args:
        per_class_metrics: {class_name: {metric_name: value}}
        classes: 类别列表

    Returns:
        {'macro_auroc': float, 'macro_auprc': float, ...}
    """
    aggregated = {}

    # 提取所有指标名称
    if not per_class_metrics or not classes:
        return {}

    first_class = classes[0]
    metric_names = list(per_class_metrics[first_class].keys())

    # 对每个指标计算macro average
    for metric_name in metric_names:
        values = []
        for class_name in classes:
            if class_name in per_class_metrics:
                value = per_class_metrics[class_name].get(metric_name, np.nan)
                if not np.isnan(value):
                    values.append(value)

        if values:
            aggregated[f'macro_{metric_name}'] = float(np.mean(values))
        else:
            aggregated[f'macro_{metric_name}'] = np.nan

    return aggregated
