"""
Evaluation metric computation functions

Contains various evaluation metrics for disease detection:
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
    Compute AUROC (Area Under ROC Curve)

    Args:
        y_true: True labels (N,) 0/1
        y_score: Predicted scores (N,) between 0-1

    Returns:
        AUROC value, returns np.nan if cannot be computed
    """
    try:
        # Check if only one class is present
        if len(np.unique(y_true)) < 2:
            warnings.warn("Only one class present in y_true. AUROC is undefined.")
            return np.nan

        return roc_auc_score(y_true, y_score)
    except Exception as e:
        warnings.warn(f"Error computing AUROC: {e}")
        return np.nan


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUPRC (Area Under Precision-Recall Curve)

    Also known as Average Precision (AP)

    Args:
        y_true: True labels (N,) 0/1
        y_score: Predicted scores (N,) between 0-1

    Returns:
        AUPRC value, returns np.nan if cannot be computed
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
    Compute F1 Score at optimal threshold

    Iterates through all thresholds on the PR curve to find the maximum F1

    Args:
        y_true: True labels (N,) 0/1
        y_score: Predicted scores (N,) between 0-1

    Returns:
        (optimal_f1, optimal_threshold)
    """
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan

        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # Compute F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        # Find maximum F1
        optimal_idx = np.argmax(f1_scores)
        optimal_f1 = f1_scores[optimal_idx]

        # precision_recall_curve returns one fewer threshold than precision
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
    Compute various metrics based on confusion matrix

    Args:
        y_true: True labels (N,) 0/1
        y_pred: Predicted labels (N,) 0/1 (already binarized)

    Returns:
        dict containing:
        - precision: Precision TP / (TP + FP)
        - recall: Recall TP / (TP + FN)  (sensitivity)
        - specificity: Specificity TN / (TN + FP)
        - f1: F1 Score
        - accuracy: Accuracy
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

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Compute each metric
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
    Compute Youden's Index (J = sensitivity + specificity - 1)

    Find the threshold that maximizes J

    Args:
        y_true: True labels (N,) 0/1
        y_score: Predicted scores (N,) between 0-1

    Returns:
        (optimal_youden_index, optimal_threshold)
    """
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan, np.nan

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Youden's Index = TPR - FPR = sensitivity - (1 - specificity) = sensitivity + specificity - 1
        youden_index = tpr - fpr

        # Find maximum value
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
    Aggregate metrics across classes

    Compute macro average (each class has equal weight)

    Args:
        per_class_metrics: {class_name: {metric_name: value}}
        classes: List of class names

    Returns:
        {'macro_auroc': float, 'macro_auprc': float, ...}
    """
    aggregated = {}

    # Extract all metric names
    if not per_class_metrics or not classes:
        return {}

    first_class = classes[0]
    metric_names = list(per_class_metrics[first_class].keys())

    # Compute macro average for each metric
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


def aggregate_metrics_weighted(
    per_class_metrics: Dict[str, Dict[str, float]],
    classes: list,
    support: Dict[str, int]
) -> Dict[str, float]:
    """
    Aggregate metrics across classes with weighted average

    Compute weighted average (classes weighted by sample count)

    Args:
        per_class_metrics: {class_name: {metric_name: value}}
        classes: List of class names
        support: {class_name: sample_count} - number of positive samples per class

    Returns:
        {'weighted_auroc': float, 'weighted_auprc': float, ...}
    """
    aggregated = {}

    # Extract all metric names
    if not per_class_metrics or not classes:
        return {}

    first_class = classes[0]
    metric_names = list(per_class_metrics[first_class].keys())

    # Compute total support
    total_support = sum(support.get(class_name, 0) for class_name in classes)

    if total_support == 0:
        # If no support information, fall back to macro average
        for metric_name in metric_names:
            aggregated[f'weighted_{metric_name}'] = np.nan
        return aggregated

    # Compute weighted average for each metric
    for metric_name in metric_names:
        weighted_sum = 0.0
        valid_support = 0

        for class_name in classes:
            if class_name in per_class_metrics and class_name in support:
                value = per_class_metrics[class_name].get(metric_name, np.nan)
                class_support = support[class_name]

                if not np.isnan(value) and class_support > 0:
                    weighted_sum += value * class_support
                    valid_support += class_support

        if valid_support > 0:
            aggregated[f'weighted_{metric_name}'] = float(weighted_sum / valid_support)
        else:
            aggregated[f'weighted_{metric_name}'] = np.nan

    return aggregated
