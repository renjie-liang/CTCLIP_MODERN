"""
Unified disease detection evaluator

Features:
1. During training: Fast evaluation without bootstrap
2. During inference: Complete evaluation with bootstrap CI
3. Ensures training and inference use the same evaluation logic
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

from .metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1_optimal,
    compute_confusion_matrix_metrics,
    aggregate_metrics,
    aggregate_metrics_weighted
)


class DiseaseEvaluator:
    """
    Unified disease detection evaluator

    Usage example:
        # During training (fast)
        evaluator = DiseaseEvaluator(
            pathology_classes=['Atelectasis', 'Cardiomegaly'],
            metrics=['auroc', 'auprc', 'f1'],
            use_bootstrap=False
        )

        # During inference (complete)
        evaluator = DiseaseEvaluator(
            pathology_classes=['Atelectasis', 'Cardiomegaly'],
            metrics=['auroc', 'auprc', 'f1', 'precision', 'recall'],
            use_bootstrap=True,
            n_bootstrap=1000
        )

        # Evaluate
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
            pathology_classes: List of pathology classes
            metrics: List of metrics to compute
                Supported: 'auroc', 'auprc', 'f1', 'precision', 'recall', 'specificity'
            use_bootstrap: Whether to use bootstrap to compute confidence intervals
            n_bootstrap: Number of bootstrap samples
            bootstrap_ci: Confidence interval level (default: 0.95 = 95% CI)
            threshold: Binary classification threshold (default: 0.5)
            random_state: Random seed
        """
        self.pathology_classes = pathology_classes
        self.num_classes = len(pathology_classes)
        self.metrics = metrics
        self.use_bootstrap = use_bootstrap
        self.n_bootstrap = n_bootstrap
        self.bootstrap_ci = bootstrap_ci
        self.threshold = threshold
        self.random_state = random_state

        # Validate metrics
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
        Evaluate prediction results

        Args:
            predictions: Prediction scores (N, num_classes), range [0, 1]
            labels: True labels (N, num_classes), 0/1
            return_per_class: Whether to return detailed metrics for each class
            verbose: Whether to print detailed information

        Returns:
            Evaluation results dictionary containing:
            - macro_auroc, macro_auprc, ... (macro average)
            - If return_per_class=True:
                per_class: {
                    'Atelectasis': {'auroc': 0.85, 'auprc': 0.72, ...},
                    ...
                }
            - If use_bootstrap=True, also includes CI:
                macro_auroc_ci_lower, macro_auroc_ci_upper, ...
        """
        # Validate input
        assert predictions.shape == labels.shape, \
            f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        assert predictions.shape[1] == self.num_classes, \
            f"Number of classes mismatch: {predictions.shape[1]} vs {self.num_classes}"

        if verbose:
            print(f"Evaluating {predictions.shape[0]} samples across {self.num_classes} classes")
            print(f"Metrics: {self.metrics}")
            print(f"Bootstrap: {self.use_bootstrap}")

        # Compute metrics for each class
        per_class_metrics = {}
        support_dict = {}  # Track number of positive samples per class
        skipped_classes = []  # Track classes with single-class issue

        for i, class_name in enumerate(self.pathology_classes):
            y_true = labels[:, i]
            y_score = predictions[:, i]

            # Compute support (number of positive samples)
            support_dict[class_name] = int(np.sum(y_true))

            # Check if only one class present
            if len(np.unique(y_true)) < 2:
                skipped_classes.append(class_name)

            class_metrics = self._evaluate_single_class(y_true, y_score)
            per_class_metrics[class_name] = class_metrics

        # Print summary if classes were skipped
        if skipped_classes and verbose:
            print(f"⚠️  {len(skipped_classes)}/{self.num_classes} classes skipped due to single-class samples: {skipped_classes[:3]}{'...' if len(skipped_classes) > 3 else ''}")

        # Aggregate metrics (macro average)
        results = aggregate_metrics(per_class_metrics, self.pathology_classes)

        # Aggregate metrics (weighted average)
        weighted_results = aggregate_metrics_weighted(per_class_metrics, self.pathology_classes, support_dict)
        results.update(weighted_results)

        # If needed, add detailed information for each class
        if return_per_class:
            results['per_class'] = per_class_metrics

        # If using bootstrap, compute confidence intervals
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
        Evaluate a single class

        Args:
            y_true: True labels (N,)
            y_score: Prediction scores (N,)

        Returns:
            All metrics for this class
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

        # Metrics based on fixed threshold
        need_cm_metrics = any(m in self.metrics for m in ['precision', 'recall', 'specificity', 'accuracy'])

        if need_cm_metrics:
            y_pred = (y_score >= self.threshold).astype(int)
            cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)

            # Only add requested metrics
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
        Compute confidence intervals using bootstrap

        Compute CI for macro metrics

        Args:
            predictions: (N, num_classes)
            labels: (N, num_classes)
            verbose: Whether to show progress

        Returns:
            Dictionary containing CI: {
                'macro_auroc_ci_lower': float,
                'macro_auroc_ci_upper': float,
                ...
            }
        """
        if verbose:
            print(f"Computing bootstrap CI with {self.n_bootstrap} samples...")

        n_samples = predictions.shape[0]
        np.random.seed(self.random_state)

        # Store macro metrics for each bootstrap iteration
        bootstrap_results = {metric: [] for metric in self.metrics}

        for i in range(self.n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_boot = predictions[indices]
            label_boot = labels[indices]

            # Compute metrics for each class
            per_class_boot = {}
            for j, class_name in enumerate(self.pathology_classes):
                y_true = label_boot[:, j]
                y_score = pred_boot[:, j]
                class_metrics = self._evaluate_single_class(y_true, y_score)
                per_class_boot[class_name] = class_metrics

            # Aggregate to macro
            macro_boot = aggregate_metrics(per_class_boot, self.pathology_classes)

            # Record macro metrics
            for metric in self.metrics:
                key = f'macro_{metric}'
                if key in macro_boot and not np.isnan(macro_boot[key]):
                    bootstrap_results[metric].append(macro_boot[key])

        # Compute confidence intervals
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
        Format results as a readable string

        Args:
            results: Results returned by evaluate()
            indent: Indentation level

        Returns:
            Formatted string
        """
        lines = []
        prefix = "  " * indent

        # Macro metrics
        lines.append(f"{prefix}Macro Metrics:")
        for metric in self.metrics:
            key = f'macro_{metric}'
            if key in results:
                value = results[key]
                line = f"{prefix}  {metric.upper()}: {value:.4f}"

                # If CI is available, add it
                if self.use_bootstrap:
                    ci_lower_key = f'{key}_ci_lower'
                    ci_upper_key = f'{key}_ci_upper'
                    if ci_lower_key in results and ci_upper_key in results:
                        ci_lower = results[ci_lower_key]
                        ci_upper = results[ci_upper_key]
                        line += f" (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"

                lines.append(line)

        # Weighted metrics
        lines.append(f"\n{prefix}Weighted Metrics:")
        for metric in self.metrics:
            key = f'weighted_{metric}'
            if key in results:
                value = results[key]
                if not np.isnan(value):
                    lines.append(f"{prefix}  {metric.upper()}: {value:.4f}")

        # Per-class metrics
        if 'per_class' in results:
            lines.append(f"\n{prefix}Per-Class Metrics:")
            for class_name, class_metrics in results['per_class'].items():
                lines.append(f"{prefix}  {class_name}:")
                for metric_name, value in class_metrics.items():
                    if not np.isnan(value):
                        lines.append(f"{prefix}    {metric_name}: {value:.4f}")

        return "\n".join(lines)
