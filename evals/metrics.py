from typing import List
import numpy as np
from sklearn.metrics import f1_score

def compute_accuracy(predictions: List[int], true_labels: List[int]) -> float:
    """
    Compute accuracy (%) for QA predictions.

    Args:
        predictions (List[int]): Predicted labels.
        true_labels (List[int]): True labels.

    Returns:
        float: Accuracy percentage.
    """
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    return (correct / len(true_labels)) * 100 if true_labels else 0.0

def compute_drift(accuracies: List[float]) -> float:
    """
    Compute performance drift (%) as the absolute difference in accuracy between consecutive batches.

    Args:
        accuracies (List[float]): List of accuracy values for each batch.

    Returns:
        float: Average drift percentage.
    """
    if len(accuracies) < 2:
        return 0.0
    drifts = [abs(accuracies[i] - accuracies[i-1]) for i in range(1, len(accuracies))]
    return np.mean(drifts)

def compute_f1(predictions: List[int], true_labels: List[int]) -> float:
    """
    Compute macro F1 score (for completeness, not used in Test 2).

    Args:
        predictions (List[int]): Predicted labels.
        true_labels (List[int]): True labels.

    Returns:
        float: Macro F1 score.
    """
    return f1_score(true_labels, predictions, average='macro') * 100 if true_labels else 0.0

def compute_causal_accuracy(predictions: List[int], true_labels: List[int], causal_pairs: List[Tuple[int, int]]) -> float:
    """
    Compute causal accuracy (for completeness, not used in Test 2).

    Args:
        predictions (List[int]): Predicted labels.
        true_labels (List[int]): True labels.
        causal_pairs (List[Tuple[int, int]]): List of (cause, effect) label pairs.

    Returns:
        float: Percentage of correct predictions for causal relationships.
    """
    correct = 0
    total = 0
    for pred, true, (cause, effect) in zip(predictions, true_labels, causal_pairs):
        if (pred == cause and true == effect) or (pred == effect and true == cause):
            correct += 1
        total += 1
    return (correct / total) * 100 if total else 0.0
