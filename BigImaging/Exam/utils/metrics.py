import numpy as np
from torch import Tensor
from typing import Dict, List, Literal, Union
from torch.cuda import is_available

def compute_metrics_torch(
    y_true: Tensor,
    y_pred: Tensor,
    num_classes: int,
    device = 'cuda' if is_available() else 'cpu'
) -> Dict:
    """
    Compute evaluation metrics for semantic segmentation using PyTorch tensors.

    Args:
        y_true (torch.Tensor): Ground truth segmentation mask tensor.
        y_pred (torch.Tensor): Predicted segmentation mask tensor.
        num_classes (int): Number of classes in the segmentation task.
        device (torch.device): Device to perform the computation on.

    Returns:
        dict: A dictionary containing the computed metrics.
            - 'weighted_mean_iou' (float): Weighted Mean Intersection over Union (IoU) score.
            - 'per_class_iou' (list): IoU score for each class.
            - 'accuracy' (float): Overall accuracy.
            - 'weighted_mean_dice' (float): Weighted Mean Dice score.
            - 'per_class_dice' (list): Dice score for each class.
    """
    
    # Move arrays to GPU
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)

    # Flatten the arrays for metric computation
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)

    # Compute overall accuracy
    acc = (y_true_flat == y_pred_flat).float().mean().item()

    # Initialize lists to store per-class metrics
    iou_list = []
    dice_list = []
    class_frequencies = []

    for cls in range(num_classes):
        # Compute intersection and union for IoU
        intersection = ((y_true_flat == cls) & (y_pred_flat == cls)).float().sum().item()
        union = ((y_true_flat == cls) | (y_pred_flat == cls)).float().sum().item()
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)

        # Compute intersection and total for Dice
        dice_intersection = 2 * intersection
        total = (y_true_flat == cls).float().sum().item() + (y_pred_flat == cls).float().sum().item()
        dice = dice_intersection / total if total != 0 else 0
        dice_list.append(dice)

        # Compute class frequency
        class_frequency = (y_true_flat == cls).float().sum().item()
        class_frequencies.append(class_frequency)

    # Compute total number of pixels
    total_pixels = y_true_flat.size(0)
    class_weights = [freq / total_pixels for freq in class_frequencies]

    # Compute weighted mean IoU
    weighted_mean_iou = np.sum([iou * weight for iou, weight in zip(iou_list, class_weights)])

    # Compute weighted mean Dice
    weighted_mean_dice = np.sum([dice * weight for dice, weight in zip(dice_list, class_weights)])

    # Return the metrics
    return {
        'weighted_mean_iou': weighted_mean_iou,
        'per_class_iou': iou_list,
        'accuracy': acc,
        'weighted_mean_dice': weighted_mean_dice,
        'per_class_dice': dice_list
    }