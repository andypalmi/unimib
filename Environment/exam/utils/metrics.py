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
            - 'mean_iou' (float): Mean Intersection over Union (IoU) score.
            - 'per_class_iou' (list): IoU score for each class.
            - 'accuracy' (float): Overall accuracy.
            - 'mean_dice' (float): Mean Dice score.
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

    # Helper function to compute IoU for a single class
    def compute_iou(cls):
        intersection = ((y_true_flat == cls) & (y_pred_flat == cls)).float().sum().item()
        union = ((y_true_flat == cls) | (y_pred_flat == cls)).float().sum().item()
        return intersection / union if union != 0 else 0

    # Helper function to compute Dice score for a single class
    def compute_dice(cls):
        intersection = 2 * ((y_true_flat == cls) & (y_pred_flat == cls)).float().sum().item()
        total = (y_true_flat == cls).float().sum().item() + (y_pred_flat == cls).float().sum().item()
        return intersection / total if total != 0 else 0

    # Compute IoU
    iou_list = [compute_iou(cls) for cls in range(num_classes)]
    mean_iou = np.mean(iou_list)

    # Compute Dice
    dice_list = [compute_dice(cls) for cls in range(num_classes)]
    mean_dice = np.mean(dice_list)

    # Return the metrics
    return {
        'mean_iou': mean_iou,
        'per_class_iou': iou_list,
        'accuracy': acc,
        'mean_dice': mean_dice,
        'per_class_dice': dice_list
    }