import torch
import segmentation_models_pytorch as smp
from utils.train import validate
from utils.metrics import compute_metrics_torch
from tqdm import tqdm
import torch
from typing import Any, Dict, Tuple, List
from torch import nn
from torch.utils.data import DataLoader

def load_model_from_checkpoint(
        checkpoint_path: str, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[nn.Module, Dict[str, Any], int, int]:
    """
    Load a model from a checkpoint file.
    Args:
        checkpoint_path (str): The path to the checkpoint file.
        device (str, optional): The device to load the model on. Defaults to 'cuda' if available, else 'cpu'.
    Returns:
        Tuple[nn.Module, Dict[str, Any], Tuple[int, int], Tuple[int, int]]: A tuple containing the loaded model, the model configuration, the tiles dimension, and the final dimension.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    tiles_dim = checkpoint['tiles_dim']
    final_dim = checkpoint['final_dim']

    # Create the model
    model = smp.create_model(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # print(f'Created model with config {config} and loaded weights from {checkpoint_path}')

    return model, config, tiles_dim, final_dim


def evaluate_model(
        num_classes: int, 
        model: Any, 
        test_loader: DataLoader, 
        criterion: Any, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        decimal_places: int = 5) -> Tuple[float, float, float, List[float]]:
    """
    Evaluate a model and compute metrics.
    Args:
        num_classes (int): The number of classes.
        model (Any): The model to evaluate.
        test_loader (DataLoader): The data loader for the test dataset.
        criterion (Any): The loss criterion.
        device (str, optional): The device to use for evaluation. Defaults to 'cuda' if available, else 'cpu'.
        decimal_places (int, optional): The number of decimal places to round the metrics to. Defaults to 5.
    Returns:
        Tuple[float, float, float, List[float]]: A tuple containing the mean IoU, accuracy, mean dice, and per-class IoU.
        """
    model.eval()
    test_loss = torch.tensor(0.0).to(device)
    total_iou = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    num_batches = 0
    per_class_iou_accumulators = [0.0] * num_classes
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc='Testing', ncols=100):
            test_loss, preds = validate(test_loss, imgs, masks, model, criterion, use_amp=True)
            batch_metrics = compute_metrics_torch(masks.to(device), preds.to(device), num_classes, device)
            
            total_iou += batch_metrics["weighted_mean_iou"]
            total_accuracy += batch_metrics["accuracy"]
            total_dice += batch_metrics["weighted_mean_dice"]
            for cls in range(num_classes):
                per_class_iou_accumulators[cls] += batch_metrics["per_class_iou"][cls]
            num_batches += 1

    mean_iou = round(total_iou / num_batches, decimal_places)
    accuracy = round(total_accuracy / num_batches, decimal_places)
    mean_dice = round(total_dice / num_batches, decimal_places)
    per_class_iou = [round(iou / num_batches, decimal_places) for iou in per_class_iou_accumulators]

    return mean_iou, accuracy, mean_dice, per_class_iou