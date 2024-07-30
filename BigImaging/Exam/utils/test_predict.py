import torchvision
from tqdm import tqdm
import torch
from torch.amp.autocast_mode import autocast

from utils.train import reshape_imgs_masks
from utils.metrics import compute_metrics_torch

from torch.cuda import is_available

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor
from typing import Tuple

def test(testloader: DataLoader, 
         model: Module, 
         criterion: Module, 
         num_classes: int, 
         device: str = 'cuda' if is_available() else 'cpu'
        ) -> Tuple[Tensor, Tensor]:
    """
    Perform testing on the given testloader using the provided model and criterion.

    Args:
        testloader (torch.utils.data.DataLoader): The data loader for the test dataset.
        model (torch.nn.Module): The model to be used for testing.
        criterion (torch.nn.Module): The loss criterion to be used for calculating the loss.
        num_classes (int): The number of classes in the dataset.
        device (str, optional): The device to be used for testing. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        tuple: A tuple containing the flattened true labels and predicted labels.

    """
    model.eval()
    test_loss = torch.tensor(0.0)
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(testloader, desc=f'Test'):
            imgs, masks = reshape_imgs_masks(imgs, masks)

            with autocast(device):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                test_loss += loss.detach()

            preds = torch.argmax(outputs, dim=1)
            all_y_true.append(masks)
            all_y_pred.append(preds)

    all_y_true_flattened = torch.cat(all_y_true, dim=0)
    all_y_pred_flattened = torch.cat(all_y_pred, dim=0)

    print(f'All y true shape: {all_y_true_flattened.shape}, All y pred shape: {all_y_pred_flattened.shape}')

    metrics = compute_metrics_torch(all_y_true_flattened, all_y_pred_flattened, num_classes)

    print(f'Test Loss: {test_loss.item()/len(testloader):.3f}, Mean IoU: {metrics["mean_iou"]:.3f}, '
        f'Accuracy: {metrics["accuracy"]:.3f}, Dice Score: {metrics["mean_dice"]:.3f}, '
        f'per-class IoU: {[f"Class {i}: {iou:.3f}" for i, iou in enumerate(metrics["per_class_iou"])]}')
    
    return all_y_true_flattened, all_y_pred_flattened

def convert_to_rgb(masks: Tensor,
                   colors: Tensor, 
                   device: str = 'cuda' if is_available() else 'cpu'
                   ) -> Tensor:
    """
    Convert a 4D tensor of masks to a 5D tensor of RGB masks.

    Args:
        masks (Tensor): A 4D tensor of masks with shape [batch_size, num_tiles, height, width].
        colors (Tensor): A tensor of RGB color values for each class.
        device (str, optional): The device to be used. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        Tensor: A 5D tensor of RGB masks with shape [batch_size, num_tiles, 3, height, width].
    """
    batch_size, num_tiles, height, width = masks.shape
    masks_rgb = torch.zeros((batch_size, num_tiles, 3, height, width), dtype=torch.uint8).to(device)

    for i, color in enumerate(colors):
        color_tensor = torch.tensor(color, dtype=torch.uint8).view(1, 1, 3, 1, 1).to(device)
        masks_rgb += (masks == i).unsqueeze(2) * color_tensor

    return masks_rgb
    
def visualize_predictions(true_masks, pred_masks, colors, dims=(256, 256), images_to_visualize=3, batch_size=1):
    '''
    Visualize the predictions of a model.

    Args:
        true_masks (torch.Tensor): Ground truth masks of shape [batch_size * num_tiles, 256, 256].
        pred_masks (torch.Tensor): Predicted masks of shape [batch_size * num_tiles, 256, 256].
        dims (tuple): Dimensions of the images (default: (256, 256)).
        images_to_visualize (int): Number of images to visualize (default: 3).
        batch_size (int): Batch size of the test loader (default: 1).
    '''
    # Compute nr of images per row
    img_per_col = 4000 // dims[0]
    imgs_per_row = 6000 // dims[0]
    num_tiles = img_per_col * imgs_per_row

    # Clip the number of images to visualize
    num_images = images_to_visualize * num_tiles * batch_size
    true_masks = true_masks[:num_images]
    pred_masks = pred_masks[:num_images]
    
    # Resizing the masks to dims
    true_masks = F.interpolate(true_masks.unsqueeze(1).float(), size=dims, mode='nearest').squeeze(1).to(torch.uint8)
    pred_masks = F.interpolate(pred_masks.unsqueeze(1).float(), size=dims, mode='nearest').squeeze(1).to(torch.uint8)

    # Reshape masks for batching
    # [batch_size * num_tiles, 256, 256] -> [batch_size, num_tiles, dims[0], dims[1]]
    true_masks = true_masks.view(batch_size, -1, *true_masks.shape[1:])
    pred_masks = pred_masks.view(batch_size, -1, *pred_masks.shape[1:])

    # Convert masks to RGB
    true_masks_rgb = convert_to_rgb(true_masks, colors)
    pred_masks_rgb = convert_to_rgb(pred_masks, colors)

    for i in range(batch_size):
        # Create a grid of predictions
        pred_grid = torchvision.utils.make_grid(pred_masks_rgb[i], nrow=imgs_per_row, normalize=False, pad_value=1)
        # Create grid of true masks
        true_grid = torchvision.utils.make_grid(true_masks_rgb[i], nrow=imgs_per_row, normalize=False, pad_value=1)
        
        # Display the grids side by side
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display the true masks grid
        axes[0].imshow(true_grid.permute(1, 2, 0).cpu().numpy()) # type: ignore
        axes[0].axis('off')  # type: ignore
        axes[0].set_title('True Masks') # type: ignore
        
        # Display the predicted masks grid
        axes[1].imshow(pred_grid.permute(1, 2, 0).cpu().numpy()) # type: ignore
        axes[1].axis('off') # type: ignore
        axes[1].set_title('Predicted Masks') # type: ignore
        
        plt.show()