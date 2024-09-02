import pandas as pd
import numpy as np
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torchvision.transforms import transforms
from utils.evaluate import load_model_from_checkpoint

def read_class_colors(
    file_path: str,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Reads the class colors from a CSV file and returns the dataframe, 
    the RGB colors, and the number of classes.

    Parameters:
    file_path (str): The path to the CSV file containing the class colors.
    verbose (bool, optional): Whether to print the number of classes. Defaults to True.

    Returns:
    tuple: A tuple containing the dataframe, the RGB colors as a numpy array, 
           and the number of classes.
    """
    # Read number of classes
    labels_colors = pd.read_csv(file_path)
    columns = ['class', 'r', 'g', 'b']
    labels_colors.columns = columns

    # Extract RGB values
    labels_colors['RGB'] = labels_colors[['r', 'g', 'b']].apply(tuple, axis=1)
    colors = np.array(labels_colors['RGB'].values)

    # No conflicting labels were found during anaylysis, therefore there are 23 classes in the dataset
    labels_colors = labels_colors[labels_colors['class'] != 'conflicting']
    nr_classes = len(labels_colors)
    if verbose:
        print(f'Number of classes: {nr_classes}')

    return labels_colors, colors, nr_classes

def save_profiling_tables(prof, logs_dir):
    """
    Save CPU and CUDA time tables from the profiler to text files.

    Args:
        prof: The profiler object containing the profiling data.
        logs_dir: The directory where the log files will be saved.
    """
    cpu_time_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    cuda_time_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

    # Save CPU time table to a text file
    with open(f'{logs_dir}/cpu_time_total.txt', 'w') as f:
        f.write(cpu_time_table)
        print(f'CPU time table saved to {logs_dir}/cpu_time_total.txt')

    # Save CUDA time table to a text file
    with open(f'{logs_dir}/cuda_time_total.txt', 'w') as f:
        f.write(cuda_time_table)
        print(f'CUDA time table saved to {logs_dir}/cuda_time_total.txt')

def load_tiles(
    image_number,
    path,
    tiles_dim=512,
    width=4,
    height=3
):
    """
    Load image and mask tiles for a given image number.

    Args:
        image_number: The number of the image.
        path: The path to the image and mask tiles.
        tiles_dim: The original size of the image tiles. Default is 512.
        width: The number of columns of tiles. Default is 4.
        height: The number of rows of tiles. Default is 3.

    Returns:
        tuple: A tuple containing arrays of image tiles and mask tiles.
    """
    width = 6000 // tiles_dim
    height = 4000 // tiles_dim

    image_tiles = []
    mask_tiles = []
    for i in range(width * height):
        image_path = os.path.join(path, 'images', f'{image_number}_{i}.png')
        mask_path = os.path.join(path, 'masks', f'{image_number}_{i}.png')
        image_tiles.append(np.array(Image.open(image_path)))
        mask_tiles.append(np.array(Image.open(mask_path)))
    return np.array(image_tiles), np.array(mask_tiles)

def reconstruct_image(
    tiles,
    tiles_dim=512,
    final_dim=256,
    width=4,
    height=3,
    channels=3
):
    """
    Reconstruct an image from its tiles.

    Args:
        tiles: The array of image tiles.
        tiles_dim: The original size of the image tiles. Default is 512.
        final_dim: The actual (px) size of each tile. Default is 256.
        width: The number of columns of tiles. Default is 4.
        height: The number of rows of tiles. Default is 3.
        channels: The number of channels in the image. Default is 3 (RGB).

    Returns:
        np.ndarray: The reconstructed image.
    """

    width = 6000 // tiles_dim
    height = 4000 // tiles_dim

    if channels == 3:
        image = np.zeros((height * final_dim, width * final_dim, channels), dtype=tiles[0].dtype)
    else:
        image = np.zeros((height * final_dim, width * final_dim), dtype=tiles[0].dtype)
    
    for i in range(height):
        for j in range(width):
            image[i*final_dim:(i+1)*final_dim, j*final_dim:(j+1)*final_dim] = tiles[i * width + j]
    
    return image

def colorize_mask(
    mask,
    colors
):
    """
    Colorize a mask using the provided colors.

    Args:
        mask: The mask to be colorized.
        colors: The list of colors for each class.

    Returns:
        np.ndarray: The colorized mask.
    """
    colorized_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    for i, color in enumerate(colors):
        colorized_mask[mask == i] = color
    return colorized_mask

import matplotlib.patches as mpatches

def plot_different_sizes(
    model_paths,
    model_name,
    image_number,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Plot the models predictions for different tile sizes in a single row.

    Args:
        model_paths (list): List of models to plot predictions for.
        model_name (str): The name of the model.
        image_number (int): The image number.
        path_to_tiles (str): The path to the image tiles.
        tile_sizes (list): List of tile sizes to plot.
        device (str, optional): The device to use for prediction. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        None
    """

    diff_masks = []
    tile_sizes = []

    for model_path in model_paths:
            model_path = os.path.join('models/best', model_path)
            if model_name in model_path:
                model, config, model_tiles_dim, model_final_dim = load_model_from_checkpoint(model_path, verbose=False)
                print(f'Model tiles dim: {model_tiles_dim}, Model final dim: {model_final_dim}')
                model.eval()
                path_to_tiles = f'data/tiles_256/{model_tiles_dim}x{model_tiles_dim}'
                image_tiles, mask_tiles = load_tiles(image_number, path_to_tiles, model_tiles_dim)
                                                    
                pred_mask_tiles, _ = get_pred_prob_tiles(model, image_tiles, device=device)

                pred_mask = reconstruct_image(np.array(pred_mask_tiles), tiles_dim=model_tiles_dim, final_dim=model_final_dim, channels=1)

                true_mask = reconstruct_image(mask_tiles, tiles_dim=model_tiles_dim, channels=3)
                diff_mask = (true_mask[:, :, 0] != pred_mask).astype(np.uint8) * 255
                diff_mask = (diff_mask - np.min(diff_mask)) / (np.max(diff_mask) - np.min(diff_mask))
                diff_masks.append(diff_mask)
                tile_sizes.append(model_tiles_dim)


    fig, axes = plt.subplots(1, len(tile_sizes), figsize=(15, 5), dpi=300)
    for i, (diff_mask, tiles_size) in enumerate(zip(diff_masks, tile_sizes)):
        axes[i].imshow(diff_mask)
        axes[i].set_title(f'{tiles_size} px')
        axes[i].axis('off')

    # Add text with encoder name
    encoder_name = model_name
    encoder_name = 'ResNet34' if encoder_name == 'resnet34' else 'EfficientNet-B5' if encoder_name == 'efficientnet-b5' else 'MobileNetV2' if encoder_name == 'mobilenet_v2' else encoder_name
    fig.text(0.5, 0.04, f"{encoder_name}", ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    plt.savefig(f"output/different_sizes_{image_number}_{encoder_name}.png", bbox_inches='tight')
    plt.close()
    
def plot_grid(
    image_number,
    config,
    original_img,
    true_mask,
    pred_mask,
    pred_probs,
    overlaid_img,
    diff_mask,
    class_df,
    tiles_dim=512
):
    """
    Plot a grid of images including the original image, true mask, predicted mask, 
    model confidence, overlaid image, and difference mask.

    Args:
        image_number: The number of the image.
        config: The configuration object.
        original_img: The original image.
        true_mask: The ground truth mask.
        pred_mask: The predicted mask.
        pred_probs: The model confidence probabilities.
        overlaid_img: The image with the predicted mask overlaid.
        diff_mask: The difference mask between the true and predicted masks.
        class_df: The dataframe containing the class information.
        tiles_dim: The original size of the image tiles. Default is 512.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    
    axes[0, 1].imshow(true_mask)
    axes[0, 1].set_title('Ground Truth Mask')
    
    axes[0, 2].imshow(pred_mask)
    axes[0, 2].set_title('Predicted Mask')
    
    im_pred_probs = axes[1, 0].imshow(pred_probs, cmap='afmhot')
    axes[1, 0].set_title('Model Confidence')
    fig.colorbar(im_pred_probs, ax=axes[1, 0], location='bottom', shrink=0.5)
    
    axes[1, 1].imshow(overlaid_img)
    axes[1, 1].set_title('Mask Overlaid on Image')
    
    im_diff_mask = axes[1, 2].imshow(diff_mask, cmap='gray')
    axes[1, 2].set_title('Difference Mask')
    fig.colorbar(im_diff_mask, ax=axes[1, 2], location='bottom', shrink=0.5)
    
    for ax in axes.flatten():
        ax.axis('off')

    legend_handles = []
    for _, row in class_df.iterrows():
        normalized_rgb = tuple(c / 255.0 for c in row['RGB'])
        patch = mpatches.Patch(color=normalized_rgb, label=row['class'])
        legend_handles.append(patch)

    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Classes")

    # Add text with encoder name and tiles dimension
    encoder_name = config['encoder_name']
    encoder_name = 'ResNet34' if encoder_name == 'resnet34' else 'EfficientNet-B5' if encoder_name == 'efficientnet-b5' else 'MobileNetV2' if encoder_name == 'mobilenet_v2' else encoder_name
    fig.text(0.5, 0.04, f"{encoder_name} @ {tiles_dim} px", ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    plt.savefig(f"output/grid_{image_number}_@{tiles_dim}px_{config['encoder_name']}_with_legend.png", bbox_inches='tight')
    plt.close()

def get_pred_prob_tiles(
    model,
    image_tiles,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    model.to(device)

    pred_mask_tiles, pred_prob_tiles = [], []
    for tile in image_tiles:
        tile_tensor = torch.tensor(tile).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
        tile_tensor = normalize(tile_tensor)
        with torch.no_grad():
            output = model(tile_tensor)
            pred_probs = torch.softmax(output, dim=1).squeeze(0)
            pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy()
            pred_mask_tiles.append(pred_mask)
            max_probs = pred_probs.max(0)[0].cpu().numpy()
            pred_prob_tiles.append(max_probs)
    
    return pred_mask_tiles, pred_prob_tiles

def predict_and_plot_grid(
    model,
    config,
    tiles_dim,
    image_number,
    path_to_tiles,
    colors,
    classes_df,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    tile_size=256
):
    """
    Predicts and plots a grid of images using a given model.

    Args:
        model: The model used for prediction.
        config: The configuration object.
        image_number: The number of the image.
        path_to_tiles: The path to the image tiles.
        colors: The list of colors for mask visualization.
        classes_df: The dataframe containing the classes information.
        device: The device to use for prediction. Defaults to 'cuda' if available, else 'cpu'.
        tile_size: The size of the image tiles. Defaults to 256.
    """
    model.eval()
    image_tiles, mask_tiles = load_tiles(image_number, path_to_tiles, tiles_dim)
    
    original_img = reconstruct_image(image_tiles, tiles_dim=tiles_dim, channels=3)
    true_mask = reconstruct_image(mask_tiles, tiles_dim=tiles_dim, channels=3)
    
    colorized_true_mask = colorize_mask(true_mask, colors)

    pred_mask_tiles, pred_prob_tiles = get_pred_prob_tiles(model, image_tiles, device=device)

    pred_mask = reconstruct_image(np.array(pred_mask_tiles), tiles_dim=tiles_dim, final_dim=tile_size, channels=1)
    pred_probs = reconstruct_image(np.array(pred_prob_tiles), tiles_dim=tiles_dim, final_dim=tile_size, channels=1) * 255
    pred_probs = (pred_probs - np.min(pred_probs)) / (np.max(pred_probs) - np.min(pred_probs))
    
    colorized_pred_mask = colorize_mask(pred_mask, colors)

    alpha = 0.6
    overlaid_img = cv2.addWeighted(original_img, 1-alpha, colorized_pred_mask, alpha, 0)

    diff_mask = (true_mask[:, :, 0] != pred_mask).astype(np.uint8) * 255
    diff_mask = (diff_mask - np.min(diff_mask)) / (np.max(diff_mask) - np.min(diff_mask)) 

    plot_grid(image_number, config, original_img, colorized_true_mask, colorized_pred_mask, pred_probs, overlaid_img, diff_mask, classes_df, tiles_dim=tiles_dim)