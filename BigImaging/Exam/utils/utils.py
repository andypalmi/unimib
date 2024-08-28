import pandas as pd
import numpy as np
import glob
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import torch

def create_splits(final_dim, tiles_dim, base_path='data', verbose=True):
    """
    Create train, validation, and test splits based on the images and masks present in the specified path.

    Args:
        final_dim (int): The final dimension of the tiles.
        tiles_dim (int): The dimension of the tiles.
        base_path (str): The base path where the data is stored. Default is 'data'.

    Returns:
        tuple: A tuple containing train_split, val_split, and test_split.
    """
    tiles_path = f'{base_path}/tiles_{final_dim}/{tiles_dim}x{tiles_dim}'
    splits = ['train', 'val', 'test']
    all_paths = {'train': [], 'val': [], 'test': []}

    for split in splits:
        img_paths = sorted(glob.glob(f'{tiles_path}/{split}/images/*.png'))
        mask_paths = sorted(glob.glob(f'{tiles_path}/{split}/masks/*.png'))
        all_paths[split] = np.array(list(zip(img_paths, mask_paths))).tolist()

    train_split = all_paths['train']
    val_split = all_paths['val']
    test_split = all_paths['test']

    print(f'Train split: {len(train_split)} samples')
    print(f'Validation split: {len(val_split)} samples')
    print(f'Test split: {len(test_split)} samples')

    return train_split, val_split, test_split

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

def load_tiles(image_number, path, rows=3, cols=4):
    image_tiles = []
    mask_tiles = []
    for i in range(rows * cols):
        image_path = os.path.join(path, 'images', f'{image_number}_{i}.png')
        mask_path = os.path.join(path, 'masks', f'{image_number}_{i}.png')
        image_tiles.append(np.array(Image.open(image_path)))
        mask_tiles.append(np.array(Image.open(mask_path)))
    return np.array(image_tiles), np.array(mask_tiles)

def reconstruct_image(tiles, rows=3, cols=4, tile_size=256, channels=3):
    if channels == 3:  # RGB image
        image = np.zeros((rows * tile_size, cols * tile_size, channels), dtype=np.uint8)
    else:  # Grayscale image (single channel)
        image = np.zeros((rows * tile_size, cols * tile_size), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tiles[i * cols + j]
    
    return image

def colorize_mask(mask, colors):
    colorized_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)  # Ensure output has only 3 channels
    mask = mask[:, :, 0] # Ensure mask is 2D
    for i, color in enumerate(colors):
        colorized_mask[mask == i] = color
    return colorized_mask

def plot_grid(original_img, true_mask, pred_mask, pred_probs, overlaid_img, diff_mask):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    images = [original_img, true_mask, pred_mask, pred_probs, overlaid_img, diff_mask]
    titles = ['Original Image', 'Ground Truth Mask', 'Predicted Mask', 'Model Confidence', 'Mask Overlaid on Image', 'Difference Mask']
    
    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)  # Automatically handle grayscale images
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def predict_and_plot_grid(model, image_number, path_to_tiles, colors, device='cuda' if torch.cuda.is_available() else 'cpu', tile_size=256):
    image_tiles, mask_tiles = load_tiles(image_number, path_to_tiles)
    
    original_img = reconstruct_image(image_tiles, channels=3)  # RGB image reconstruction
    true_mask = reconstruct_image(mask_tiles, channels=1)  # Grayscale mask reconstruction
    
    colorized_true_mask = colorize_mask(true_mask, colors)

    pred_mask_tiles, pred_prob_tiles = [], []
    for tile in image_tiles:
        tile_tensor = torch.tensor(tile).unsqueeze(0).permute(0, 3, 1, 2).float().to(device) / 255.0
        with torch.no_grad():
            output = model(tile_tensor)
            pred_probs = torch.softmax(output, dim=1).squeeze(0)
            pred_mask = torch.argmax(pred_probs, dim=0).cpu().numpy()
            pred_mask_tiles.append(pred_mask)
            pred_prob_tiles.append(pred_probs.max(0)[0].cpu().numpy())

    # Reconstruct the predicted mask and probability images as single-channel
    pred_mask = reconstruct_image(np.array(pred_mask_tiles), tile_size=tile_size, channels=1)
    pred_probs = reconstruct_image(np.array(pred_prob_tiles), tile_size=tile_size, channels=1)
    
    # Colorize the predicted mask
    colorized_pred_mask = colorize_mask(pred_mask, colors)

    # Create the overlay image
    alpha = 0.4
    overlaid_img = cv2.addWeighted(original_img, 1-alpha, colorized_pred_mask, alpha, 0)

    # Compute the difference mask (note: compare single-channel masks)
    diff_mask = (true_mask != pred_mask).astype(np.uint8) * 255

    # Plot everything
    plot_grid(original_img, colorized_true_mask, colorized_pred_mask, pred_probs, overlaid_img, diff_mask)