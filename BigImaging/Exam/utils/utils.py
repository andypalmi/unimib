import pandas as pd
import numpy as np
import glob
from typing import Tuple

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