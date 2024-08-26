import pandas as pd
import os
import glob
from utils.utils import read_class_colors
from tqdm import tqdm
import cv2
import numpy as np

# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory set to:", os.getcwd())

# Load the CSV file
best_model_stats = pd.read_csv('best_model_stats.csv')

# Print the columns to inspect them
print("Columns in DataFrame:", best_model_stats.columns)

# Strip any leading/trailing spaces from column names
best_model_stats.columns = best_model_stats.columns.str.strip()

# Access the 'val_loss' column
try:
    best_val_loss = best_model_stats['val_loss'].values[0]
    print(f"Best Validation Loss: {best_val_loss}")
except KeyError:
    print("Error: 'val_loss' column not found in the DataFrame.")

from utils.train import save_model_stats

save_model_stats(999999, 999999, 9999, 0.0001, {'arch': 'DeepLabV3Plus', 'encoder_name': 'mobilenet_v2'}, 'test', 'test', 256, 99999999, is_best=True)

# labels_path = sorted(glob.glob('data/label_images_semantic/*.png'))
# # Read class colors from CSV file
# labels_colors, colors, num_classes = read_class_colors('data/class_dict_seg.csv')
# print("Class Colors:", labels_colors)
