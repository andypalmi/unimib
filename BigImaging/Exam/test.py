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

# save_model_stats(0.373, 0.3522, 22, 0.001, {'arch': 'unet', 'encoder_name': 'resnet34'}, '/home/andrea/Documents/unimib/BigImaging/Exam/logs/unet/resnet34/tiles_256/512x512/2024-07-31_15-42-22', ',models/best/model_epoch_22_final_dim_256_tiles_dim_512_val_loss_0.3522_train_loss_0.3730_arch_unet_encoder_name_resnet34.pt', 256, 000000, is_best=True)

labels_path = sorted(glob.glob('data/label_images_semantic/*.png'))
# Read class colors from CSV file
labels_colors, colors, num_classes = read_class_colors('data/class_dict_seg.csv')
print("Class Colors:", labels_colors)
class_counts = {}
for label_path in tqdm(labels_path):
    label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # Count the occurrences of each class in the label image
    unique_classes, counts = np.unique(label_image, return_counts=True)

    # Update the overall class counts dictionary
    for class_index, class_count in zip(unique_classes, counts):
        class_name = colors[class_index]
        class_counts[class_name] = class_counts.get(class_name, 0) + class_count

print("Class Counts:")
for i, (class_name, count) in enumerate(class_counts.items()):
    print(f"{class_name}: {count}")