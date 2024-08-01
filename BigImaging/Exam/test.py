import pandas as pd
import os

import csv

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

save_model_stats(0.373, 0.3522, 22, {'arch': 'unet', 'encoder_name': 'resnet34'}, '/home/andrea/Documents/unimib/BigImaging/Exam/logs/unet/resnet34/tiles_256/512x512/2024-07-31_15-42-22', ',models/best/model_epoch_22_final_dim_256_tiles_dim_512_val_loss_0.3522_train_loss_0.3730_arch_unet_encoder_name_resnet34.pt', 256, 000000, is_best=True)