import pandas as pd
import os

import csv

# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory set to:", os.getcwd())

# Define the columns based on the stats dictionary
columns = [
    'arch', 'encoder_name', 'train_loss', 'val_loss', 'epoch',
    'final_dim', 'tiles_dim', 'logs_dir', 'model_dir'
]

# Create a placeholder CSV file
placeholder_file = 'models/best_model_stats.csv'
with open(placeholder_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

placeholder_file = 'models/model_stats.csv'
with open(placeholder_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

# Load the CSV file
best_model_stats = pd.read_csv('models/best_model_stats.csv')

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