# import pandas as pd
import os
import pandas as pd
# import glob
# from utils.utils import read_class_colors
# from tqdm import tqdm
# import cv2
# import numpy as np

# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory set to:", os.getcwd())

# # Load the CSV file
# best_model_stats = pd.read_csv('best_model_stats.csv')

# # Print the columns to inspect them
# print("Columns in DataFrame:", best_model_stats.columns)

# # Strip any leading/trailing spaces from column names
# best_model_stats.columns = best_model_stats.columns.str.strip()

# # Access the 'val_loss' column
# try:
#     best_val_loss = best_model_stats['val_loss'].values[0]
#     print(f"Best Validation Loss: {best_val_loss}")
# except KeyError:
#     print("Error: 'val_loss' column not found in the DataFrame.")

# from utils.train import save_model_stats

# save_model_stats(999999, 999999, 9999, 0.0001, {'arch': 'DeepLabV3Plus', 'encoder_name': 'mobilenet_v2'}, 'test', 'test', 256, 99999999, is_best=True)

# # labels_path = sorted(glob.glob('data/label_images_semantic/*.png'))
# # # Read class colors from CSV file
# # labels_colors, colors, num_classes = read_class_colors('data/class_dict_seg.csv')
# # print("Class Colors:", labels_colors)

# Load the CSV file
df = pd.read_csv('models/best/test_results.csv')

# Round all decimal numbers to 5 decimal places
df = df.round(5)

# Round the decimal numbers in the 'per_class_iou' column to 5 decimal places
df['per_class_iou'] = df['per_class_iou'].apply(lambda x: [round(float(v), 5) for v in eval(x)])

# Save the updated DataFrame back to the CSV file
df.to_csv('models/best/test_results.csv', index=False)