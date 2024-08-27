# import pandas as pd
import os
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

import pandas as pd
import glob

from utils.tiling import create_and_save_tiles

# Tiling

path = 'data/FloodNet'

data = []

for folder in glob.glob(f'{path}/*'):
    images_paths = sorted(glob.glob(f'{folder}/images/*'))
    masks_paths = sorted(glob.glob(f'{folder}/masks/*'))
    
    for img_path, mask_path in zip(images_paths, masks_paths):
        data.append({
            'split': folder.split('/')[-1].split('/')[-1],
            'img': img_path,
            'mask': mask_path
        })

df = pd.DataFrame(data, columns=['split', 'img', 'mask'])

print(df.shape)

import concurrent.futures
from tqdm import tqdm

already_created = False

def process_row(row):
    tiles_dim = 750
    final_dim = 256
    create_and_save_tiles(split=row['split'], img_path=row['img'], mask_path=row['mask'], 
                          tiles_dim=tiles_dim, final_dim=final_dim, output_dir=f'data/tiles_{final_dim}')

if not already_created:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_row, [df.iloc[i] for i in range(df.shape[0])]), total=df.shape[0]))