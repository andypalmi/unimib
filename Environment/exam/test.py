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

import os

# First path
first_path = 'data/tiles_256/512x512/images/198_56.png'

# Get the directory of the first path
second_path = os.path.dirname(first_path)
print(second_path)

# Navigate up one more level to get the desired directory
second_path = os.path.dirname(second_path)

print(second_path)