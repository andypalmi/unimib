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

# freq = pd.read_csv('data/class_frequencies.csv')
# freq = freq.sort_values('Frequency', ascending=False)
# tot = freq['Frequency'].sum()
# print(tot)
# freq['%'] = round(freq['Frequency'] / tot * 100, 3)
# print(freq)
# freq.to_csv('data/frequencies.csv', index=False)