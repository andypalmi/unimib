import pandas as pd
import numpy as np
import os
from utils.networks.FoodNet import FoodNet
from utils.FoodDataset import FoodDataset
from utils.utils import load_data
from utils.transforms import train_transforms, valtest_transforms
from torch.utils.data import DataLoader
from utils.train import train


os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_dict, val_dict, test_dict, train_transform_dict, val_transform_dict, test_transform_dict = load_data()

# Create a FoodDataset object for the training set
train_dataset = FoodDataset(train_dict, train_transform_dict, transform=train_transforms)
val_dataset = FoodDataset(val_dict, val_transform_dict, transform=valtest_transforms)
test_dataset = FoodDataset(test_dict, test_transform_dict, transform=valtest_transforms)

# Create a DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=12, prefetch_factor=10)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

train(dataloader=train_loader)