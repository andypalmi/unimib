import pandas as pd
import numpy as np
import os
from utils.networks.FoodNet import FoodNet
from utils.datasets.FoodDataset import FoodDataset
from utils.datasets.ContrastiveDataset import ContrastiveDataset
from utils.utils import load_data
from utils.transforms import train_transforms, valtest_transforms, ssl_transforms
from torch.utils.data import DataLoader
from utils.train import train

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_dict, val_dict, test_dict = load_data()

# Create a FoodDataset object for the training set
ssl_train_dataset = ContrastiveDataset(train_dict, transform=ssl_transforms)
train_dataset = FoodDataset(train_dict, transform=train_transforms)
val_dataset = FoodDataset(val_dict, transform=valtest_transforms)
test_dataset = FoodDataset(test_dict, transform=valtest_transforms)

# Create a DataLoader for the training set
ssl_train_loader = DataLoader(ssl_train_dataset, batch_size=160, shuffle=True, num_workers=12, prefetch_factor=5)
train_loader = DataLoader(train_dataset, batch_size=160, shuffle=True, num_workers=12, prefetch_factor=5)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# train(trainloader=train_loader, valloader=val_loader)
train(trainloader=ssl_train_loader, valloader=val_loader, run_ssl=True, lr=0.001, device='cuda', epochs=100, patience=10, first_epochs=5, profile_run=False, verbose=True)