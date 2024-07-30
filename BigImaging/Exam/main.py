import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import os
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.optim.adam import Adam

from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from sklearn.model_selection import train_test_split
from utils.tiling import create_and_save_tiles
from utils.utils import read_class_colors

from utils.TilesDataset import TilesDataset
from utils.transforms import train_transform, valtest_transform
from utils.metrics import compute_metrics_torch
from utils.train import train, validate

# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory set to:", os.getcwd())

# Get the paths of the images and sort them
images_path = sorted(glob.glob('data/original_images/*.jpg'))
labels_path = sorted(glob.glob('data/label_images_semantic/*.png'))
rgb_masks_path = sorted(glob.glob('data/RGB_color_image_masks/*.png'))

paths = np.column_stack((images_path, labels_path))
# print(paths[0])

# Tiling
already_created = True
if not already_created:
    resize_dim = 256
    for path in tqdm(paths):
        img_path, mask_path = path
        create_and_save_tiles(img_path, mask_path, tiles_dim=2000, 
                              final_dim=256, output_dir=f'data/tiles_{resize_dim}')

# Apply 80-10-10 split
train_split, valtest_split = train_test_split(paths, test_size=0.2, random_state=69420)
val_split, test_split = train_test_split(valtest_split, test_size=0.5, random_state=69420)

# Read class colors
labels_colors, colors, num_classes = read_class_colors('data/class_dict_seg.csv')
print(f'type of labels_colors: {type(labels_colors)}, type of colors: {type(colors)}, num_classes: {type(num_classes)}')

# Get image and mask paths
final_dim = 256
tiles_dim = 512
tiles_path = f'data/tiles_{final_dim}/{tiles_dim}x{tiles_dim}'

# Get image paths
for folder in os.listdir(tiles_path):
    if folder == 'images':
        img_paths = sorted(glob.glob(f'{tiles_path}/{folder}/*.png'))
    elif folder == 'masks':
        mask_paths = sorted(glob.glob(f'{tiles_path}/{folder}/*.png'))

# Combine image and mask paths
paths = np.array(list(zip(img_paths, mask_paths)))
# print(paths.shape)
# print(paths[0])

# Apply 80-10-10 split
train_split, valtest_split = train_test_split(paths, test_size=0.2, random_state=69420)
val_split, test_split = train_test_split(valtest_split, test_size=0.5, random_state=69420)

# Create the Datasets
train_ds = TilesDataset(train_split, transform=train_transform, tiles_dim=tiles_dim, tiles=False)
val_ds = TilesDataset(val_split, transform=valtest_transform, tiles_dim=tiles_dim, tiles=False)
test_ds = TilesDataset(test_split, transform=valtest_transform, tiles_dim=tiles_dim, tiles=False)

print(f'Train dataset length: {len(train_ds)}, Val dataset length: {len(val_ds)}, Test dataset length: {len(test_ds)}')

# Create the DataLoaders
num_workers = 12
batch_size_train = 75
batch_size_valtest = 75
train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size_valtest, shuffle=False, pin_memory=False, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=batch_size_valtest, shuffle=False, pin_memory=False, num_workers=num_workers)

config = {
    'arch': 'unet',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': num_classes
}

MODEL = smp.create_model(**config)

# Create a TensorBoard callback
logs_dir = f'logs/{config["arch"]}/{config["encoder_name"]}/tiles_{final_dim}/{tiles_dim}x{tiles_dim}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists
# Get full path to the logs directory
logs_dir = os.path.abspath(logs_dir)

writer = SummaryWriter(log_dir=logs_dir)

# Set up mixed precision training
scaler = GradScaler()

# Accumulation steps
accumulation_steps = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(MODEL.parameters(), lr=1e-3)

NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    MODEL.train()
    train_loss = torch.tensor(0.0)

    if epoch == 0:
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
            record_shapes=True, profile_memory=True, with_stack=True
        ) as prof:
            for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}')):
                train_loss = train(train_loss, imgs, masks, MODEL, scaler, optimizer, criterion, i, accumulation_steps, use_amp=True)
                # Profile each 2 batches
                if i % 2 == 0:
                    prof.step()
    else:
        for i, (imgs, masks) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}')):
            train_loss = train(train_loss, imgs, masks, MODEL, scaler, optimizer, criterion, i, accumulation_steps, use_amp=True)

    writer.close()

    MODEL.eval()
    val_loss = torch.tensor(0.0)
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{NUM_EPOCHS}'):
            val_loss, preds = validate(val_loss, imgs, masks, MODEL, criterion, use_amp=True)
            all_y_true.append(masks.to(DEVICE))
            all_y_pred.append(preds)

    all_y_true = torch.cat(all_y_true, dim=0)
    all_y_pred = torch.cat(all_y_pred, dim=0)

    metrics = compute_metrics_torch(all_y_true, all_y_pred, num_classes, DEVICE)

    print(f'Validation Loss: {val_loss.item()/len(val_loader):.3f}, Mean IoU: {metrics["mean_iou"]:.3f}, '
      f'Accuracy: {metrics["accuracy"]:.3f}, Dice Score: {metrics["mean_dice"]:.3f}, '
      f'per-class IoU: {[f"Class {i}: {iou:.3f}" for i, iou in enumerate(metrics["per_class_iou"])]}')