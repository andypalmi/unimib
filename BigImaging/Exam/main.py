import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import os
import segmentation_models_pytorch as smp
import concurrent.futures

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from sklearn.model_selection import train_test_split
from utils.tiling import process_row
from utils.utils import read_class_colors

from utils.TilesDataset import TilesDataset
from utils.transforms import train_transform, valtest_transform
from utils.metrics import compute_metrics_torch
from utils.train import train, validate
from utils.utils import save_profiling_tables, create_splits
from utils.train import save_model, save_model_stats, read_csv_with_empty_handling, initialize_best_val_loss

# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory set to:", os.getcwd())

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

already_created = True
if not already_created:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_row, [df.iloc[i] for i in range(df.shape[0])]), total=df.shape[0]))

# Read class colors from CSV file
labels_colors, colors, num_classes = read_class_colors('data/ColorMasks/ColorPalette-Values.csv')

# Get image and mask paths for tiles
final_dim = 256
tiles_dim = 750
train_split, val_split, test_split = create_splits(final_dim, tiles_dim)

# Create the Datasets
train_ds = TilesDataset(train_split, transform=train_transform, tiles_dim=tiles_dim, tiles=False)
val_ds = TilesDataset(val_split, transform=valtest_transform, tiles_dim=tiles_dim, tiles=False)
test_ds = TilesDataset(test_split, transform=valtest_transform, tiles_dim=tiles_dim, tiles=False)


# Create the DataLoaders
num_workers = 12
batch_size_train = 60
batch_size_valtest = 150
dataloader_kwargs = {'shuffle': True, 'pin_memory': True, 'num_workers': num_workers, 
                     'persistent_workers': True, 'prefetch_factor': 5, 
                     'pin_memory_device': 'cuda' if torch.cuda.is_available() else 'cpu'}
train_kwargs = {'batch_size': batch_size_train, **dataloader_kwargs}
valtest_kwargs = {'batch_size': batch_size_valtest, **dataloader_kwargs}
print(f'train_kwargs = {train_kwargs}, \nvaltest_kwargs = {valtest_kwargs}')

train_loader = DataLoader(train_ds, **train_kwargs)
val_loader = DataLoader(val_ds, **valtest_kwargs)
test_loader = DataLoader(test_ds, **valtest_kwargs)

# Define model configuration
config = {
    'arch': 'DeepLabV3Plus',
    # 'encoder_name': 'efficientnet-b5',
    # 'encoder_name': 'resnet34',
    'encoder_name': 'mobilenet_v2',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': num_classes
}

# Create the model
MODEL = smp.create_model(**config)

# Create a TensorBoard writer
logs_dir = f'logs/{config["arch"]}/{config["encoder_name"]}/tiles_{final_dim}/{tiles_dim}x{tiles_dim}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists
logs_dir = os.path.abspath(logs_dir)  # Get full path to the logs directory
writer = SummaryWriter(log_dir=logs_dir)

# Set up mixed precision training
scaler = GradScaler()

# Define training parameters
accumulation_steps = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL.to(DEVICE)
criterion = nn.CrossEntropyLoss()
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
optimizer = AdamW(MODEL.parameters(), lr=LEARNING_RATE)
NUM_EPOCHS = 100
TRAIN = True
PROFILE = False

# Early stopping parameters
PATIENCE = 5
MIN_EPOCHS = 5
early_stopping_counter = 0

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25)

if TRAIN:
    # Initialize best validation loss
    best_val_loss = initialize_best_val_loss(final_dim, tiles_dim, config)
    current_best_val_loss = torch.tensor(float('inf')).to(DEVICE)

    # Start the training loop
    for epoch in range(NUM_EPOCHS):
        # Set model to training mode
        MODEL.train()
        train_loss = torch.tensor(0.0).to(DEVICE)

        try:
            pbar.close()
        except:
            pass

        pbar_desc = f'Training | Epoch {epoch+1}/{NUM_EPOCHS}'
        pbar = tqdm(train_loader, desc=pbar_desc, ncols=100)

        if PROFILE:
            # For the first epoch, use profiler
            if epoch == 0:
                with torch.profiler.profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=5, warmup=1, active=1, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
                    record_shapes=True, profile_memory=True, with_stack=True
                ) as prof:
                    for i, (imgs, masks) in enumerate(pbar):
                        train_loss = train(train_loss, imgs, masks, MODEL, scaler, optimizer, criterion, i, use_amp=True)
                        pbar.update(1)
                        pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i+1):.3f}')
                        # Profile each batch
                        prof.step()
                        if i > 25:
                            save_profiling_tables(prof, logs_dir)
                            break
            else:
                # For subsequent epochs, train without profiling
                for i, (imgs, masks) in enumerate(pbar):
                        train_loss = train(train_loss, imgs, masks, MODEL, scaler, optimizer, criterion, i, use_amp=True)
                        pbar.update(1)
                        pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i+1):.3f}')
        # ELSE NO PROFILE
        else:
            for i, (imgs, masks) in enumerate(pbar):
                train_loss = train(train_loss, imgs, masks, MODEL, scaler, optimizer, criterion, i, use_amp=True)
                pbar.update(1)
                pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i+1):.3f}')
                if i % 100 == 0 or i == (len(train_loader) - 1):
                    writer.add_scalar('Loss training', train_loss / (i+1), (i/len(train_loader)) + epoch)

        # Calculate average training loss
        train_loss = train_loss.item() / len(train_loader)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Set model to evaluation mode
        MODEL.eval()
        val_loss = torch.tensor(0.0).to(DEVICE)

        # Initialize accumulators for metrics
        total_iou = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        num_batches = 0
        per_class_iou_accumulators = [0.0] * num_classes

        pbar.close()
        pbar_desc = f'Validating | Epoch {epoch+1}/{NUM_EPOCHS}'
        pbar = tqdm(val_loader, desc=pbar_desc, ncols=100)

        # Validation loop
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(pbar):
                val_loss, preds = validate(val_loss, imgs, masks, MODEL, criterion, use_amp=True)

                # Compute metrics for the current batch
                batch_metrics = compute_metrics_torch(masks.to(DEVICE), preds.to(DEVICE), num_classes, DEVICE)
                
                # Update accumulators
                total_iou += batch_metrics["weighted_mean_iou"]
                total_accuracy += batch_metrics["accuracy"]
                total_dice += batch_metrics["weighted_mean_dice"]
                for cls in range(num_classes):
                    per_class_iou_accumulators[cls] += batch_metrics["per_class_iou"][cls]
                num_batches += 1

                pbar.update(1)
                pbar.set_description(f'{pbar_desc} | Loss: {val_loss / (i+1):.3f}')

        # Calculate average validation loss
        val_loss = val_loss.item() / len(val_loader)

        # Compute final metrics
        mean_iou = total_iou / num_batches
        accuracy = total_accuracy / num_batches
        mean_dice = total_dice / num_batches
        per_class_iou = [iou / num_batches for iou in per_class_iou_accumulators]

       # Print validation results
        print(f'\nEpoch: {epoch} - Validation Loss: {val_loss:.3f}, Mean IoU: {mean_iou:.3f}, '
            f'Accuracy: {accuracy:.3f}, Mean Dice: {mean_dice:.3f}, '
            f'\nper-class IoU: {[f"| Class {i}: {iou:.3f} " for i, iou in enumerate(per_class_iou)]} \n')

        # Log metrics to Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/mean_dice', mean_dice, epoch)
        writer.add_scalar('Metrics/mean_iou', mean_iou, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # Check if this is the best model so far (across all runs)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            is_best = True
        else:
            is_best = False

        # Check if this is the best model so far (for current run)
        if val_loss < current_best_val_loss:
            current_best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            if epoch >= MIN_EPOCHS:
                early_stopping_counter += 1

        # Save the model and its stats
        model_dir = save_model(MODEL, train_loss, val_loss, optimizer, epoch, LEARNING_RATE, config, final_dim, tiles_dim, is_best=is_best)
        save_model_stats(train_loss, val_loss, epoch, LEARNING_RATE, config, logs_dir, model_dir, final_dim, tiles_dim, is_best=is_best)

        # Early stopping check
        if early_stopping_counter >= PATIENCE:
            print(f"Early stopping triggered. No improvement for {PATIENCE} epochs.")
            break

        # Step the learning rate scheduler
        scheduler.step()

        # Free up memory
        del val_loss, train_loss
        torch.cuda.empty_cache()

    # Close Tensorboard writer
    writer.close()