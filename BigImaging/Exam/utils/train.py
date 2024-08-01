from torch.amp.autocast_mode import autocast
from torch.nn import Module
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torch import Tensor
import torch
import os
import csv
import glob
import pandas as pd

def reshape_imgs_masks(imgs, masks):
    """
    Reshapes the input images and masks tensors.

    Args:
        imgs (torch.Tensor): Input images tensor of shape [batch_size, num_tiles, channels, height, width].
        masks (torch.Tensor): Input masks tensor of shape [batch_size, num_tiles, height, width].

    Returns:
        torch.Tensor: Reshaped images tensor of shape [batch_size * num_tiles, channels, height, width].
        torch.Tensor: Reshaped masks tensor of shape [batch_size * num_tiles, height, width].
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, masks = imgs.to(device), masks.to(device)

    # Reshape images: [batch_size, num_tiles, channels, height, width] -> [batch_size * num_tiles, channels, height, width]
    imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
    # Reshape masks: [batch_size, num_tiles, height, width] -> [batch_size * num_tiles, height, width]
    masks = masks.view(-1, masks.shape[2], masks.shape[3])

    # Convert masks to Long() type
    masks = masks.to(torch.long)

    return imgs, masks

def train(
    train_loss: Tensor,
    imgs: Tensor,
    masks: Tensor,
    model: Module,
    scaler: GradScaler,
    optimizer: Optimizer,
    criterion: Module,
    iteration: int,
    use_amp: bool = True,
    tiles: bool = False
) -> Tensor:
    """
    Trains the model using the given images and masks.

    Args:
        train_loss (torch.Tensor): The current training loss.
        imgs (torch.Tensor): The input images.
        masks (torch.Tensor): The target masks.
        model (torch.nn.Module): The model to be trained.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (torch.nn.Module): The loss function.
        iteration (int): The current iteration number.
        use_amp (bool, optional): Whether to use automatic mixed precision training. Defaults to True.
        tiles (bool, optional): Whether the images and masks are tiled by the Dataset. Defaults to False.

    Returns:
        torch.Tensor: The updated training loss.
    """

    device='cuda' if torch.cuda.is_available() else 'cpu'

    # Only reshape images and masks if tiles are being computed by the Dataset class
    # Else the source is the already tiled images and masks
    if tiles:
        with torch.profiler.record_function("Reshaping images and masks"):
            imgs, masks = reshape_imgs_masks(imgs, masks)
    else:
        with torch.profiler.record_function("Transferring images and masks to device"):
            # TODO - Check if non_blocking=True is necessary
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, torch.long, non_blocking=True)
            masks = masks.to(torch.long)

    optimizer.zero_grad()

    with torch.profiler.record_function("Train Forward pass"):
        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, masks)

    with torch.profiler.record_function("Training Backward pass"):
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    with torch.profiler.record_function("Training Optimizer step"):
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    train_loss += loss.detach()

    return train_loss

def validate(
    val_loss: Tensor,
    imgs: Tensor,
    masks: Tensor,
    model: Module,
    criterion: Module,
    use_amp: bool = True,
    tiles: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Perform validation on the given images and masks using the provided model and criterion.

    Args:
        val_loss (float): Current validation loss.
        imgs (torch.Tensor): Input images.
        masks (torch.Tensor): Ground truth masks.
        model (torch.nn.Module): Model to be used for validation.
        criterion (torch.nn.Module): Loss criterion.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        tiles (bool, optional): Whether the images and masks are tiled. Defaults to False.

    Returns:
        tuple: A tuple containing the updated validation loss and predicted masks.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        if tiles:
            imgs, masks = reshape_imgs_masks(imgs, masks)
        else:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, torch.long, non_blocking=True)

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.detach()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.detach()

        preds = torch.argmax(outputs, dim=1)

    return val_loss, preds

def save_model(
    model: Module,
    train_loss: float,
    val_loss: float,
    optimizer: Optimizer,
    epoch: int,
    config: dict,
    final_dim: int,
    tiles_dim: int,
    save_path: str = 'models/',
    is_best: bool = False
) -> str:
    """
    Save the model and optimizer state dictionaries to disk.
    If is_best is True, save it as the best model.
    If epoch > 0, delete the previous model with the same configuration.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_name = f'model_epoch_{epoch}_final_dim_{final_dim}_tiles_dim_{tiles_dim}_val_loss_{val_loss:.4f}_train_loss_{train_loss:.4f}_arch_{config["arch"]}_encoder_name_{config["encoder_name"]}.pt'
    
    if is_best:
        model_dir = os.path.join(save_path, 'best', model_name)
    else:
        model_dir = os.path.join(save_path, model_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    # If epoch > 0, delete the previous model with the same configuration
    if epoch > 0 and not is_best:
        previous_model_pattern = os.path.join(save_path, f'model_epoch_*_final_dim_{final_dim}_tiles_dim_{tiles_dim}_*_arch_{config["arch"]}_encoder_name_{config["encoder_name"]}.pt')
        for file in glob.glob(previous_model_pattern):
            os.remove(file)
            print(f"Deleted previous model: {file}")

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'final_dim': final_dim,
            'tiles_dim': tiles_dim,
            'train_loss': train_loss,
            'val_loss': val_loss
        },
        model_dir
    )

    print(f'{"Saving best model" if is_best else "Saving model"} to {model_dir}')
    return model_dir

def save_model_stats(
    train_loss: float,
    val_loss: float,
    epoch: int,
    config: dict,
    logs_dir: str,
    model_dir: str,
    final_dim: int,
    tiles_dim: int,
    is_best: bool = False
) -> None:
    """
    Save the training and validation loss to disk.
    If is_best is True, save it as the best model stats.
    Overwrite the line with the same configuration, else add a new line.
    """
    stats = {
        'arch': config['arch'],
        'encoder_name': config['encoder_name'],
        'train_loss': round(train_loss, 4),
        'val_loss': round(val_loss, 4),
        'epoch': epoch,
        'final_dim': final_dim,
        'tiles_dim': tiles_dim,
        'logs_dir': logs_dir,
        'model_dir': model_dir,
    }

    if is_best:
        stats_file = 'best_model_stats.csv'
        mode = 'w'  # Overwrite for best model stats
    else:
        stats_file = 'model_stats.csv'
        mode = 'a+'  # Append+ for regular model stats (allows reading and writing)

    # Read existing data
    existing_data = read_csv_with_empty_handling(stats_file)

    # Check if a row with the same configuration exists
    same_config = existing_data[
        (existing_data['arch'] == stats['arch']) &
        (existing_data['encoder_name'] == stats['encoder_name']) &
        (existing_data['final_dim'] == stats['final_dim']) &
        (existing_data['tiles_dim'] == stats['tiles_dim'])
    ]

    if not same_config.empty and is_best:
        # Update the existing row
        for key, value in stats.items():
            existing_data.loc[same_config.index[0], key] = value
    else:
        # Append a new row
        existing_data = pd.concat([existing_data, pd.DataFrame([stats])], ignore_index=True)

    # Write the updated data back to the CSV file
    existing_data.to_csv(stats_file, index=False)

    print(f'{"Saving best model stats" if is_best else "Updating model stats"} in {stats_file}')

    # If saving best model stats, also update the regular model stats to include the best new model
    if is_best:
        save_model_stats(train_loss, val_loss, epoch, config, logs_dir, model_dir, final_dim, tiles_dim, is_best=False)

# Function to read the CSV file and handle empty file case
def read_csv_with_empty_handling(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"The file {file_path} is empty.")
        return df
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty or does not exist.")
        return pd.DataFrame(columns=[
            'arch', 'encoder_name', 'train_loss', 'val_loss', 'epoch',
            'final_dim', 'tiles_dim', 'logs_dir', 'model_dir'
        ])
    
def initialize_best_val_loss(final_dim: int, tiles_dim: int, config: dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    """
    Initialize the best validation loss from the best model stats CSV file.
    
    Args:
        final_dim (int): The final dimension of the images.
        tiles_dim (int): The dimension of the tiles.
        config (dict): The configuration dictionary containing 'arch' and 'encoder_name'.
        device (str): The device to use. Defaults to 'cuda' if available, else 'cpu'.
    
    Returns:
        float: The best validation loss.
    """
    best_val_loss = float('inf')
    
    # Read the best model stats from the CSV file
    best_model_stats = read_csv_with_empty_handling('best_model_stats.csv')
    
    # Filter the stats for the same configuration
    same_config_stats = best_model_stats[
        (best_model_stats['arch'] == config['arch']) &
        (best_model_stats['encoder_name'] == config['encoder_name']) &
        (best_model_stats['final_dim'] == final_dim) &
        (best_model_stats['tiles_dim'] == tiles_dim)
    ]
    
    try:
        # Try to get the best validation loss from the stats
        best_val_loss = same_config_stats['val_loss'].values[0]
        print(f'Loaded Best Validation Loss: {best_val_loss}')
    except IndexError:
        # If no stats exist, set best_val_loss to infinity
        best_val_loss = torch.tensor(float('inf')).to(device).item()
        print(f'No Best Validation Loss found for configuration {config['arch']} {config['encoder_name']} with final_dim {final_dim} and tiles_dim {tiles_dim}.')
    
    return best_val_loss