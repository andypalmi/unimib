import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from utils.utils import load_pretrained_model
from utils.networks.FoodNetResiduals import FoodNetResiduals, FoodNetResidualsSSL
from utils.networks.FoodNetInvertedResiduals import FoodNetInvertedResidualsSSL
from utils.networks.FoodNetExtraDW import FoodNetExtraDW
from utils.networks.FoodNetUnetExtraDW import FoodNetUnetExtraDW
from torch.utils.data import DataLoader
from utils.loss.ContrastiveLoss import ContrastiveLoss
from tqdm import tqdm
# from torchsummary import summary
from utils.utils import summary
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from datetime import datetime
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.optimizer import Optimizer
from torch.nn.modules.module import Module
from typing import Union, Tuple, Dict
from torch.utils.data import DataLoader
from nvidia.dali.plugin.pytorch import DALIGenericIterator

def get_logs_dir(base_dir: str = 'logs') -> str:
    """
    Creates and returns a directory path for storing logs with timestamp.

    Args:
        base_dir (str): Base directory path for storing logs. Defaults to 'Supervised/Exam/logs'.

    Returns:
        str: Absolute path to the created logs directory.
    """
    logs_dir = f'{base_dir}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.abspath(logs_dir)

def train(
    trainloader: Union[DataLoader, DALIGenericIterator],
    valloader: Union[DataLoader, DALIGenericIterator],
    run_ssl: bool = True,
    lr: float = 0.005,
    device: str = 'cuda',
    epochs: int = 100,
    patience: int = 10,
    first_epochs: int = 5,
    profile_run: bool = False,
    verbose: bool = True
) -> None:
    """
    Trains a neural network model using either supervised or self-supervised learning.

    Args:
        trainloader (DataLoader): DataLoader for training data.
        valloader (DataLoader): DataLoader for validation data.
        run_ssl (bool): If True, runs self-supervised learning. If False, runs supervised learning.
        lr (float): Initial learning rate. Defaults to 0.001.
        device (str): Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda'.
        epochs (int): Maximum number of training epochs. Defaults to 100.
        patience (int): Number of epochs to wait for improvement before early stopping. Defaults to 10.
        first_epochs (int): Number of initial epochs before applying early stopping. Defaults to 5.
        profile_run (bool): If True, enables performance profiling. Defaults to False.
        verbose (bool): If True, prints training progress. Defaults to True.
    """
    # Initialize the model

    print(f'{'-'*50}')
    print(f'Running {"self-supervised" if run_ssl else "supervised"} learning')
    print(f'{'-'*50}')

    is_dali = isinstance(trainloader, DALIGenericIterator)

    if run_ssl:
        model = FoodNetExtraDW()
        summary(model, [(3, 256, 256), (3, 256, 256)], print_network=True, device=str(device))
        criterion = ContrastiveLoss()
    else:
        model = load_pretrained_model('models/best/FoodNetExtraDW_epoch_4_lr_0.000090_T0_25_train_loss_4.4485_val_loss_4.3950.pt')
        summary(model, (3, 256, 256), print_network=False, device=str(device))
        criterion = nn.CrossEntropyLoss()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    t_0 = 25
    scheduler = CosineAnnealingWarmRestarts(optimizer, t_0, eta_min=lr / 1000)
    
    logs_dir = get_logs_dir()
    writer = SummaryWriter(logs_dir)

    previous_train_loss = float('inf')
    previous_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        if run_ssl:
            pbar_desc = f'SSL Training | Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f}'
        else:
            pbar_desc = f'Head Training | Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f}'
        
        pbar = tqdm(range(len(trainloader)), desc=pbar_desc, ncols=150)

        if run_ssl:
            train_loss = run_ssl_training_step(
                trainloader, model, optimizer, criterion, 
                pbar, pbar_desc, device, train_loss, 
                writer, epoch, is_dali
            )
        else:
            if profile_run and epoch == 0:
                train_loss = profile_training_step(
                    trainloader, model, optimizer, criterion,
                    pbar, pbar_desc, logs_dir, device,
                    train_loss, writer, epoch, is_dali
                )
            else:
                train_loss = run_training_step(
                    trainloader, model, optimizer, criterion,
                    pbar, pbar_desc, device, train_loss,
                    writer, epoch, is_dali
                )

        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        if epoch > first_epochs:
            if train_loss > previous_train_loss:
                patience -= 1
                if patience == 0:
                    if verbose:
                        print('Early stopping at epoch', epoch)
                    break
            else:
                patience = 10

        if verbose:
            print(f'Epoch {epoch+1}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | '
                  f'Training Loss: {train_loss:.4f} | Previous Loss: {previous_train_loss:.4f} | '
                  f'Patience: {patience}')

        if not run_ssl:
            accuracy, class_accuracy, val_loss = validate(valloader, model, device, is_dali)

        if run_ssl:
            if train_loss < previous_train_loss:
                previous_train_loss = train_loss
                save_model(model, train_loss, float('inf'), optimizer, epoch,
                          scheduler.get_last_lr()[0], t_0, logs_dir, is_best=True)
        else:
            if val_loss < previous_val_loss:
                previous_val_loss = val_loss
                save_model(model, train_loss, val_loss, optimizer, epoch,
                          scheduler.get_last_lr()[0], t_0, logs_dir, is_best=True)

        previous_train_loss = train_loss

        # DALIGenericIterator reset() is called automatically thanks to auto_reset=True!

    writer.close()

def run_ssl_training_step(
    dataloader: Union[DataLoader, DALIGenericIterator],
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    pbar: tqdm,
    pbar_desc: str,
    device: str,
    train_loss: float,
    writer: SummaryWriter,
    epoch: int,
    is_dali: bool
) -> float:
    """
    Executes one epoch of self-supervised learning training.

    Args:
        dataloader (DataLoader): DataLoader containing training data pairs.
        model (Module): Neural network model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Module): Loss function for self-supervised learning.
        pbar (tqdm): Progress bar object.
        pbar_desc (str): Description for progress bar.
        device (str): Device to run training on.
        train_loss (torch.Tensor): Tensor to accumulate training loss.
        writer (SummaryWriter): TensorBoard writer object.
        epoch (int): Current epoch number.
        is_dali (bool): If True, indicates DALI pipeline is used.
    Returns:
        float: Average training loss for the epoch.
    """
    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 20)

    for i, data in enumerate(dataloader):
        # Handle different data formats
        if is_dali:
            img1, img2 = data[0]['view1'], data[0]['view2']
        else:
            img1, img2 = data[0].to(device), data[1].to(device) # type: ignore

        optimizer.zero_grad()
        z1, z2 = model(img1, img2)
        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            writer.add_scalar('SSL Training Loss', train_loss / (i + 1), epoch * num_batches + i + 1)

        pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i + 1):.4f}')
        pbar.update(1)

    torch.cuda.empty_cache()
    return train_loss / num_batches

def run_training_step(
    dataloader: Union[DataLoader, DALIGenericIterator],
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    pbar: tqdm,
    pbar_desc: str,
    device: str,
    train_loss: float,
    writer: SummaryWriter,
    epoch: int,
    is_dali: bool
) -> float:
    """
    Executes one epoch of supervised learning training.

    Args:
        dataloader (DataLoader): DataLoader containing training data.
        model (Module): Neural network model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Module): Loss function for supervised learning.
        pbar (tqdm): Progress bar object.
        pbar_desc (str): Description for progress bar.
        device (str): Device to run training on.
        train_loss (torch.Tensor): Tensor to accumulate training loss.
        writer (SummaryWriter): TensorBoard writer object.
        epoch (int): Current epoch number.

    Returns:
        torch.Tensor: Average training loss for the epoch.
    """
    num_batches = len(dataloader)
    log_interval = max(1, num_batches // 20)
    
    correct = 0
    total = 0

    for i, data in enumerate(dataloader):
        # Handle different data formats
        if is_dali:
            inputs = data[0]['images']
            labels = data[0]['labels'].squeeze()
        else:
            inputs, labels = data[0].to(device), data[1].to(device) # type: ignore

        optimizer.zero_grad()
        outputs = model(inputs, mode='train_supervised')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            writer.add_scalar('Training Loss', train_loss / (i + 1), epoch * num_batches + i + 1)
            writer.add_scalar('Training Accuracy', 100 * correct / total, epoch * num_batches + i + 1)

        accuracy = 100 * correct / total
        pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i + 1):.4f} | Accuracy: {accuracy:.2f}%')
        pbar.update(1)

    torch.cuda.empty_cache()
    return train_loss / num_batches

def profile_training_step(
    dataloader: Union[DataLoader, DALIGenericIterator],
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    pbar: tqdm,
    pbar_desc: str,
    logs_dir: str,
    device: str,
    train_loss: float,
    writer: SummaryWriter,
    epoch: int,
    is_dali: bool
) -> float:
    """
    Executes one epoch of training with performance profiling enabled.

    Args:
        dataloader (DataLoader): DataLoader containing training data.
        model (Module): Neural network model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Module): Loss function.
        pbar (tqdm): Progress bar object.
        pbar_desc (str): Description for progress bar.
        logs_dir (str): Directory to save profiling logs.
        device (str): Device to run training on.
        train_loss (torch.Tensor): Tensor to accumulate training loss.

    Returns:
        torch.Tensor: Average training loss for the epoch.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=5, warmup=10, active=20),
        on_trace_ready=tensorboard_trace_handler(logs_dir),
        record_shapes=True, profile_memory=True, with_stack=True
    ) as prof:
        num_batches = len(dataloader)
        log_interval = max(1, num_batches // 20)
        
        correct = 0
        total = 0

        for i, data in enumerate(dataloader):
            # Handle different data formats
            if is_dali:
                inputs = data[0]['images']
                labels = data[0]['labels'].squeeze()
            else:
                inputs, labels = data[0].to(device), data[1].to(device) # type: ignore

            optimizer.zero_grad()
            outputs = model(inputs, mode='train_supervised')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
                writer.add_scalar('Training Loss', train_loss / (i + 1), epoch * num_batches + i + 1)
                writer.add_scalar('Training Accuracy', 100 * correct / total, epoch * num_batches + i + 1)

            accuracy = 100 * correct / total
            pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i + 1):.4f} | Accuracy: {accuracy:.2f}%')
            pbar.update(1)

            prof.step()

            if i == 50:
                save_profiling_tables(prof, logs_dir)

        torch.cuda.empty_cache()
        return train_loss / num_batches

def save_profiling_tables(prof: profile, logs_dir: str) -> None:
    """
    Saves CPU and CUDA profiling data to text files.

    Args:
        prof (profile): PyTorch profiler object containing performance data.
        logs_dir (str): Directory to save the profiling tables.
    """
    cpu_time_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    cuda_time_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

    with open(f'{logs_dir}/cpu_time_total.txt', 'w') as f:
        f.write(cpu_time_table)

    with open(f'{logs_dir}/cuda_time_total.txt', 'w') as f:
        f.write(cuda_time_table)

    print('Profiling tables saved successfully')

def validate(
    valloader: Union[DataLoader, DALIGenericIterator],
    model: Module,
    device: str = 'cuda',
    is_dali: bool = False,
    num_classes: int = 251
) -> tuple[float, torch.Tensor, float]:
    """
    Validates the model on a validation dataset with integer labels from 0 to 250.

    Args:
        valloader (DataLoader): DataLoader containing validation data.
        model (Module): Neural network model to validate.
        device (str): Device to run validation on. Defaults to 'cuda'.
        is_dali (bool): Whether using DALI data loader. Defaults to False.
        num_classes (int): Number of classes in the dataset. Defaults to 251.

    Returns:
        tuple[float, torch.Tensor, float]: Tuple containing:
            - float: Overall accuracy percentage
            - torch.Tensor: Per-class accuracy percentages
            - float: Average validation loss
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    pbar_desc = "Validation Progress"
    pbar = tqdm(valloader, total=len(valloader), desc=pbar_desc, ncols=120)

    with torch.no_grad():
        for i, data in enumerate(valloader):
            # Handle different data formats
            if is_dali:
                images = data[0]['images']
                labels = data[0]['labels'].squeeze()
            else:
                images, labels = data[0].to(device), data[1].to(device) # type: ignore

            outputs = model(images, mode='train_supervised')
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update class-wise statistics
            for label in range(num_classes):
                mask = (labels == label)
                class_total[label] += mask.sum().item()
                class_correct[label] += ((predicted == labels) & mask).sum().item()

            accuracy = 100 * correct / total
            pbar.set_description(f'{pbar_desc} | Loss {val_loss / (i + 1):.4f} | Accuracy: {accuracy:.2f}%')

    pbar.close()

    # Compute class-wise accuracy
    class_accuracy = 100 * class_correct / class_total

    # Print classes falling into different accuracy bins
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for low, high in bins:
        classes_in_bin = torch.where((class_accuracy >= low) & (class_accuracy < high))[0].tolist()
        print(f'Classes with accuracy between {low}% and {high}%: {classes_in_bin}')

    # Compute average validation loss
    val_loss /= len(valloader)
    print(f'Validation Loss: {val_loss:.4f}')

    return accuracy, class_accuracy, val_loss

def save_model(
    model: Module,
    train_loss: float,
    val_loss: float,
    optimizer: Optimizer,
    epoch: int,
    learning_rate: float,
    T_0: int,
    logs_dir: str,
    save_path: str = 'models/',
    is_best: bool = False,
    verbose: bool = False
) -> str:
    """
    Saves model checkpoint with training state and metadata.

    Args:
        model (Module): Neural network model to save.
        train_loss (float): Current training loss.
        val_loss (float): Current validation loss.
        optimizer (Optimizer): Optimizer state to save.
        epoch (int): Current epoch number.
        learning_rate (float): Current learning rate.
        T_0 (int): Period of learning rate restart in CosineAnnealingWarmRestarts.
        logs_dir (str): Directory containing training logs.
        save_path (str): Directory to save model. Defaults to 'models/'.
        is_best (bool): If True, saves as best model. Defaults to False.

    Returns:
        str: Path to saved model file.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get the model class name (e.g., FoodNetResiduals)
    model_name = model.__class__.__name__

    # Create the model filename
    model_filename = f'{model_name}_epoch_{epoch}_lr_{learning_rate:.6f}_T0_{T_0}_train_loss_{train_loss:.4f}_val_loss_{val_loss:.4f}.pt'

    # Save model as best if is_best is True, else save normally
    model_dir = os.path.join(save_path, 'best' if is_best else '', model_filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    # # Delete any previous models with similar names
    # previous_model_pattern = os.path.join(save_path, f'*{model_name}_epoch_*_lr_*_T0_{T_0}_*.pt')
    # for file in glob.glob(previous_model_pattern):
    #     os.remove(file)
    #     print(f"Deleted previous model: {file}")

    # Save the model state, optimizer state, and the entire model architecture
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': learning_rate,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_architecture': model  # Saving the model architecture
        },
        model_dir
    )

    if verbose:
        print(f'{"Saving best model" if is_best else "Saving model"} to {model_dir}')

    save_model_stats(model, train_loss, val_loss, epoch, learning_rate, logs_dir, model_dir, T_0, is_best)
    
    return model_dir

def load_model(model_path: str): 
    """
    Load a saved model from a file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        Module: Loaded model.
    """
    checkpoint = torch.load(model_path)
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def save_model_stats(
    model: Module,
    train_loss: float,
    val_loss: float,
    epoch: int,
    learning_rate: float,
    logs_dir: str,
    model_dir: str,
    T_0: int,
    is_best: bool = False,
    verbose=False
) -> None:
    """
    Saves model training statistics to CSV files.

    Args:
        model (Module): Neural network model.
        train_loss (float): Training loss value.
        val_loss (float): Validation loss value.
        epoch (int): Current epoch number.
        learning_rate (float): Current learning rate.
        logs_dir (str): Directory containing training logs.
        model_dir (str): Directory containing saved model.
        T_0 (int): Period of learning rate restart in CosineAnnealingWarmRestarts.
        is_best (bool): If True, updates best model statistics. Defaults to False.
        verbose (bool): If True, prints the updated stats file. Defaults to False.
    """
    model_name = model.__class__.__name__

    stats = {
        'model_name': model_name,
        'epoch': epoch,
        'train_loss': round(train_loss, 4),
        'val_loss': round(val_loss, 4),
        'learning_rate': learning_rate,
        'T_0': T_0,
        'logs_dir': logs_dir,
        'model_dir': model_dir,
    }

    # Always update regular model stats
    regular_stats_file = os.path.join('models/stats', 'model_stats.csv')
    update_stats_file(regular_stats_file, stats, overwrite=False)
    if verbose:
        print(f'Updated model stats in {regular_stats_file}')

    # If it's the best model, update best model stats
    if is_best:
        best_stats_file = os.path.join('models/stats', 'best_model_stats.csv')
        update_stats_file(best_stats_file, stats, overwrite=True)
        if verbose:
            print(f'Updated best model stats in {best_stats_file}')

def update_stats_file(file_path: str, stats: dict, overwrite: bool = False, verbose=False) -> None:
    """
    Update the stats file with the given stats.
    If overwrite is True, replace the existing row for the same configuration.
    Otherwise, append a new row.

    Args:
        file_path (str): Path to the CSV file.
        stats (dict): Dictionary containing model statistics.
        overwrite (bool): If True, overwrites existing stats for same configuration. Defaults to False.
        verbose (bool): If True, prints the updated file path. Defaults to False.
    """
    # Read existing data
    existing_data = read_csv_with_empty_handling(file_path)

    # Check if a row with the same model name and epoch exists
    same_config = existing_data[
        (existing_data['model_name'] == stats['model_name']) &
        (existing_data['epoch'] == stats['epoch'])
    ]

    if not same_config.empty and overwrite:
        # Update the existing row
        for key, value in stats.items():
            existing_data.loc[same_config.index[0], key] = value
    else:
        # Append a new row
        new_data = pd.DataFrame([stats])
        if existing_data.empty:
            existing_data = new_data
        else:
            existing_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Write the updated data back to the CSV file
    existing_data.to_csv(file_path, index=False)
    if verbose:
        print(f"{'Overwritten' if overwrite else 'Appended'} row in {file_path}")


def read_csv_with_empty_handling(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file with proper handling of empty files.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data or empty DataFrame with predefined columns.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"The file {file_path} is empty.")
        return df
    except pd.errors.EmptyDataError:
        print(f"The file {file_path} is empty or does not exist.")
        return pd.DataFrame(columns=[
            'model_name', 'epoch', 'train_loss', 'val_loss', 'learning_rate', 'T_0', 'logs_dir', 'model_dir'
        ])
