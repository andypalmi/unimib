import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from utils.networks.FoodNet import FoodNet, FoodNetASPP
from utils.networks.FoodNetResiduals import FoodNetResiduals
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
from torch.profiler import ProfilerActivity
from datetime import datetime
import os

logs_dir = f'Supervised/Exam/logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists
logs_dir = os.path.abspath(logs_dir)  # Get full path to the logs directory

def train(dataloader: DataLoader, lr=0.001, device='cuda', epochs=100, patience=10, first_epochs=5, profile=True, verbose=True):
    # Initialize the model
    model = FoodNetResiduals()
    model.to(device)

    summary(model, (3, 256, 256), device=str(device))

    # Define the loss function and optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    previous_loss = torch.tensor(float('inf')).to(device)

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()

        try:
            pbar.close()
        except:
            pass

        # Initialize the loss and accuracy
        train_loss = torch.tensor(0.0).to(device)

        # Initialize pbar
        pbar_desc = f'Training | Epoch {epoch+1}/{epochs}'
        pbar = tqdm(dataloader, total=len(dataloader), desc=pbar_desc, ncols=150)

        if epoch == 0 and profile:
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
                record_shapes=True, profile_memory=True, with_stack=True
            ) as prof:
                for i, data in enumerate(pbar):
                    # Get the inputs and labels
                    inputs, labels = data[0].to(device), data[1].to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Optimize
                    optimizer.step()

                    # Update the loss
                    train_loss += loss.item()

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i+1):.4f} | Previous Loss {previous_loss:.4f} | Patience {patience}')

                    # Profile each batch
                    prof.step()
                    if i == 25:
                        save_profiling_tables(prof, logs_dir)
        else:
            # Get the inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            # Update the loss
            train_loss += loss.item()

            # Update the progress bar
            pbar.update(1)
            pbar.set_description(f'{pbar_desc} | Loss: {train_loss / (i+1):.4f} | Previous Loss {previous_loss:.4f} | Patience {patience}')

        # Calculate the average loss
        train_loss /= len(dataloader)

        # Early stopping with patience and first_epochs before triggering
        if epoch > first_epochs:
            if train_loss > previous_loss:
                patience -= 1
                if patience == 0:
                    if verbose:
                        print('Early stopping at epoch', epoch)
                    break
            else:
                patience = 10

        # Update the previous loss
        previous_loss = train_loss

        if verbose:
            print(f'Epoch {epoch+1}/{epochs} | Training Loss: {train_loss:.4f} | Previous Loss {previous_loss:.4f} | Patience {patience}')

def save_profiling_tables(
    prof,
    logs_dir
):
    """
    Save CPU and CUDA time tables from the profiler to text files.

    Args:
        prof: The profiler object containing the profiling data.
        logs_dir: The directory where the log files will be saved.
    """
    cpu_time_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    cuda_time_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)

    with open(f'{logs_dir}/cpu_time_total.txt', 'w') as f:
        f.write(cpu_time_table)
        print(f'CPU time table saved to {logs_dir}/cpu_time_total.txt')

    with open(f'{logs_dir}/cuda_time_total.txt', 'w') as f:
        f.write(cuda_time_table)
        print(f'CUDA time table saved to {logs_dir}/cuda_time_total.txt')