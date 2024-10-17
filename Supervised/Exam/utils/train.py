import torch
import torch.nn as nn
from torch.optim.adam import Adam
from utils.networks.FoodNet import FoodNet
from torch.utils.data import DataLoader

def train(dataloader: DataLoader, lr=0.001, device='cuda', epochs=100, patience=10, first_epochs=5, verbose=True):
    # Initialize the model
    model = FoodNet()

    # Define the loss function and optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    previous_loss = torch.tensor(float('inf')).to(device)

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()

        # Initialize the loss and accuracy
        train_loss = torch.tensor(0.0).to(device)

        for i, data in enumerate(dataloader):
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
