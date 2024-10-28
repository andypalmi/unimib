import os
# Suppress Albumentations update check
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.datasets.FoodDataset import FoodDataset
from utils.datasets.ContrastiveDataset import ContrastiveDataset
from utils.utils import load_data
from utils.transforms import train_transforms, valtest_transforms, ssl_transforms, train_transforms_kornia, valtest_transforms_kornia
from torch.utils.data import DataLoader
from utils.train import train
from utils.datasets.DaliPipeline import create_dataloaders

def main():
    # Set the working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data
    train_dict, val_dict, test_dict = load_data()

    # Create DALI dataloaders
    # Note: DALI handles data augmentation internally through the pipeline
    train_ssl_loader, train_loader, val_loader, test_loader = create_dataloaders(
        train_image_paths_labels=train_dict,
        val_image_paths_labels=val_dict,
        test_image_paths_labels=test_dict,
        batch_size=64,
        num_threads=10,  # Equivalent to num_workers in PyTorch
        device_id=0  # Assuming using first GPU
    )
    
    # Start training
    train(
        # trainloader=train_ssl_loader,
        trainloader=train_loader,
        valloader=val_loader,
        run_ssl=False,
        lr=0.001,
        device='cuda',
        epochs=100,
        patience=10,
        first_epochs=5,
        profile_run=False,
        verbose=True
    )

if __name__ == '__main__':
    # Run the main function
    main()