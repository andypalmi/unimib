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
import torch.multiprocessing as mp
from utils.datasets.DaliPipeline import create_dataloaders

def main():
    # Set the working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data
    train_dict, val_dict, test_dict = load_data()
    
    # Create dataset objects
    ssl_train_dataset = ContrastiveDataset(train_dict, transform=ssl_transforms)
    train_dataset = FoodDataset(train_dict, transform=train_transforms_kornia)
    val_dataset = FoodDataset(val_dict, transform=valtest_transforms_kornia)
    test_dataset = FoodDataset(test_dict, transform=valtest_transforms_kornia)
    
    # Create DataLoader objects with reduced number of workers to minimize duplicate warnings
    common_dataloader_args = {
        'batch_size': 96,
        'num_workers': 8,  # Reduced from 10
        'prefetch_factor': 2,
        'pin_memory': True
    }
    
    # ssl_train_loader = DataLoader(
    #     ssl_train_dataset,
    #     shuffle=True,
    #     **common_dataloader_args
    # )
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     **common_dataloader_args
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     shuffle=False,
    #     **common_dataloader_args
    # )
    
    # test_loader = DataLoader(
    #     test_dataset,
    #     shuffle=False,
    #     **common_dataloader_args
    # )

    # Create DALI dataloaders
    # Note: DALI handles data augmentation internally through the pipeline
    train_loader, val_loader, test_loader = create_dataloaders(
        train_image_paths_labels=train_dict,
        val_image_paths_labels=val_dict,
        test_image_paths_labels=test_dict,
        batch_size=256,
        num_threads=8,  # Equivalent to num_workers in PyTorch
        device_id=0  # Assuming using first GPU
    )
    
    # Start training
    train(
        trainloader=train_loader,
        valloader=val_loader,
        run_ssl=False,
        lr=0.005,
        device='cuda',
        epochs=100,
        patience=10,
        first_epochs=5,
        profile_run=False,
        verbose=True
    )

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    # Run the main function
    main()