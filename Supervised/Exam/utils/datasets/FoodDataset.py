import random
from collections import defaultdict
from torch.utils.data import Dataset
import albumentations
import cv2
import os

class FoodDataset(Dataset):
    """
    A custom dataset class for loading and augmenting food images.
    Attributes:
        image_paths_labels (dict): A dictionary where keys are image filenames and values are their respective file paths.
        transform (albumentations.core.composition.Compose, optional): A collection of albumentations transformations to be applied to the images.
        augmentation_counts (defaultdict): A dictionary to count the number of augmentations applied per class.
    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and its label at the specified index. Applies augmentation if the count for the label has not been reached.
    """
    def __init__(self, image_paths_labels: dict[str, str], transform: albumentations.Compose | None = None):
        self.image_paths_labels = image_paths_labels
        self.transform = transform
        self.augmentation_counts = defaultdict(int)

    def __len__(self):
        return len(self.image_paths_labels)

    def __getitem__(self, idx):
        # Get the image path and label using the index
        img_path = list(self.image_paths_labels.keys())[idx]
        label = self.image_paths_labels[img_path]
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img, label  # Return the image and the label