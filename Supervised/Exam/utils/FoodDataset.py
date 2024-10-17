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
        transform_counts (dict): A dictionary where keys are labels and values are the number of augmentations to be applied for each label.
        transform (albumentations.core.composition.Compose, optional): A collection of albumentations transformations to be applied to the images.
        augmentation_counts (defaultdict): A dictionary to count the number of augmentations applied per class.
    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and its label at the specified index. Applies augmentation if the count for the label has not been reached.
    """
    def __init__(self, image_paths_labels: dict[str, str], split : str, transform_counts: dict[int, int], transform: albumentations.Compose | None = None):
        self.image_paths_labels = image_paths_labels
        self.split = split
        self.transform_counts = transform_counts
        self.transform = transform
        self.augmentation_counts = defaultdict(int)

    def __len__(self):
        return len(self.image_paths_labels)

    def __getitem__(self, idx):
        # Get the image path and label using the index
        img_path = list(self.image_paths_labels.keys())[idx]
        label = self.image_paths_labels[img_path]

        root = 'data/images/' + ('train_set' if self.split == 'train' else 'train_set' if self.split == 'val' else 'val_set')
        img_path = os.path.join(root, img_path)
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Check how many augmentations are remaining for this class
        if self.augmentation_counts[label] < self.transform_counts[int(label)]:
            # Perform augmentation if we haven't reached the count yet
            if self.transform is not None:
                augmented = self.transform(image=img)
                img = augmented['image']
            
            # Increment the augmentation count for the current label
            self.augmentation_counts[label] += 1

        return img, label  # Return the image and the label