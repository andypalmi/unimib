import torch
import cv2
import os
from torch.utils.data import Dataset
import kornia as K
from typing import Dict
from functools import lru_cache

class FoodDataset(Dataset):
    """
    Optimized dataset class with proper CUDA handling
    """
    def __init__(self, image_paths_labels: Dict[str, str], transform: torch.nn.Sequential | None = None):
        self.image_paths_labels = image_paths_labels
        self.transform = transform
        self.images = {}
        self.max_cache_size = 1000
        self.paths_list = list(image_paths_labels.keys())
        
        # Don't move transforms to GPU in __init__ since this will be done in worker processes
        self.device = 'cpu'  # Default to CPU, will be updated in __getitem__ if needed
    
    @lru_cache(maxsize=1000)
    def _load_image(self, img_path: str) -> torch.Tensor:
        """Cache-enabled image loading"""
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = K.utils.image_to_tensor(img, keepdim=False).float() / 255.0
        return img_tensor.squeeze(0)

    def __len__(self):
        return len(self.image_paths_labels)

    def __getitem__(self, idx):
        img_path = self.paths_list[idx]
        label = self.image_paths_labels[img_path]
        
        # Load image (always on CPU first)
        img_tensor = self._load_image(img_path)
        
        # Apply transforms
        if self.transform:
            # Keep data on CPU during transforms
            img_tensor = self.transform(img_tensor)
            if isinstance(img_tensor, torch.Tensor):
                img_tensor = img_tensor.squeeze(0)
        
        return img_tensor, label