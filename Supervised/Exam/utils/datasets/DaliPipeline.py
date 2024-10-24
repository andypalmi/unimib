import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import os
from typing import Dict, List, Tuple

class FoodDALIPipeline(Pipeline):
    def __init__(self, image_paths: List[str], labels: List[int], 
                 batch_size: int, num_threads: int, device_id: int, 
                 is_training: bool = True):
        super().__init__(batch_size=batch_size, 
                        num_threads=num_threads, 
                        device_id=device_id)
        
        self.input = fn.readers.file(
            files=image_paths,
            labels=labels,
            random_shuffle=is_training,  # Only shuffle during training
            name="reader"
        )
        
        # Common transformations
        self.decode = fn.decoders.image(
            self.input[0],  # Unpack the file handle from input tuple
            device="mixed",
            output_type=types.DALIImageType.RGB
        )
        
        # Define scale based on training/validation/test
        scale = (0.8, 1.0) if is_training else (0.99, 1.0)
        
        self.resize = fn.random_resized_crop(
            self.decode,
            size=(256, 256),
            random_area=scale,
            random_aspect_ratio=(0.99, 1.0),
            device="gpu"
        )
        
        # Training-specific augmentations
        if is_training:
            self.flip = fn.flip(
                self.resize,
                horizontal=fn.random.coin_flip(probability=0.5),
                device="gpu"
            )
            self.colorjit = fn.color_twist(
                self.flip,
                hue=0.1,
                device="gpu"
            )
            self.blur = fn.gaussian_blur(
                self.colorjit,
                window_size=fn.random.uniform(range=(3, 7)),
                sigma=fn.random.uniform(range=(0.1, 0.5)),
                device="gpu"
            )
            self.processed = self.blur
        else:
            self.processed = self.resize
        
        # Normalization for RGB images (per channel)
        mean_tensor = fn.external_source(
            source=[[0.485 * 255, 0.456 * 255, 0.406 * 255]],
            device="gpu",
            layout="C"
        )
        stddev_tensor = fn.external_source(
            source=[[0.229 * 255, 0.224 * 255, 0.225 * 255]],
            device="gpu",
            layout="C"
        )
        self.normalized = fn.normalize(
            self.processed,
            mean=mean_tensor,  # Per-channel mean values (for R, G, B)
            stddev=stddev_tensor,  # Per-channel stddev values (for R, G, B)
            device="gpu",
            dtype=types.DALIDataType.FLOAT  # Ensure output is float
        )
        
        # Arrange data into NCHW format
        self.transformed = fn.transpose(
            self.normalized,
            perm=[2, 0, 1],
            device="gpu"
        )

    def define_graph(self):
        images, labels = self.input
        return [self.transformed, labels.gpu()]

class FoodDatasetDALI:
    def __init__(self, 
                 image_paths_labels: Dict[str, str],
                 batch_size: int,
                 num_threads: int = 4,
                 device_id: int = 0,
                 is_training: bool = True):
        
        self.image_paths = list(image_paths_labels.keys())
        # Convert string labels to integers
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(image_paths_labels.values())))}
        self.labels = [self.label_to_idx[image_paths_labels[path]] for path in self.image_paths]
        
        self.pipeline = FoodDALIPipeline(
            image_paths=self.image_paths,
            labels=self.labels,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            is_training=is_training
        )
        
        self.pipeline.build()
        
        self.iterator = DALIGenericIterator(
            pipelines=self.pipeline,
            output_map=['images', 'labels'],
            size=len(self.image_paths),
            auto_reset=True
        )
    
    def __iter__(self):
        return self.iterator
    
    def __len__(self):
        return len(self.iterator)
    
    @property
    def idx_to_label(self):
        """Return mapping from index to original label string"""
        return {idx: label for label, idx in self.label_to_idx.items()}

def create_dataloaders(
    train_image_paths_labels: Dict[str, str],
    val_image_paths_labels: Dict[str, str],
    test_image_paths_labels: Dict[str, str],
    batch_size: int = 32,
    num_threads: int = 4,
    device_id: int = 0
) -> Tuple[FoodDatasetDALI, FoodDatasetDALI, FoodDatasetDALI]:
    """
    Create train, validation, and test datasets using DALI.
    
    Args:
        train_image_paths_labels: Dictionary mapping training image paths to labels
        val_image_paths_labels: Dictionary mapping validation image paths to labels
        test_image_paths_labels: Dictionary mapping test image paths to labels
        batch_size: Batch size for all datasets
        num_threads: Number of threads for parallel processing
        device_id: GPU device ID to use
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = FoodDatasetDALI(
        image_paths_labels=train_image_paths_labels,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_training=True
    )
    
    val_dataset = FoodDatasetDALI(
        image_paths_labels=val_image_paths_labels,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_training=False
    )
    
    test_dataset = FoodDatasetDALI(
        image_paths_labels=test_image_paths_labels,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        is_training=False
    )
    
    return train_dataset, val_dataset, test_dataset