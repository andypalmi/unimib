import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import torch
import os
from typing import Dict, List, Tuple

class FoodDALIPipeline(Pipeline):
    def __init__(self, image_paths: List[str], labels: List[int], 
                 batch_size: int, num_threads: int, device_id: int, 
                 is_training: bool = True, is_ssl: bool = False):
        super().__init__(batch_size=batch_size, 
                        num_threads=num_threads, 
                        device_id=device_id)
        
        self.is_training = is_training
        self.is_ssl = is_ssl
        self.input = fn.readers.file(
            files=image_paths,
            labels=labels,
            random_shuffle=is_training,
            name="reader"
        )
        
        self.decode = fn.decoders.image(
            self.input[0],
            device="mixed",
            output_type=types.DALIImageType.RGB
        )

    def create_augmented_view(self, image):
        """Creates a single augmented view with random transformations"""
        # Basic crop and normalize with random mirror
        augmented = fn.crop_mirror_normalize(
            image,
            dtype=types.DALIDataType.FLOAT,
            output_layout='HWC',
            crop=(256, 256),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip(),
            device="gpu"
        )
        
        # Color augmentations
        augmented = fn.color_twist(
            augmented,
            brightness=fn.random.uniform(range=(0.6, 1.4)),
            contrast=fn.random.uniform(range=(0.6, 1.4)),
            saturation=fn.random.uniform(range=(0.6, 1.4)),
            hue=fn.random.uniform(range=(-0.1, 0.1)),
            device="gpu"
        )
        
        # Random blur
        augmented = fn.gaussian_blur(
            augmented,
            window_size=fn.random.choice([3, 5, 7]),
            sigma=fn.random.uniform(range=(0.1, 0.5), dtype=types.DALIDataType.FLOAT),
            device="gpu"
        )
        
        # Channel transpose for PyTorch
        return fn.transpose(
            augmented,
            perm=[2, 0, 1],
            device="gpu"
        )

    def define_graph(self):
        images, labels = self.input
        
        if not self.is_ssl:
            # Original supervised learning pipeline
            if self.is_training:
                processed = self.create_augmented_view(self.decode)
            else:
                # For validation/testing, no augmentations except normalization
                processed = fn.crop_mirror_normalize(
                    self.decode,
                    dtype=types.DALIDataType.FLOAT,
                    output_layout='HWC',
                    crop=(256, 256),
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    mirror=False,
                    device="gpu"
                )
                processed = fn.transpose(
                    processed,
                    perm=[2, 0, 1],
                    device="gpu"
                )
            return [processed, fn.cast(self.input[1], dtype=types.DALIDataType.INT64).gpu()]
        else:
            # Self-supervised learning pipeline - create two different views
            view1 = self.create_augmented_view(self.decode)
            view2 = self.create_augmented_view(self.decode)
            return [view1, view2]

def create_dataloaders(
    train_image_paths_labels: Dict[str, str],
    val_image_paths_labels: Dict[str, str],
    test_image_paths_labels: Dict[str, str],
    batch_size: int = 32,
    num_threads: int = 10,
    device_id: int = 0,
) -> Tuple[DALIGenericIterator, DALIGenericIterator, DALIGenericIterator, DALIGenericIterator]:
    """
    Create train, validation, and test DALI iterators.
    
    Args:
        train_image_paths_labels: Dictionary mapping training image paths to labels
        val_image_paths_labels: Dictionary mapping validation image paths to labels
        test_image_paths_labels: Dictionary mapping test image paths to labels
        batch_size: Batch size for all datasets
        num_threads: Number of threads for parallel processing
        device_id: GPU device ID to use
    
    Returns:
        Tuple of (train_iterator, val_iterator, test_iterator)
    """
    def create_iterator(image_paths_labels: Dict[str, str], is_training: bool, is_ssl_training: bool = False) -> DALIGenericIterator:
        image_paths = list(image_paths_labels.keys())
        if not is_ssl_training:
            label_to_idx = {label: idx for idx, label in enumerate(sorted(set(image_paths_labels.values())))}
            labels = [label_to_idx[image_paths_labels[path]] for path in image_paths]
        else:
            # For SSL, we don't need actual labels, just dummy values
            labels = [0] * len(image_paths)
        
        pipeline = FoodDALIPipeline(
            image_paths=image_paths,
            labels=labels,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            is_training=is_training,
            is_ssl=is_ssl_training
        )
        
        pipeline.build()
        
        output_map = ['view1', 'view2'] if is_ssl_training else ['images', 'labels']
        
        return DALIGenericIterator(
            pipelines=pipeline,
            output_map=output_map,
            size=len(image_paths),
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.FILL
        )
    
    train_ssl_iterator = create_iterator(train_image_paths_labels, is_training=True, is_ssl_training=True)
    train_iterator = create_iterator(train_image_paths_labels, is_training=True)
    val_iterator = create_iterator(val_image_paths_labels, is_training=False)
    test_iterator = create_iterator(test_image_paths_labels, is_training=False)
    
    return train_ssl_iterator, train_iterator, val_iterator, test_iterator