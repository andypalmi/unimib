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
                 is_training: bool = True):
        super().__init__(batch_size=batch_size, 
                        num_threads=num_threads, 
                        device_id=device_id)
        
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

        self.crop_mirror_norm = fn.crop_mirror_normalize(
            self.decode,
            dtype=types.DALIDataType.FLOAT,
            output_layout='HWC',
            crop=(256, 256),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=fn.random.coin_flip(),
            device="gpu"
        )
        
        if is_training:
            self.colorjit = fn.color_twist(
                self.crop_mirror_norm,
                hue=0.1,
                device="gpu"
            )
            self.blur = fn.gaussian_blur(
                self.colorjit,
                window_size=fn.random.choice([3, 5, 7]),
                sigma=fn.random.uniform(range=(0.1, 0.5), dtype=types.DALIDataType.FLOAT),
                device="gpu"
            )
            self.processed = self.blur
        else:
            self.processed = self.crop_mirror_norm
        
        self.transformed = fn.transpose(
            self.processed,
            perm=[2, 0, 1],
            device="gpu"
        )

        self.labels = fn.cast(self.input[1], dtype=types.DALIDataType.INT64)

    def define_graph(self):
        images, labels = self.input
        return [self.transformed, self.labels.gpu()]

def create_dataloaders(
    train_image_paths_labels: Dict[str, str],
    val_image_paths_labels: Dict[str, str],
    test_image_paths_labels: Dict[str, str],
    batch_size: int = 32,
    num_threads: int = 4,
    device_id: int = 0
) -> Tuple[DALIGenericIterator, DALIGenericIterator, DALIGenericIterator]:
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
    def create_iterator(image_paths_labels: Dict[str, str], is_training: bool) -> DALIGenericIterator:
        image_paths = list(image_paths_labels.keys())
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(image_paths_labels.values())))}
        labels = [label_to_idx[image_paths_labels[path]] for path in image_paths]
        
        pipeline = FoodDALIPipeline(
            image_paths=image_paths,
            labels=labels,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            is_training=is_training
        )
        
        pipeline.build()
        
        return DALIGenericIterator(
            pipelines=pipeline,
            output_map=['images', 'labels'],
            size=len(image_paths),
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.FILL
        )
    
    train_iterator = create_iterator(train_image_paths_labels, is_training=True)
    val_iterator = create_iterator(val_image_paths_labels, is_training=False)
    test_iterator = create_iterator(test_image_paths_labels, is_training=False)
    
    return train_iterator, val_iterator, test_iterator