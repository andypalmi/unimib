import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random

def load_data(split=0.2, random_state=69420, verbose=True):
    """
    Load data and return dictionaries for training, validation, and test sets and their respective transform counts.
    Duplicates images based on class distribution to balance the dataset.
    
    Returns:
        train_dict (dict): Dictionary where keys are image paths and values are labels for the training set.
        val_dict (dict): Dictionary where keys are image paths and values are labels for the validation set.
        test_dict (dict): Dictionary where keys are image paths and values are labels for the test set.
        train_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the training set.
        val_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the validation set.
        test_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the test set.
    """
    # Load the data
    columns = ['image', 'label']
    labels_path = 'data/labels'
    train_df = pd.read_csv(os.path.join(labels_path, 'train_info.csv'), names=columns, header=None)
    val_df = pd.read_csv(os.path.join(labels_path, 'val_info.csv'), names=columns, header=None)

    # Add full paths to images
    images_path = 'data/images'
    train_path = 'train_set'
    val_path = 'val_set'
    train_df['image'] = images_path + '/' + train_path + '/' + train_df['image']
    val_df['image'] = images_path + '/' + val_path + '/' + val_df['image']

    # Split the training data
    train_df, val_split_df = train_test_split(
        train_df, 
        test_size=split, 
        random_state=random_state, 
        stratify=train_df['label']
    )

    # Function to duplicate images based on class counts
    def duplicate_images(df):
        """
        Duplicate images in each class to match the size of the largest class.
        """
        class_counts = Counter(df['label'])
        max_count = max(class_counts.values())
        new_data = []
        
        for label in class_counts.keys():
            # Get all images for this class
            class_images = df[df['label'] == label]['image'].tolist()
            if not class_images:  # Skip if no images for this class
                continue
                
            current_count = len(class_images)
            
            # Add all original images
            new_data.extend([{'image': img, 'label': label} for img in class_images])
            
            # Calculate how many additional copies we need
            needed_copies = max_count - current_count
            
            # Duplicate images randomly until we reach max_count
            for _ in range(needed_copies):
                img = random.choice(class_images)
                new_data.append({'image': img, 'label': label})
        return pd.DataFrame(new_data)

    # Create balanced datasets by duplicating images
    train_df_balanced = duplicate_images(train_df)
    val_df_balanced = duplicate_images(val_split_df)
    test_df_balanced = duplicate_images(val_df)

    # Create dictionaries
    train_dict = dict(zip(train_df_balanced['image'], train_df_balanced['label']))
    val_dict = dict(zip(val_df_balanced['image'], val_df_balanced['label']))
    test_dict = dict(zip(test_df_balanced['image'], test_df_balanced['label']))

    if verbose:
        print(f'Starting Training samples: {len(train_df)} | Validation samples: {len(val_split_df)} | Test samples: {len(val_df)}')
        print(f'Actual   Training samples: {len(train_df_balanced)} | Validation samples: {len(val_df_balanced)} | Test samples: {len(test_df_balanced)}')

    return (train_dict, val_dict, test_dict)