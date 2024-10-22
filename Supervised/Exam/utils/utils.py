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

    # Get class distribution for each split
    train_counts = Counter(train_df['label'])
    val_counts = Counter(val_split_df['label'])
    test_counts = Counter(val_df['label'])

    # Function to duplicate images based on class counts
    def duplicate_images(df, counts):
        new_data = []
        for label in counts.keys():
            # Get all images for this class
            class_images = df[df['label'] == label]['image'].tolist()
            if not class_images:  # Skip if no images for this class
                continue
            # Calculate how many times to duplicate each image
            count = counts[label]
            # Duplicate images randomly
            for _ in range(count):
                img = random.choice(class_images)
                new_data.append({'image': img, 'label': label})
        return pd.DataFrame(new_data)

    # Create balanced datasets by duplicating images
    train_df_balanced = duplicate_images(train_df, train_counts)
    val_df_balanced = duplicate_images(val_split_df, val_counts)
    test_df_balanced = duplicate_images(val_df, test_counts)

    # Create dictionaries
    train_dict = dict(zip(train_df_balanced['image'], train_df_balanced['label']))
    val_dict = dict(zip(val_df_balanced['image'], val_df_balanced['label']))
    test_dict = dict(zip(test_df_balanced['image'], test_df_balanced['label']))

    if verbose:
        print('Actual training samples:', len(train_dict))
        print('Actual validation samples:', len(val_dict))
        print('Actual test samples:', len(test_dict))

    return (train_dict, val_dict, test_dict)