import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def load_data(split=0.2, random_state=69420, verbose=True):
    """
    Load data and return dictionaries for training, validation, and test sets and their respective transform counts.

    Returns:
        train_dict (dict): Dictionary where keys are image paths and values are labels for the training set.
        val_dict (dict): Dictionary where keys are image paths and values are labels for the validation set.
        test_dict (dict): Dictionary where keys are image paths and values are labels for the test set.
        train_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the training set.
        val_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the validation set.
        test_transform_dict (dict): Dictionary where keys are labels and values are the number of augmentations for the test set.
    """

    columns = ['image', 'label']
    labels_path = 'data/labels'
    train_df = pd.read_csv(os.path.join(labels_path, 'train_info.csv'), names=columns, header=None)
    val_df = pd.read_csv(os.path.join(labels_path, 'val_info.csv'), names=columns, header=None)

    images_path = 'data/images'
    train_path = 'train_set'
    val_path = 'val_set'
    train_df['image'] = images_path + '/' + train_path + '/' + train_df['image']
    val_df['image'] = images_path + '/' + val_path + '/' + val_df['image']

    # Split the training data into 80% training and 20% validation
    train_df, val_split_df = train_test_split(train_df, test_size=split, random_state=random_state, stratify=train_df['label'])

    # Create dictionaries for train, validation, and test sets
    train_dict = dict(zip(train_df['image'], train_df['label']))
    val_dict = dict(zip(val_split_df['image'], val_split_df['label']))
    test_dict = dict(zip(val_df['image'], val_df['label']))

    # Recompute transform counts based on the actual data splits
    train_transform_dict = dict(Counter(train_df['label']))
    val_transform_dict = dict(Counter(val_split_df['label']))
    test_transform_dict = dict(Counter(val_df['label']))

    if verbose:
        print('Training set length :', len(train_transform_dict))
        print('\nValidation set length :', len(val_transform_dict))
        print('\nTest set length :', len(test_transform_dict))

    return train_dict, val_dict, test_dict, train_transform_dict, val_transform_dict, test_transform_dict