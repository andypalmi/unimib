import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Albumentations transformations
train_transform = A.Compose([
    # A.Resize(new_height, new_width, p=1.0),  # Resize the image to the desired shape
    A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    A.VerticalFlip(p=0.5),  # Apply vertical flip with 50% probability
    A.RandomBrightnessContrast(p=0.2),  # Randomly change brightness and contrast
    A.OneOf([
        A.GaussianBlur(p=1.0),  # Apply Gaussian blur
        A.MotionBlur(p=1.0),  # Apply motion blur
    ], p=0.2),  # Apply one of the blur operations with 20% probability
    A.HueSaturationValue(p=0.2),  # Randomly change hue, saturation, and value
    A.RandomGamma(p=0.2),  # Randomly change gamma
    A.CLAHE(p=0.2),  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize the image
    ToTensorV2(),  # Convert image and mask to PyTorch tensors
])

valtest_transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    A.VerticalFlip(p=0.5),  # Apply vertical flip with 50% probability
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize the image
    ToTensorV2(),  # Convert image and mask to PyTorch tensors
])