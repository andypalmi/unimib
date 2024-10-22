import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the transformation pipeline
train_transforms = A.Compose([
    # Randomly crop and resize to 256x256
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=1.0),
    
    # Horizontal Flip
    A.HorizontalFlip(p=0.5),
    
    # Random brightness and contrast
    A.RandomBrightnessContrast(p=0.2),

    A.HueSaturationValue(p=0.2),  # Randomly change hue, saturation, and value

    A.RandomGamma(p=0.2),  # Randomly change gamma
    
    # Normalize using ImageNet mean and std
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # Convert image to a PyTorch tensor
    ToTensorV2(),
])

valtest_transforms = A.Compose([
    # Resize to 256x256
    A.Resize(height=256, width=256, always_apply=True),
    
    # Normalize using ImageNet mean and std
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # Convert image to a PyTorch tensor
    ToTensorV2(),
])

ssl_transforms = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.5, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(0.4, 0.4, 0.4, 0.1),
    A.ToGray(p=0.1),
    A.GaussianBlur(blur_limit=(3,7), sigma_limit=(0.1, 2.0), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])