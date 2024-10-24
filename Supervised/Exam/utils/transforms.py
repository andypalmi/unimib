import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the transformation pipeline
train_transforms = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(hue=(-0.1, 0.1)),
    A.GaussianBlur(blur_limit=(3,7), sigma_limit=0, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

valtest_transforms = A.Compose([
    # Resize to 256x256
    A.RandomResizedCrop(height=256, width=256, scale=(1.0, 1.0), p=1.0),
    
    # Normalize using ImageNet mean and std
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # Convert image to a PyTorch tensor
    ToTensorV2(),
])

ssl_transforms = A.Compose([
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(),
    A.GaussianBlur(blur_limit=(3,7), sigma_limit=0, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])