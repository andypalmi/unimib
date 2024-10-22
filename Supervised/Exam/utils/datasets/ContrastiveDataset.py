import cv2
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, image_paths_labels: dict[str, str], transform=None):
        self.image_paths_labels = image_paths_labels
        self.transform = transform    # Augmentation transformations

    def __len__(self):
        return len(self.image_paths_labels)

    def __getitem__(self, idx):
        image_path = list(self.image_paths_labels.keys())[idx]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Apply two different augmentations to the same image
        if self.transform:
            image1 = self.transform(image=image)['image']
            image2 = self.transform(image=image)['image']
        else:
            image1 = image2 = image

        return image1, image2
