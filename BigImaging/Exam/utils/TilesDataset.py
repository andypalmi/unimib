import cv2
import torch.nn.functional as F
from torch.utils.data import Dataset

class TilesDataset(Dataset):
    """
    A custom dataset class for handling image and mask data.

    Args:
        image_paths (list): List of tuples containing image and mask file paths.
        transform (callable, optional): Optional transform to be applied to the image and mask. Default is None.
        tiles (bool, optional): Flag indicating whether to create image and mask tiles. Default is False.
        tiles_dim (int, optional): Dimension of the tiles. Default is 512.

    Returns:
        tuple: A tuple containing the image tiles and mask tiles.

    """

    def __init__(self, image_paths, transform=None, tiles=False, tiles_dim=512):
        self.image_paths = image_paths
        self.transform = transform
        self.tiles = tiles
        self.tiles_dim = tiles_dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_paths[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        if self.tiles:
            img, mask = self.create_tiles(img, mask, self.tiles_dim)

        return img, mask
    
    def create_tiles(self, img, mask, tiles_dim):
        """
        Create image and mask tiles.

        Args:
            img (numpy.ndarray): Input image.
            mask (numpy.ndarray): Input mask.
            tiles_dim (int): Dimension of the tiles.

        Returns:
            tuple: A tuple containing the image tiles and mask tiles.

        """

        # Check if resizing is necessary
        if img.shape[1] % tiles_dim != 0 or img.shape[2] % tiles_dim != 0:
            # Round down to the nearest multiple of tiles_dim
            new_height = img.shape[1] // tiles_dim * tiles_dim
            new_width = img.shape[2] // tiles_dim * tiles_dim
            new_shp = (new_height, new_width)

            # Resize the image and mask
            img = F.interpolate(img.unsqueeze(0), size=new_shp, mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=new_shp, mode='nearest').squeeze(0).squeeze(0)
        
        # Create img tiles and mask tiles
        img_tiles = img.unfold(1, tiles_dim, tiles_dim).unfold(2, tiles_dim, tiles_dim)
        img_tiles = img_tiles.contiguous().view(3, -1, tiles_dim, tiles_dim).permute(1, 0, 2, 3)
        
        mask_tiles = mask.unfold(0, tiles_dim, tiles_dim).unfold(1, tiles_dim, tiles_dim)
        mask_tiles = mask_tiles.contiguous().view(-1, tiles_dim, tiles_dim)

        # Resize tiles if necessary
        if tiles_dim > 256:
            resize_dim = 256
            img_tiles = F.interpolate(img_tiles, size=(resize_dim, resize_dim), mode='bilinear', align_corners=False)
            mask_tiles = F.interpolate(mask_tiles.unsqueeze(1), size=(resize_dim, resize_dim), mode='nearest').squeeze(1)

        return img_tiles, mask_tiles