import os
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from cv2 import cvtColor, imread, IMREAD_GRAYSCALE, COLOR_BGR2RGB

def create_and_save_tiles(img_path, mask_path, tiles_dim=512, final_dim=256, output_dir="data"):
    """
    Creates and saves tiles from an image and its corresponding mask.

    Args:
        img_path (str): The path to the image file.
        mask_path (str): The path to the mask file.
        tiles_dim (int, optional): The dimension of each tile. Defaults to 512.
        final_dim (int, optional): The final dimension of each tile after resizing. Defaults to 256.
        output_dir (str, optional): The directory to save the tiles. Defaults to "data".

    Returns:
        img_tiles (Tensor): The tensor containing the image tiles.
        mask_tiles (Tensor): The tensor containing the mask tiles.
    """
    # Extract the original index from the image path
    original_index = os.path.basename(img_path)[:3]

    # Load image and mask
    img = cvtColor(imread(img_path), COLOR_BGR2RGB)
    mask = imread(mask_path, IMREAD_GRAYSCALE)

    # Convert to tensor
    transform = transforms.ToTensor()
    img = transform(img)
    mask = transform(mask)

    # Check if resizing is necessary
    if img.shape[1] % tiles_dim != 0 or img.shape[2] % tiles_dim != 0:
        # Round down to the nearest multiple of tiles_dim
        new_height = img.shape[1] // tiles_dim * tiles_dim
        new_width = img.shape[2] // tiles_dim * tiles_dim
        new_shp = (new_height, new_width)

        # Resize the image and mask
        img = F.interpolate(img.unsqueeze(0), size=new_shp, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=new_shp, mode='nearest').squeeze(0)
  
    img_tiles = img.unfold(1, tiles_dim, tiles_dim).unfold(2, tiles_dim, tiles_dim)
    img_tiles = img_tiles.contiguous().view(3, -1, tiles_dim, tiles_dim).permute(1, 0, 2, 3)
    
    mask_tiles = mask.unfold(1, tiles_dim, tiles_dim).unfold(2, tiles_dim, tiles_dim)
    mask_tiles = mask_tiles.contiguous().view(-1, tiles_dim, tiles_dim)

    # Resize tiles to 256x256
    resize_dim = final_dim
    img_tiles = F.interpolate(img_tiles, size=(resize_dim, resize_dim), mode='bilinear', align_corners=False)
    mask_tiles = F.interpolate(mask_tiles.unsqueeze(1), size=(resize_dim, resize_dim), mode='nearest').squeeze(1)

    # Create output directories if they don't exist
    img_output_dir = os.path.join(output_dir, f"{tiles_dim}x{tiles_dim}", "images")
    mask_output_dir = os.path.join(output_dir, f"{tiles_dim}x{tiles_dim}", "masks")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Save tiles
    for i, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
        img_tile_path = os.path.join(img_output_dir, f"{original_index}_{i}.png")
        mask_tile_path = os.path.join(mask_output_dir, f"{original_index}_{i}.png")
        save_image(img_tile, img_tile_path)
        save_image(mask_tile, mask_tile_path)

    return img_tiles, mask_tiles