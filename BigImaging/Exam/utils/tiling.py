import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from cv2 import cvtColor, imread, IMREAD_GRAYSCALE, COLOR_BGR2RGB

def create_and_save_tiles(split: str, img_path: str, mask_path: str, tiles_dim=512, final_dim=256, output_dir="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_index = os.path.basename(img_path).split('.')[0]

    img = cvtColor(imread(img_path), COLOR_BGR2RGB)
    mask = imread(mask_path, IMREAD_GRAYSCALE)

    transform = transforms.ToTensor()
    img = transform(img).to(device)
    mask = transform(mask).to(device)

    if img.shape[1] % tiles_dim != 0 or img.shape[2] % tiles_dim != 0:
        new_height = img.shape[1] // tiles_dim * tiles_dim
        new_width = img.shape[2] // tiles_dim * tiles_dim
        new_shp = (new_height, new_width)

        img = F.interpolate(img.unsqueeze(0), size=new_shp, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=new_shp, mode='nearest').squeeze(0)
  
    img_tiles = img.unfold(1, tiles_dim, tiles_dim).unfold(2, tiles_dim, tiles_dim)
    img_tiles = img_tiles.contiguous().view(3, -1, tiles_dim, tiles_dim).permute(1, 0, 2, 3)
    
    mask_tiles = mask.unfold(1, tiles_dim, tiles_dim).unfold(2, tiles_dim, tiles_dim)
    mask_tiles = mask_tiles.contiguous().view(-1, tiles_dim, tiles_dim)

    resize_dim = final_dim
    img_tiles = F.interpolate(img_tiles, size=(resize_dim, resize_dim), mode='bilinear', align_corners=False)
    mask_tiles = F.interpolate(mask_tiles.unsqueeze(1), size=(resize_dim, resize_dim), mode='nearest').squeeze(1)

    img_output_dir = os.path.join(output_dir, f"{tiles_dim}x{tiles_dim}", split, "images")
    mask_output_dir = os.path.join(output_dir, f"{tiles_dim}x{tiles_dim}", split, "masks")
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    for i, (img_tile, mask_tile) in enumerate(zip(img_tiles, mask_tiles)):
        img_tile_path = os.path.join(img_output_dir, f"{original_index}_{i}.png")
        mask_tile_path = os.path.join(mask_output_dir, f"{original_index}_{i}.png")
        save_image(img_tile.cpu(), img_tile_path)
        save_image(mask_tile.cpu(), mask_tile_path)

    return img_tiles, mask_tiles

def process_row(row):
    tiles_dim = 1000
    final_dim = 256
    create_and_save_tiles(split=row['split'], img_path=row['img'], mask_path=row['mask'], 
                          tiles_dim=tiles_dim, final_dim=final_dim, output_dir=f'data/tiles_{final_dim}')