import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import base64
import cv2
import numpy as np
import torch
import os
import pandas as pd
import torch
from torchvision.transforms import transforms
from utils.evaluate import load_model_from_checkpoint, evaluate_model
from utils.utils import read_class_colors
# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory set to:", os.getcwd())

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload Image'),
        multiple=False
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[],
        placeholder='Select a model'
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(contents, tiles_dim=1000, final_dim=256):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Calculate the new dimensions
    new_height = (img_height // tiles_dim) * tiles_dim
    new_width = (img_width // tiles_dim) * tiles_dim

    new_shape = (new_height, new_width)

    print(f'\nold size {img_width} {img_height}, \nnew size: ({new_width}, {new_height})\n')

    # Resize the image
    img = cv2.resize(img, (new_width, new_height))

    tiles = []
    for y in range(0, new_height, tiles_dim):
        for x in range(0, new_width, tiles_dim):
            tile = img[y:y + tiles_dim, x:x + tiles_dim]
            tile = cv2.resize(tile, (final_dim, final_dim))
            if tile.size > 0:
                tiles.append(tile)
    return tiles, new_shape

def get_model_options(models_folder):
    model_options = []
    for filename in os.listdir(models_folder):
        if filename.endswith('.pt') and '1000' in filename:
            parts = filename.split('_')
            print(f'parts: {parts}')
            if 'mobilenet_v2' in filename:
                arch = parts[-5]
                encoder_name = '_'.join(parts[-2:]).split('.')[0]
            else:
                arch = parts[-4]
                encoder_name = parts[-1].split('.')[0]
            print(f'encoder_name: {encoder_name}')
            tiles_dim = parts[8]
            model_options.append({'label': '_'.join([arch, encoder_name]) + ' (Dim: ' + tiles_dim + ')', 'value': filename})
    return model_options

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def prepare_tiles(tiles, device='cuda' if torch.cuda.is_available() else 'cpu'):
    normalized_tiles = []
    for tile in tiles:
        tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0  # Reshape for PyTorch
        tile_tensor = normalize(tile_tensor)
        normalized_tiles.append(tile_tensor)
    
    # Stack the list of tensors into a single tensor
    batch_tensor = torch.stack(normalized_tiles)
    
    # Move the tensor to the specified device
    batch_tensor = batch_tensor.to(device)
    
    return batch_tensor

def reconstruct_mask(mask_tiles, img_shape, tiles_dim, final_dim=256):
    mask_width = img_shape[0] // tiles_dim * final_dim
    mask_height = img_shape[1] // tiles_dim * final_dim
    reconstructed_mask = np.zeros((mask_width, mask_height), dtype=np.uint8)
    tile_index = 0
    num_tiles = mask_tiles.shape[0]
    print(f'Number of tiles: {num_tiles}')
    print(f'Image shape: {img_shape}\nMask shape: {(mask_width, mask_height)}')
    
    for i in range(0, mask_width, final_dim):
        for j in range(0, mask_height, final_dim):
            if tile_index >= num_tiles:
                print(f'Error: tile_index {tile_index} out of bounds for mask_tiles with size {num_tiles}')
                break
            reconstructed_mask[i:i + final_dim, j:j + final_dim] = mask_tiles[tile_index]
            tile_index += 1
    
    print(f'Reconstructed mask shape: {reconstructed_mask.shape}')
    return reconstructed_mask

class_colors = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (180, 120, 120),
    3: (160, 150, 20),
    4: (140, 140, 140),
    5: (61, 230, 250),
    6: (0, 82, 255),
    7: (255, 0, 245),
    8: (255, 235, 0),
    9: (4, 250, 7),
}

labels_colors, colors, num_classes = read_class_colors('data/ColorMasks/ColorPalette-Values.csv')

def colorize_mask(predicted_mask):
    colorized_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        colorized_mask[predicted_mask == class_id] = color
    return colorized_mask

def load_color_palette(csv_file):
    color_palette = pd.read_csv(csv_file)
    return color_palette

def update_model_options(_):
    return [{'label': model['label'], 'value': model['value']} for model in get_model_options('models/best')]

@app.callback(
    Output('model-dropdown', 'options'),
    Input('upload-image', 'contents')
)
def update_dropdown_options(contents):
    return update_model_options(contents)

@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    Input('model-dropdown', 'value')
)
def update_output(contents, model_filename):
    if contents and model_filename:
        tiles_dim = int(model_filename.split('_')[8])
        tiles, new_shape = parse_contents(contents=contents, tiles_dim=tiles_dim)
        print(f'Loaded {len(tiles)} tiles')
        model_path = os.path.join('models', 'best', model_filename)
        print(f'Loading model')
        model, _, _, _ = load_model_from_checkpoint(model_path)
        batch_tiles = prepare_tiles(tiles)
        with torch.no_grad():
            predicted_masks = model(batch_tiles)
            predicted_masks = torch.softmax(predicted_masks, dim=1).argmax(dim=1).cpu().numpy()
        predicted_mask = reconstruct_mask(predicted_masks, new_shape, tiles_dim, final_dim=256)
        colorized_mask = colorize_mask(predicted_mask)
        # Display the colorized mask
        return html.Img(src=f'data:image/png;base64,{base64.b64encode(cv2.imencode(".png", colorized_mask)[1]).decode()}')
    return 'Upload an image and select a model.'

if __name__ == '__main__':
    app.run_server(debug=True)