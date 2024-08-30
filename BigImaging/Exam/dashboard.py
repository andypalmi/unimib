import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64
import cv2
import numpy as np
import torch
import os
import pandas as pd
from torchvision.transforms import transforms
from utils.evaluate import load_model_from_checkpoint
from utils.utils import read_class_colors
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the current working directory to the directory where main.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
logger.info("Current working directory set to: %s", os.getcwd())

app = dash.Dash(__name__)

app.layout = html.Div(id='app-container', children=[
    html.Div([
        html.Img(src='assets/drone_white.png', style={'height': '50px', 'margin-right': '10px'}),
        html.H1('SkySegmenter', style={'display': 'inline-block', 'vertical-align': 'middle'}),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload Image'),
        multiple=False,
        style={'display': 'block', 'margin': '0 auto'}
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[],
        placeholder='Select a model',
        style={'width': '50%', 'margin': '20px auto', 'color': 'black'}
    ),
    html.Div(
        dcc.Slider(
            id='alpha-slider',
            min=0,
            max=1,
            step=0.01,
            value=0.5,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True, "transform": "percentageFormat"},
        ),
        style={'width': '50%', 'margin': '20px auto'}
    ),
    html.Div(id='output-image-upload', style={'text-align': 'center'}),
])

def parse_contents(contents, tiles_dim=1000, final_dim=256):
    """
    Parses the uploaded image contents and splits it into tiles.

    Args:
        contents (str): The base64 encoded contents of the uploaded image.
        tiles_dim (int): The dimension of the tiles.
        final_dim (int): The final dimension of the resized tiles.

    Returns:
        list: A list of image tiles.
        tuple: The new shape of the image.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
    img_height, img_width, _ = img.shape

    # Calculate the new dimensions
    new_height = (img_height // tiles_dim) * tiles_dim
    new_width = (img_width // tiles_dim) * tiles_dim
    new_shape = (new_height, new_width)

    logger.info('Old size: (%d, %d), New size: (%d, %d)', img_width, img_height, new_width, new_height)

    # Resize the image
    img = cv2.resize(img, (new_width, new_height))

    tiles = []
    for y in range(0, new_height, tiles_dim):
        for x in range(0, new_width, tiles_dim):
            tile = img[y:y + tiles_dim, x:x + tiles_dim]
            tile = cv2.resize(tile, (final_dim, final_dim))
            if tile.size > 0:
                tiles.append(tile)
    return tiles, new_shape, img

def get_model_options(models_folder):
    """
    Retrieves the available model options from the specified folder.

    Args:
        models_folder (str): The folder containing the model files.

    Returns:
        list: A list of dictionaries containing model labels and values.
    """
    model_options = []
    for filename in os.listdir(models_folder):
        if filename.endswith('.pt') and '1000' in filename:
            parts = filename.split('_')
            logger.info('Parts: %s', parts)
            if 'mobilenet_v2' in filename:
                arch = parts[-5]
                encoder_name = '_'.join(parts[-2:]).split('.')[0]
            else:
                arch = parts[-4]
                encoder_name = parts[-1].split('.')[0]
            logger.info('Encoder name: %s', encoder_name)
            tiles_dim = parts[8]
            model_options.append({'label': '_'.join([arch, encoder_name]) + ' (Dim: ' + tiles_dim + ')', 'value': filename})
    return model_options

def prepare_tiles(tiles, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Preprocesses a list of image tiles for input to a PyTorch model.

    Args:
        tiles (list): A list of image tiles.
        device (str, optional): The device to use for tensor operations. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        torch.Tensor: A batch tensor containing the preprocessed image tiles.
    """
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    # Convert list of tiles to a single NumPy array
    tiles_np = np.stack(tiles, axis=0)
    
    # Convert to tensor and move to device
    batch_tensor = torch.tensor(tiles_np).permute(0, 3, 1, 2).float() / 255.0  # Reshape for PyTorch
    
    # Apply normalization
    batch_tensor = normalize(batch_tensor).to(device)
    
    return batch_tensor

def reconstruct_mask(mask_tiles, img_shape, tiles_dim, final_dim=256):
    """
    Reconstructs a mask from a set of mask tiles.

    Args:
        mask_tiles (ndarray): Array of mask tiles.
        img_shape (tuple): Shape of the original image.
        tiles_dim (int): Dimension of the tiles.
        final_dim (int, optional): Dimension of the final mask tiles. Defaults to 256.

    Returns:
        ndarray: Reconstructed mask.
    """
    mask_width = img_shape[0] // tiles_dim * final_dim
    mask_height = img_shape[1] // tiles_dim * final_dim
    reconstructed_mask = np.zeros((mask_width, mask_height), dtype=np.uint8)
    tile_index = 0
    num_tiles = mask_tiles.shape[0]
    logger.info('Number of tiles: %d', num_tiles)
    logger.info('Image shape: %s, Mask shape: (%d, %d)', img_shape, mask_width, mask_height)
    
    for i in range(0, mask_width, final_dim):
        for j in range(0, mask_height, final_dim):
            if tile_index >= num_tiles:
                logger.error('Error: tile_index %d out of bounds for mask_tiles with size %d', tile_index, num_tiles)
                break
            reconstructed_mask[i:i + final_dim, j:j + final_dim] = mask_tiles[tile_index]
            tile_index += 1
    
    logger.info('Reconstructed mask shape: %s', reconstructed_mask.shape)
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
    """
    Colorizes a predicted mask by assigning colors to different class IDs.

    Args:
        predicted_mask (numpy.ndarray): The predicted mask to be colorized.

    Returns:
        numpy.ndarray: The colorized mask with assigned colors for each class ID.
    """
    colorized_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        colorized_mask[predicted_mask == class_id] = color
    return colorized_mask

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlays the colorized mask on the original image with adjustable alpha transparency.

    Args:
        image (numpy.ndarray): The original image.
        mask (numpy.ndarray): The colorized mask.
        alpha (float): The alpha transparency for the mask.

    Returns:
        numpy.ndarray: The image with the mask overlay.
    """
    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    return overlay

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
    Input('model-dropdown', 'value'),
    Input('alpha-slider', 'value')
)
def update_output(contents, model_filename, alpha):
    if contents and model_filename:
        try:
            tiles_dim = int(model_filename.split('_')[8])
            tiles, new_shape, original_image = parse_contents(contents=contents, tiles_dim=tiles_dim)
            logger.info('Loaded %d tiles', len(tiles))
            model_path = os.path.join('models', 'best', model_filename)
            logger.info('Loading model from %s', model_path)
            model, _, _, _ = load_model_from_checkpoint(model_path)
            batch_tiles = prepare_tiles(tiles)
            with torch.no_grad():
                predicted_masks = model(batch_tiles)
                predicted_masks = torch.softmax(predicted_masks, dim=1).argmax(dim=1).cpu().numpy()
            predicted_mask = reconstruct_mask(predicted_masks, new_shape, tiles_dim, final_dim=256)
            colorized_mask = colorize_mask(predicted_mask)
            resized_original_image = cv2.resize(original_image, (colorized_mask.shape[1], colorized_mask.shape[0]))
            overlay_image = overlay_mask_on_image(resized_original_image, colorized_mask, alpha)
            # Display the overlay image
            return html.Img(src=f'data:image/png;base64,{base64.b64encode(cv2.imencode(".png", overlay_image)[1]).decode()}')
        except Exception as e:
            logger.error('Error in update_output: %s', str(e))
            return 'An error occurred during processing.'
    return 'Upload an image and select a model.'

if __name__ == '__main__':
    app.run_server(debug=True)