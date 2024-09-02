# SkySegmenter: Semantic Segmentation of Urban Drone Imagery

## Overview

SkySegmenter is a project aimed at performing semantic segmentation on urban drone imagery using the DeepLabv3+ architecture with various encoders. The project includes a dashboard for visualizing segmentation results and comparing different models.

## Project Structure
├── dashboard.py
├── models/
    └── best/
├── utils/
    ├── evaluate.py 
    ├── tiling.py 
    ├── train.py 
    └── utils.py
├── data/ 
    ├── original_images/ 
    ├── label_images_semantic/ 
    ├── RGB_color_image_masks/ 
    └── tiles_256/
├── main.py
└── README.md


## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in [`requirements.txt`])

### Installation

1. **Clone the repository:**

2. **Create a virtual environment and activate it:**

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Running the Dashboard

To run the dashboard, simply execute the `dashboard.py` script. Make sure to place your trained models in the [`models/best`] directory.

```sh
python dashboard.py