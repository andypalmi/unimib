import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtraDWBlock(nn.Module):
    """
    ExtraDWBlock is a custom neural network module that implements a depthwise separable convolutional block with an 
    optional skip connection. This block includes an initial depthwise convolution, an expansion phase, a second 
    depthwise convolution, and a projection phase.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        stride (int, optional): Stride for the initial depthwise convolution. Default is 1.
        expansion_factor (int, optional): Factor by which the number of channels is expanded during the expansion phase. Default is 4.
    Attributes:
        skip_connection (bool): Whether to use a skip connection, determined by the stride and channel dimensions.
        conv (nn.Sequential): Sequential container of the convolutional layers and activation functions.
    Methods:
        forward(x):
            Defines the computation performed at every call. If skip_connection is True, the input is added to the output 
            of the convolutional layers.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after applying the block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=4):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        
        self.skip_connection = stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # Initial DW Conv
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            # Expansion
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
            
            # Second DW Conv
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 1, 1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
            
            # Projection
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        if self.skip_connection:
            return x + self.conv(x)
        return self.conv(x)
    
'''
                 Input (H×W×C_in)
                         │
                ┌────────┴────────┐
                │                 │
        ┌───────┴───────┐ ┌──────┴────────┐
        │ DW Conv 3×3   │ │ DW Conv 3×3   │
        │ stride=1      │ │ stride=2      │
        │ H×W×C_in      │ │ H/2×W/2×C_in  │
        └───────┬───────┘ └──────┬────────┘
                │                │
                └────────┬───────┘
                         │
                ┌────────┴────────┐
                │   1×1 Conv      │
                │ (Expansion)     │
                │ H'×W'×(4C_in)   │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │  DW Conv 3×3    │
                │   stride=1      │
                │ H'×W'×(4C_in)   │
                └────────┬────────┘
                        │
                ┌────────┴────────┐
                │   1×1 Conv      │
                │  (Projection)   │
                │ H'×W'×C_out     │
                └────────┬────────┘
                         │
                       Output

        Note: H'×W' depends on initial stride:
        - If stride=1: H'=H, W'=W
        - If stride=2: H'=H/2, W'=W/2

        Skip Connection (if stride=1 and C_in=C_out):
        Input ─────────────────────────────> + ─> Output
'''