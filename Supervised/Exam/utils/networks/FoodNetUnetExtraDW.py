import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.networks.ExtraDWBlock import ExtraDWBlock

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Initialize the decoder block with proper channel handling for skip connections.
        
        Args:
            in_channels (int): Number of input channels from the layer below
            skip_channels (int): Number of channels from the skip connection
            out_channels (int): Number of output channels
        """
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # The block will process concatenated channels (in_channels + skip_channels)
        total_channels = in_channels + skip_channels
        self.block = nn.Sequential(
            ExtraDWBlock(total_channels, out_channels),
            ExtraDWBlock(out_channels, out_channels)
        )
    
    def forward(self, x, skip=None):
        """
        Forward pass of the decoder block.
        
        Args:
            x (torch.Tensor): Input tensor from the layer below
            skip (torch.Tensor, optional): Skip connection tensor from the encoder
        
        Returns:
            torch.Tensor: Output tensor after processing
        """
        x = self.upsample(x)
        if skip is not None:
            # Ensure the spatial dimensions match before concatenating
            if x.shape[-2:] != skip.shape[-2:]:
                print(f'Shapes do not match: {x.shape} vs {skip.shape}')
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x

class FoodNetUnetExtraDW(nn.Module):
    def __init__(self, num_classes=251, embedding_dim=256):
        super().__init__()

        # Encoder stages
        self.enc1 = self._make_encoder_stage(3, 16, 2, 2)     # 256 -> 128
        self.enc2 = self._make_encoder_stage(16, 24, 2, 2)    # 128 -> 64
        self.enc3 = self._make_encoder_stage(24, 32, 2, 2)    # 64 -> 32
        self.enc4 = self._make_encoder_stage(32, 64, 3, 2)    # 32 -> 16
        self.enc5 = self._make_encoder_stage(64, 96, 2, 2)    # 16 -> 8
        
        # Decoder stages
        self.dec4 = DecoderBlock(96, 64, 64)    # 96 from below + 64 from skip -> 64 output
        self.dec3 = DecoderBlock(64, 32, 32)    # 64 from below + 32 from skip -> 32 output
        self.dec2 = DecoderBlock(32, 24, 24)    # 32 from below + 24 from skip -> 24 output
        self.dec1 = DecoderBlock(24, 16, 16)    # 24 from below + 16 from skip -> 16 output
        
        # Global pooling and projection head
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection heads for SSL
        self.projection1 = nn.Sequential(
            nn.Linear(16, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        self.projection2 = nn.Sequential(
            nn.Linear(16, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_encoder_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ExtraDWBlock(in_channels, out_channels, stride)) # If stride=2, HxW are halved -> [H/2xW/2]
        for _ in range(1, num_blocks):
            layers.append(ExtraDWBlock(out_channels, out_channels)) # Default stride=1 -> [HxW] are preserved
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        # Encoder path with skip connections
        x1 = self.enc1(x)     # Skip connection 1
        x2 = self.enc2(x1)    # Skip connection 2
        x3 = self.enc3(x2)    # Skip connection 3
        x4 = self.enc4(x3)    # Skip connection 4
        x5 = self.enc5(x4)    # Bridge
        
        # Decoder path
        d4 = self.dec4(x5, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
                
        # Global pooling
        out = self.pool(d1)
        return out.flatten(1)
    
    def forward(self, x1, x2=None, mode='train_supervised'):
        if mode == 'train_ssl':
            # Self-supervised training mode
            f1 = self.forward_features(x1)
            f2 = self.forward_features(x2)
            
            z1 = self.projection1(f1)
            z2 = self.projection2(f2)
            
            return z1, z2
        
        elif mode == 'train_supervised' or mode == 'eval':
            # Evaluation mode (classification)
            f = self.forward_features(x1)
            z = self.projection1(f)
            return self.classifier(z)
        
        else:
            # Feature extraction mode
            f = self.forward_features(x1)
            return f