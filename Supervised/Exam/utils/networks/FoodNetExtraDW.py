import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.networks.ExtraDWBlock import ExtraDWBlock

class FoodNetExtraDW(nn.Module):
    def __init__(self, num_classes=251, embedding_dim=512):
        super().__init__()
        
        # Main stages
        self.stage1 = self._make_stage(3, 16, 2, 2, 7)   # 256 -> 126
        self.pool1 = nn.MaxPool2d(2, 2)                 # 126 -> 122
        self.stage2 = self._make_stage(16, 32, 2, 2)    # 122 -> 61
        self.stage3 = self._make_stage(32, 64, 2, 2)    # 61 -> 31
        self.stage4 = self._make_stage(64, 128, 3, 2)    # 31 -> 16
        # self.stage4 = self._make_stage(64, 96, 3, 2)    # 32 -> 16
        
        # Global pooling and projection head
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Two separate projection heads for contrastive learning
        self.projection1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        
        self.projection2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        
        # Classification head to use during supervised training
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int, kernel_size=3):
        """
        Creates a stage consisting of a sequence of ExtraDWBlock layers.

        Args:
            in_channels (int): Number of input channels for the first block.
            out_channels (int): Number of output channels for each block.
            num_blocks (int): Number of blocks in the stage.
            stride (int): Stride for the first block.
            kernel_size (int, optional): Kernel size for the first block. Default is 3.

        Returns:
            nn.Sequential: A sequential container of ExtraDWBlock layers.
        """
        layers = []
        layers.append(ExtraDWBlock(in_channels, out_channels, kernel_size, stride))
        for _ in range(1, num_blocks):
            layers.append(ExtraDWBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return x.flatten(1)
    
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