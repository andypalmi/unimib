import torch
import torch.nn as nn
import torch.nn.functional as F

class ExtraDWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=4):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        
        self.skip_connection = stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # Initial DW Conv
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            
            # Expansion
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
            
            # Second DW Conv
            nn.Conv2d(expanded_channels, expanded_channels, 3, 1, 1, groups=expanded_channels, bias=False),
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

class FoodNetExtraDW(nn.Module):
    def __init__(self, num_classes=251, embedding_dim=128):
        super().__init__()
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )
        
        # Main stages
        self.stage1 = self._make_stage(16, 24, 2, 2)    # 128 -> 64
        self.stage2 = self._make_stage(24, 32, 2, 2)    # 64 -> 32
        self.stage3 = self._make_stage(32, 64, 3, 2)    # 32 -> 16
        self.stage4 = self._make_stage(64, 96, 2, 2)    # 16 -> 8
        
        # Global pooling and projection head
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Two separate projection heads for contrastive learning
        self.projection1 = nn.Sequential(
            nn.Linear(96, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        
        self.projection2 = nn.Sequential(
            nn.Linear(96, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        
        # Optional classification head (can be used after pre-training)
        self.classifier = nn.Linear(96, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ExtraDWBlock(in_channels, out_channels, stride))
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
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return x.flatten(1)
    
    def forward(self, x1, x2=None, mode='train'):
        if mode == 'train':
            # Self-supervised training mode
            f1 = self.forward_features(x1)
            f2 = self.forward_features(x2)
            
            z1 = self.projection1(f1)
            z2 = self.projection2(f2)
            
            return z1, z2
        
        elif mode == 'eval':
            # Evaluation mode (classification)
            f = self.forward_features(x1)
            return self.classifier(f)
        
        else:
            # Feature extraction mode
            f = self.forward_features(x1)
            return f