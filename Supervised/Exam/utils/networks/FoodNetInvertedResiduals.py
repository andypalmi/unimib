import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Expansion phase
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Linear projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class FoodNetInvertedResidualsSSL(nn.Module):
    def __init__(self, embedding_size=512):
        super(FoodNetInvertedResidualsSSL, self).__init__()
        
        # Initial convolution
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted Residual blocks configuration
        # [in_channels, out_channels, stride, expand_ratio, repeat]
        self.block_config = [
            [16, 16, 1, 1, 1],   # Layer 1
            [16, 24, 2, 6, 2],   # Layer 2
            [24, 32, 2, 6, 2],   # Layer 3
        ]
        
        # Build inverted residual blocks
        self.layers = self._make_layers()
        
        # Global pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(32, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layers(self):
        layers = []
        for c in self.block_config:
            in_c, out_c, stride, expand_ratio, repeat = c
            for i in range(repeat):
                stride = stride if i == 0 else 1
                layers.append(InvertedResidual(in_c, out_c, stride, expand_ratio))
                in_c = out_c
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_one(self, x):
        x = self.convBlock1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize embeddings

    def forward(self, img1, img2):
        # Get embeddings for both images
        z1 = self.forward_one(img1)
        z2 = self.forward_one(img2)
        return z1, z2

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())