import torch
import torch.nn as nn
import torch.nn.functional as F

class FoodNet(nn.Module):
    def __init__(self, num_classes=251):
        super(FoodNet, self).__init__()
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise Separable Convolutions
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)  # Depthwise
        self.conv2_pointwise = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)  # Pointwise
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)  # Depthwise
        self.conv3_pointwise = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)  # Pointwise
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)  # Depthwise
        self.conv4_pointwise = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # Pointwise
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256)  # Depthwise
        self.conv5_pointwise = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)  # Pointwise
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Initial Conv Layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise Separable Convolutions
        x = F.relu(self.bn2(self.conv2_pointwise(F.relu(self.conv2(x)))))
        x = F.relu(self.bn3(self.conv3_pointwise(F.relu(self.conv3(x)))))
        x = F.relu(self.bn4(self.conv4_pointwise(F.relu(self.conv4(x)))))
        x = F.relu(self.bn5(self.conv5_pointwise(F.relu(self.conv5(x)))))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x
