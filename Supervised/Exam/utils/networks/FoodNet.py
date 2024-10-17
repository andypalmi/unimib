import torch
import torch.nn as nn
import torch.nn.functional as F

class FoodNet(nn.Module):
    def __init__(self, num_classes=251):
        super(FoodNet, self).__init__()
        
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Depthwise Separable Convolutions
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, groups=16)  # Depthwise
        self.conv2_pointwise = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1)  # Pointwise
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)  # Depthwise
        self.conv3_pointwise = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)  # Pointwise
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)  # Depthwise
        self.conv4_pointwise = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)  # Pointwise
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128)  # Depthwise
        self.conv5_pointwise = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)  # Pointwise
        self.bn5 = nn.BatchNorm2d(256)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(256, num_classes)
    
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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=dilation_rates[0], bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[1], dilation=dilation_rates[1], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[2], dilation=dilation_rates[2], bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rates[3], dilation=dilation_rates[3], bias=False)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )
        
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Atrous convolutions with different dilation rates
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        
        # Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate the features
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        # Apply 1x1 convolution to reduce dimensionality
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)

class FoodNetASPP(nn.Module):
    def __init__(self, num_classes=251):
        super(FoodNetASPP, self).__init__()
        # Define initial convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # ASPP layer
        self.aspp = ASPP(in_channels=64, out_channels=128)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2) # Downsample
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2) # Downsample
        
        # ASPP for multi-scale context
        x = self.aspp(x)
        
        # Final classification layer
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x