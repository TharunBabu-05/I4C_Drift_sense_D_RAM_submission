import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SiameseEncoder(nn.Module):
    """
    Lightweight ResNet-based shared encoder for the Siamese network.
    Extracts a 128-D embedding from 1-channel grayscale input.
    Suitable for CPU and edge deployment.
    """
    def __init__(self, embedding_dim=128):
        super(SiameseEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual blocks
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        # Global pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_dim)
        
        # Normalization for cosine similarity
        self.l2_norm = True

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
            
        return x

import torchvision.models as models

class MobileNetV3SiameseEncoder(nn.Module):
    """
    MobileNetV3-based shared encoder for the Siamese network.
    Extracts a 128-D embedding from 1-channel grayscale input.
    """
    def __init__(self, embedding_dim=128):
        super(MobileNetV3SiameseEncoder, self).__init__()
        
        # Load MobileNetV3 Small without pre-trained weights (since we use 1-channel SEM images)
        mobilenet = models.mobilenet_v3_small(weights=None)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        original_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=False
        )
        
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # We need to map the output of MobileNetV3-small features (usually 576 channels) to our embedding_dim
        self.fc = nn.Linear(576, embedding_dim)
        
        self.l2_norm = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
            
        return x
