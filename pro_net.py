import torch
import torch.nn as nn
import torch.nn.functional as F

class unfog_net(nn.Module):
    def __init__(self):
        super(unfog_net, self).__init__()
        # 1x1 Conv Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, padding=0,bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, padding=0,bias=True)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1, padding=0,bias=True)
        self.conv4 = nn.Conv2d(96, 32, kernel_size=1, padding=0,bias=True)
        self.conv5 = nn.Conv2d(128, 3, kernel_size=1, padding=0,bias=True)  # Output layer
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Pooling Layers
        self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(7, stride=1, padding=3)
        
    def forward(self, x):
        # Block 1
        x1 = self.conv1(x)# 32
        
        # Block 2
        x2 = F.relu(self.bn1(self.conv2(x1)))
        x2 = self.pool1(x2)
        x2 = torch.cat([x1, x2], dim=1)
        
        # Block 3
        x3 = self.conv3(x2)
        x3 = self.pool2(F.relu(self.bn2(x3)))
        x3 = torch.cat([x2, x3], dim=1)
        
        
        # Block 4
        x4 = self.conv4(x3)
        x4 = self.pool2(F.relu(self.bn3(x4)))
        x4 = torch.cat([x3,x4], dim=1)
        
        
        # Output
        out = self.conv5(x4)
        clean_image = F.relu((out * x) - out + 1) 
        return clean_image

# Creating the model and moving it to GPU if available
