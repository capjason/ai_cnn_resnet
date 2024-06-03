from torch import nn
import torch.nn.functional as F
import torch

class BottleNeck(nn.Module):
    compression = 4
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()
        compressed_channels = out_channels // self.compression
        self.conv1 = nn.Conv2d(in_channels, compressed_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(compressed_channels)
        self.conv2 = nn.Conv2d(compressed_channels, compressed_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(compressed_channels)
        self.conv3 = nn.Conv2d(compressed_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = F.relu(x)
        return x



class BabyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = self.make_layer(in_channels=3, out_channels=64,stride=2,num_blocks=3) # 32 -> 16
        self.layer1 = self.make_layer(in_channels=64,out_channels=128,stride=2,num_blocks=4) # 16->8
        self.layer2 = self.make_layer(in_channels=128,out_channels=256,stride=2,num_blocks=6) # 8->4
        self.layer3 = self.make_layer(in_channels=256,out_channels=512,stride=2,num_blocks=3) # 4->2
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    
    def make_layer(self,in_channels,out_channels,num_blocks,stride):
        layers = []
        layers.append(
            BottleNeck(in_channels=in_channels,out_channels=out_channels,stride=stride)
        )

        for _ in range(1,num_blocks):
            layers.append(
                BottleNeck(in_channels=out_channels,out_channels=out_channels)
            )
        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x