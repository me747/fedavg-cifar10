import torch.nn as nn
import torchvision.models as models

class ResNet18_Fed(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__() # parent class constructor
        # loading ResNet-18 without pre-trained weights
        self.model = models.resnet18(weights=None)
        # modify first conv layer for 32x32 CIFAR-10 images
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes) # replace final FC layer to match CIFAR-10's output classes

    def forward(self, x):
        return self.model(x)    
