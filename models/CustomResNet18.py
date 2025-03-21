import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes, freeze_layers=False):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        
        # Adjust the first convolutional layer for 32x32 input
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()  # Remove maxpool for smaller images
        
        # Freeze layers if specified
        if freeze_layers:
            for param in self.resnet18.parameters():
                param.requires_grad = False
                # Define the classification head with dropout
        
        in_features = 512        
                
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),  # Reduce dimensionality
            nn.ReLU(),
            nn.Dropout(p=0.9),   # Dropout layer
            nn.Linear(256, num_classes)   # Final output layer
        )
        
        self.resnet18.fc = self.classifier
        
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)