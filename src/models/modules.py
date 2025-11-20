import torch.nn as nn
import torchvision.models as models


class PaperModel(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights='DEFAULT' if pretrained else None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)