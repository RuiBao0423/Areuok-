import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=3, pretrained=True,dropout=0.2,freeze=False):
        super().__init__()
        if pretrained:

            weights=EfficientNet_B0_Weights.DEFAULT
        else:
            weights=None
        self.net= efficientnet_b0(weights=weights)

        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        

        if freeze:
            for i in self.net.features.parameters():
                i.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        return x

def buildmodel(num_classes=3, pretrained=True,dropout=0.2,freeze=False):
    model = EfficientNetB0(num_classes=num_classes, pretrained=pretrained,dropout=dropout,freeze=freeze)
    return model

