from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm

#trained weights are saved at the repo root, regardless of where scripts are run from
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "digit_classifier.pth"

transform = transforms.Compose([
    #(height, width): keeps the ~3:2 portrait shape of the 90x140 digit images, and both are multiples of 32 for efficientnet
    transforms.Resize((192, 128)),
    transforms.ToTensor()
])


class DigitClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)

        return output


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
