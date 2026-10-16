from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image

#trained weights are saved at the repo root, regardless of where scripts are run from
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "digit_classifier.pth"

#(height, width): keeps the ~3:2 portrait shape of the 90x140 digit images, and both are multiples of 32 for efficientnet
IMAGE_SIZE = (192, 128)
#the efficientnet backbone was pretrained on imagenet images normalized with these statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomPaper:
    """Make a clean digit image look like a photo of paper: tinted grey background, uneven lighting,
    speckles and sometimes ruled lines. The digit image is multiplied onto the paper, so ink stays dark."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if np.random.rand() > self.p:
            return image
        digit = np.asarray(image.convert("RGB"), dtype=float) / 255
        h, w, _ = digit.shape

        #paper colour, slightly tinted per channel
        paper = np.ones((h, w, 3)) * np.random.uniform(0.55, 0.95) * np.random.uniform(0.93, 1.0, size=3)
        #uneven lighting/shadow: a linear gradient in a random direction
        angle = np.random.uniform(0, 2 * np.pi)
        ys, xs = np.mgrid[0:h, 0:w]
        gradient = (np.cos(angle) * xs / w + np.sin(angle) * ys / h)
        paper *= (1 - np.random.uniform(0, 0.25) * (gradient - gradient.min()))[..., None]
        #paper texture
        paper *= np.random.normal(1, np.random.uniform(0, 0.04), size=(h, w, 1))

        #ruled lines, like a notebook page
        if np.random.rand() < 0.5:
            spacing = np.random.randint(h // 5, h // 2)
            offset = np.random.randint(0, spacing)
            slope = np.random.uniform(-0.05, 0.05)
            line_colour = np.array([np.random.uniform(0.6, 0.85), np.random.uniform(0.7, 0.9), np.random.uniform(0.6, 0.85)])
            on_line = ((ys - slope * xs - offset) % spacing) < np.random.randint(1, 3)
            paper[on_line] *= line_colour

        return Image.fromarray((np.clip(digit * paper, 0, 1) * 255).astype(np.uint8))


#deterministic, used for validation and prediction
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

#training only: random scale/position/rotation, paper backgrounds and lighting/contrast changes,
#so the model doesn't only learn big centred digits on clean white backgrounds
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    #fill=255: area uncovered by shrinking/shifting the digit is white, like the paper
    transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.3, 1.05), shear=5, fill=255),
    RandomPaper(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4, hue=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

#training images are 90x140 (width x height) crops where the digit fills the frame
TRAIN_ASPECT = 90 / 140


def preprocess_photo(image, pad_ratio=0.15):
    """Turn a photo of a digit on paper into a training-style image: cropped to the digit, dark ink on white.
    Runs before `transform`. Returns the image unchanged if it is already a crop or no ink is found."""
    rgb = np.asarray(image.convert("RGB"), dtype=float)
    saturation = np.asarray(image.convert("HSV"), dtype=float)[..., 1]
    gray = np.asarray(image.convert("L"), dtype=float)
    background = np.median(gray)

    #already a training-style crop: the digit covers a big share of the frame (~5-12% of pixels
    #in dataset images vs under 2% in full-page photos), re-cropping it only loses detail
    if (gray < background - 80).mean() > 0.03:
        return image

    #ink = blue pen strokes (grey paper, soft shadows and greenish ruled lines are not blue)
    #falls back to "much darker than the paper" for black ink, which would also catch specks on textured paper
    ink = (saturation > 60) & (rgb[..., 2] > rgb[..., 1] + 20)
    if ink.sum() < 20:
        ink = gray < background - 80
    if ink.sum() < 20:
        return image

    #bounding box of the ink, percentiles ignore isolated specks
    ys, xs = np.nonzero(ink)
    y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int)
    x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)

    #pad, then grow the box to the training aspect ratio around its center
    pad = int(pad_ratio * max(y1 - y0, x1 - x0))
    height = y1 - y0 + 2 * pad
    width = max(x1 - x0 + 2 * pad, int(height * TRAIN_ASPECT))
    height = max(height, int(width / TRAIN_ASPECT))
    top = (y0 + y1) // 2 - height // 2
    left = (x0 + x1) // 2 - width // 2

    #paper -> white, ink -> black
    ink_level = gray[ink].mean()
    normalized = np.clip((gray - ink_level) / max(background - ink_level, 1), 0, 1) * 255

    #crop onto a white canvas so a box running past the image edge is padded with paper, not black
    canvas = np.full((height, width), 255, dtype=np.uint8)
    src_top, src_left = max(top, 0), max(left, 0)
    src_bottom, src_right = min(top + height, gray.shape[0]), min(left + width, gray.shape[1])
    canvas[src_top - top:src_bottom - top, src_left - left:src_right - left] = normalized[src_top:src_bottom, src_left:src_right]

    return Image.fromarray(canvas).convert("RGB")


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
