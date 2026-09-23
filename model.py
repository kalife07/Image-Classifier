import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt #for visualization
import pandas as pd
import numpy as np

class PlayingCardDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = ImageFolder(root=root_dir, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label

    @property
    def classes(self):
        return self.data.classes

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

data_dir = 'dataset/train'

dataset = PlayingCardDataset(root_dir=data_dir, transform=transform)
image, label = dataset[100]

dataloader = DataLoader(dataset)





