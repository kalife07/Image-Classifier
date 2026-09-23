from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


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
