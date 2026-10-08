import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HandwrittenDigitDataset
from classifier import DigitClassifier, get_device, transform, CHECKPOINT_PATH


def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, device):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        #training loop
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_losses.append(running_loss / len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        #validation loop
        with torch.no_grad():
            for inputs, labels in tqdm(validation_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_losses.append(val_loss / len(validation_loader.dataset))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


def main():
    train_folder = 'dataset/train'
    validation_folder = 'dataset/valid'

    train_dataset = HandwrittenDigitDataset(train_folder, transform=transform)
    validation_dataset = HandwrittenDigitDataset(validation_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    device = get_device()
    print(f'Using device: {device}')

    model = DigitClassifier(num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs, device)

    #save weights + class names so predict.py can skip training
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": train_dataset.classes,
    }, CHECKPOINT_PATH)
    print(f'Saved model to {CHECKPOINT_PATH}')


if __name__ == "__main__":
    main()
