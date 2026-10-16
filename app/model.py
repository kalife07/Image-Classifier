import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HandwrittenDigitDataset
from classifier import DigitClassifier, get_device, transform, train_transform, CHECKPOINT_PATH


def save_checkpoint(model, class_names):
    #weights + class names so predict.py can skip training
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, CHECKPOINT_PATH)


def train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
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
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        #validation loop
        with torch.no_grad():
            for inputs, labels in tqdm(validation_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_losses.append(val_loss / len(validation_loader.dataset))
        val_accuracy = val_correct / len(validation_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        #keep the best epoch, not just the last one
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            save_checkpoint(model, train_loader.dataset.classes)
            print(f'Saved best model so far to {CHECKPOINT_PATH}')

    return train_losses, val_losses


def main():
    train_folder = 'dataset/train'
    validation_folder = 'dataset/valid'

    #augmentation only on training images, validation measures the real (unaugmented) images
    train_dataset = HandwrittenDigitDataset(train_folder, transform=train_transform)
    validation_dataset = HandwrittenDigitDataset(validation_folder, transform=transform)

    #worker processes run the augmentation in parallel so the GPU isn't waiting on the CPU
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True)

    device = get_device()
    print(f'Using device: {device}')

    model = DigitClassifier(num_classes=len(train_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    #lowers the learning rate smoothly to near zero by the last epoch, for finer final updates
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    #the best epoch is saved to CHECKPOINT_PATH during training
    train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, num_epochs, device)


if __name__ == "__main__":
    main()
