# Playing Card Classifier

A PyTorch image classifier that uses a convolutional neural network to identify playing cards from images, via transfer learning.

## What `model.py` does

- **Dataset** (`PlayingCardDataset`): wraps `torchvision.datasets.ImageFolder` to load card images from `dataset/train`, `dataset/validation`, and `dataset/test` directories, where each subfolder name is treated as a class label (e.g. one folder per card such as "ace of spades").
- **Preprocessing**: resizes every image to 128x128 and converts it to a tensor via `torchvision.transforms`.
- **Model** (`SimpleCardClassifier`): a neural network built from a pretrained `efficientnet_b0` backbone (via `timm`, a convolutional neural network) with its classification head removed, followed by a single linear layer that maps the 1280-dimensional feature output to the number of card classes (53 by default).
- **Training loop**: trains the model for 5 epochs using the Adam optimizer and cross-entropy loss, running on GPU (CUDA) if available, otherwise CPU. After each epoch it evaluates on the validation set and prints the training/validation loss.

## Requirements

- Python 3.10
- `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `matplotlib`, `tqdm`

## Dataset setup

The `dataset/` folder is not committed to this repo (it's listed in `.gitignore`) because it's too large. Before running `model.py`, you need to download it yourself:

1. Download the dataset from Kaggle: [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification?resource=download)
2. Unzip it and place it in a `dataset/` folder in the project root, matching the layout below.

## Data layout

```
dataset/
  train/
    <class_name>/*.jpg
  validation/
    <class_name>/*.jpg
  test/
    <class_name>/*.jpg
```

## Usage

```
python model.py
```
