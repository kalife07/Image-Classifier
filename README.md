# Handwritten Digit Classifier

A PyTorch image classifier that uses a convolutional neural network to recognize handwritten digits (0-9) from images, via transfer learning.

> This project started as a playing card classifier. The model and training pipeline were kept as-is and retargeted to handwritten digits; only the dataset, the number of classes and the input image size changed.

## Project structure

- **`app/dataset.py`** (`HandwrittenDigitDataset`): wraps `torchvision.datasets.ImageFolder` to load digit images from `dataset/train`, `dataset/valid`, and `dataset/test` directories, where each subfolder name is treated as a class label (one folder per digit, `0` to `9`).
- **`app/classifier.py`**: code shared by training and prediction:
  - **Preprocessing** (`transform`): resizes every image to 192x128 (height x width) and converts it to a tensor via `torchvision.transforms`. This keeps the roughly 3:2 portrait shape of the original 90x140 digit images, and both sides are multiples of 32, which suits EfficientNet.
  - **Model** (`DigitClassifier`): a neural network built from a pretrained `efficientnet_b0` backbone (via `timm`, a convolutional neural network) with its classification head removed, followed by a single linear layer that maps the 1280-dimensional feature output to the number of digit classes (10 by default).
  - **Device selection** (`get_device`): uses an NVIDIA GPU (CUDA) if available, then Apple Silicon GPU (MPS), otherwise CPU.
  - **`CHECKPOINT_PATH`**: location of the saved model file, `digit_classifier.pth` in the project root.
- **`app/model.py`** (training): trains the model for 5 epochs using the Adam optimizer and cross-entropy loss. After each epoch it evaluates on the validation set and prints the training/validation loss. When training finishes, it saves the learned weights and class names to `digit_classifier.pth`.
- **`app/predict.py`** (testing): loads the saved weights from `digit_classifier.pth`, predicts the digit in a single image, and shows a matplotlib figure with the image next to the probability of each digit. It doesn't retrain.

## Requirements

- Python 3.14
- `torch`, `torchvision`, `timm`, `numpy`, `matplotlib`, `pillow`, `tqdm` (pinned versions in `requirements.txt`, install with `pip install -r requirements.txt`)

## Dataset setup

The `dataset/` folder is not committed to this repo (it's listed in `.gitignore`) because it's too large. Before running `model.py`, you need to download it yourself:

1. Download the dataset from Kaggle: [Handwritten Digit](https://www.kaggle.com/datasets/panghalvishesh/handwritten-digit)
2. Unzip it and place it in a `dataset/` folder in the project root, matching the layout below.

The split used here has about 22,000 training images and about 4,700 images each for validation and test, spread fairly evenly across the 10 digits.

## Data layout

```
dataset/
  train/
    0/*.jpg
    1/*.jpg
    ...
    9/*.jpg
  valid/
    0/*.jpg ... 9/*.jpg
  test/
    0/*.jpg ... 9/*.jpg
```

## Usage

Run all commands from the project root (the folder containing `dataset/`).

### 1. Train the model (once)

```
python app/model.py
```

This trains the model and saves its weights and class names to `digit_classifier.pth` in the project root. The file is ignored by git.

To retrain the model, for example after changing the number of epochs or the dataset, simply rerun `python app/model.py`. It overwrites `digit_classifier.pth` with the new weights.

### 2. Test the model on an image

```
python app/predict.py
```

This loads the saved weights from `digit_classifier.pth` without retraining and shows a matplotlib figure with the image next to the predicted probability of each digit. By default it uses `dataset/test/5/Five_full (10).jpg`. To test a different image, pass its path:

```
python app/predict.py "dataset/test/3/Three_full (100).jpg"
```

If `digit_classifier.pth` doesn't exist yet, `predict.py` tells you to run `model.py` first.
