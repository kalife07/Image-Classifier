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

Or, instead of the above, just [Docker](https://docs.docker.com/get-docker/), which installs all of these for you (see [Usage](#usage)).

## Dataset setup

The `dataset/` folder is not committed to this repo (it's listed in `.gitignore`) because it's too large. You only need it if you want to retrain the model with `model.py` (see [Retrain the model](#optional-retrain-the-model)). To get it:

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

Run all commands from the project root.

The trained weights are committed to this repo in `digit_classifier.pth`, so you don't need to train the model before using it: you can go straight to predicting. A sample image of a handwritten 5, `sample_digit.jpg`, is included in the project root and is used by default, so you can try the model without downloading the dataset. You can run the code either with Docker, which installs Python and all the dependencies for you, or directly with your local Python.

### Option A: with Docker

Docker installs Python 3.14 and every dependency in `requirements.txt` inside the image, so you only need [Docker](https://docs.docker.com/get-docker/) installed.

1. Build the image (needed once, and again after changing the code). It copies `app/`, `digit_classifier.pth` and `sample_digit.jpg`, so you don't need the dataset:

   ```
   docker build -t digit-classifier .
   ```

2. Run a prediction on the default image, `sample_digit.jpg`:

   ```
   docker run --rm digit-classifier
   ```

   Any argument after the image name is passed on to `predict.py`. To test another image, mount its folder into the container and pass its path:

   ```
   docker run --rm -v "$(pwd)/my_images:/project/my_images" digit-classifier my_images/digit.jpg
   ```

The container has no display, so it only prints the predicted digit and its probability; the matplotlib figure isn't shown. The image runs on the CPU only (it installs the CPU build of PyTorch to keep it small), which is fine for predictions. To use an NVIDIA GPU instead, see [Running Docker on an NVIDIA GPU](#running-docker-on-an-nvidia-gpu). By default the image only runs `predict.py`; to retrain the model inside Docker, see [Retrain the model](#optional-retrain-the-model).

#### Running Docker on an NVIDIA GPU

`get_device()` already picks an NVIDIA GPU (CUDA) when one is available, so no code changes are needed. The default image can't use it, though, because it installs the CPU-only build of PyTorch. To run on the GPU:

1. Check that your machine meets the requirements:
   - An NVIDIA GPU. AMD and Intel GPUs aren't supported by CUDA, and very old NVIDIA cards may not be supported by current PyTorch builds.
   - Linux, or Windows with Docker Desktop using WSL2. This doesn't work on macOS: Docker runs containers in a Linux virtual machine there, and the Mac's GPU can't be passed through to it.
   - A recent NVIDIA driver that supports the CUDA version you pick in step 2. You don't need to install the CUDA toolkit itself; the PyTorch packages include the CUDA libraries they need.
   - On Linux, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Docker Desktop on Windows handles GPU access through WSL2.
2. In the `Dockerfile`, replace the CPU index URL in the `pip install` command with a CUDA one, for example:

   ```dockerfile
   RUN pip install -r requirements.txt \
           --index-url https://download.pytorch.org/whl/cu128 \
           --extra-index-url https://pypi.org/simple
   ```

   Pick the CUDA version (`cu128` here) that matches your driver from the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). This makes the image several GB larger.
3. Rebuild the image:

   ```
   docker build -t digit-classifier .
   ```

4. Add `--gpus all` to your `docker run` commands, for example:

   ```
   docker run --rm --gpus all digit-classifier
   ```

If any of these is missing, the container doesn't crash: `get_device()` falls back to the CPU, so it still runs, only slower.

### Option B: with your local Python

1. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Test the model on an image:

   ```
   python app/predict.py
   ```

   This loads the saved weights from `digit_classifier.pth` without retraining and shows a matplotlib figure with the image next to the predicted probability of each digit. By default it uses `sample_digit.jpg` from the project root. To test a different image, pass its path:

   ```
   python app/predict.py path/to/your_digit.jpg
   ```

   To change the default image instead, edit the path in the `test_image` line of `main()` in `app/predict.py`:

   ```python
   test_image = sys.argv[1] if len(sys.argv) > 1 else "sample_digit.jpg"
   ```

   If you change it in the Docker setup too, make sure the new image is copied into the image (see the `COPY` lines in the `Dockerfile`) and rebuild it.

### (Optional) Retrain the model

The model is already trained, but you can retrain it yourself, for example after changing the number of epochs or the dataset. In both cases you first need to download the full dataset from Kaggle and place it in `dataset/` (see [Dataset setup](#dataset-setup)).

#### With your local Python

```
python app/model.py
```

This trains the model and overwrites `digit_classifier.pth` with the new weights and class names. `predict.py` then uses your new weights. If `digit_classifier.pth` is missing, `predict.py` tells you to run `model.py` first.

#### With Docker

The Docker image doesn't include the dataset by default, so a few changes are needed:

1. Add a line to the `Dockerfile` that copies the dataset into the image, for example after `COPY app/ app/`:

   ```dockerfile
   COPY dataset/ dataset/
   ```

2. In `.dockerignore`, remove the `dataset/train/` and `dataset/valid/` lines. Otherwise Docker skips those folders and `COPY dataset/` doesn't include the images `model.py` trains on.
3. Rebuild the image:

   ```
   docker build -t digit-classifier .
   ```

4. Run `model.py` instead of `predict.py`, and mount `digit_classifier.pth` so the new weights are saved to your project folder and not only inside the container:

   ```
   docker run --rm --entrypoint python -v "$(pwd)/digit_classifier.pth:/project/digit_classifier.pth" digit-classifier app/model.py
   ```

5. Rebuild the image again so `predict.py` in Docker uses the new weights.

By default the Docker image uses the CPU build of PyTorch, so training in it is much slower than on a GPU. If you have an NVIDIA GPU, see [Running Docker on an NVIDIA GPU](#running-docker-on-an-nvidia-gpu) and add `--gpus all` to the `docker run` command in step 4. The dataset also makes the image much larger, so once you're done you may want to undo steps 1 and 2.
