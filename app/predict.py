import argparse
import sys

import torch
from PIL import Image
import matplotlib.pyplot as plt #for visualization

from classifier import DigitClassifier, get_device, transform, preprocess_photo, CHECKPOINT_PATH


def load_model(device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    #pretrained=False: our own trained weights replace the ImageNet ones anyway
    model = DigitClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names

def preprocess_image(image_path, transform, raw=False):
    image = Image.open(image_path).convert("RGB")
    #crop photos to the digit and whiten the paper so they look like the training images
    model_input = image if raw else preprocess_photo(image)
    return image, model_input, transform(model_input).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, model_input, probabilities, class_names):
    fig, axarr = plt.subplots(1, 3, figsize=(18, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].set_title("Original")
    axarr[0].axis("off")

    # Display what the model sees after preprocessing
    axarr[1].imshow(model_input)
    axarr[1].set_title("Model input")
    axarr[1].axis("off")

    # Display predictions
    axarr[2].barh(class_names, probabilities)
    axarr[2].set_xlabel("Probability")
    axarr[2].set_title("Class Predictions")
    axarr[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


def main():
    if not CHECKPOINT_PATH.exists():
        print(f'No trained model found at {CHECKPOINT_PATH}, run model.py first')
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Predict the digit in an image")
    parser.add_argument("image", nargs="?", default="sample_digit.jpg", help="path to the image") #replace default with desired test path
    parser.add_argument("--raw", action="store_true", help="skip photo preprocessing (crop + background whitening)")
    args = parser.parse_args()

    device = get_device()
    model, class_names = load_model(device)

    original_image, model_input, image_tensor = preprocess_image(args.image, transform, raw=args.raw)
    probabilities = predict(model, image_tensor, device)
    best = probabilities.argmax()
    print(f'Predicted class: {class_names[best]} with probability {probabilities[best]:.4f}')
    visualize_predictions(original_image, model_input, probabilities, class_names)


if __name__ == "__main__":
    main()