import sys

import torch
from PIL import Image
import matplotlib.pyplot as plt #for visualization

from classifier import DigitClassifier, get_device, transform, CHECKPOINT_PATH


def load_model(device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    #pretrained=False: our own trained weights replace the ImageNet ones anyway
    model = DigitClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


def main():
    if not CHECKPOINT_PATH.exists():
        print(f'No trained model found at {CHECKPOINT_PATH}, run model.py first')
        sys.exit(1)

    test_image = sys.argv[1] if len(sys.argv) > 1 else "add_tests/test_digit6.jpg" #replace path with desired test path

    device = get_device()
    model, class_names = load_model(device)

    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)
    best = probabilities.argmax()
    print(f'Predicted class: {class_names[best]} with probability {probabilities[best]:.4f}')
    visualize_predictions(original_image, probabilities, class_names)


if __name__ == "__main__":
    main()