import torch
from PIL import Image

from backend.config.config import DEVICE
from backend.data.preprocessing import get_train_transforms
from backend.explainability.gradcam import load_client_model

import cv2
import numpy as np
import matplotlib.pyplot as plt

from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data
from backend.config.config import RESULTS_DIR


def segment_leaf(test_index=40):

    print("\nStarting Background Segmentation...")

    # Load dataset without transformations
    dataset = load_dataset(transform=None)

    # Recreate the same test split
    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    # Get original image and label
    image, true_label = test_dataset[test_index]

    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width, channels = image_bgr.shape

    # Initialize GrabCut mask
    mask = np.zeros((height, width), np.uint8)

    # Define rectangle around the leaf
    rectangle = (
        int(width * 0.05),
        int(height * 0.05),
        int(width * 0.90),
        int(height * 0.90)
    )

    # Initialize background and foreground models
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut
    cv2.grabCut(
        image_bgr,
        mask,
        rectangle,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Create binary leaf mask
    leaf_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0
    ).astype(np.uint8)

    # Generate leaf-only image
    leaf_only = image * leaf_mask[:, :, np.newaxis]

    # Generate background-only image
    background_only = image * (1 - leaf_mask[:, :, np.newaxis])
        # Load personalized FedBN model
    print("\nLoading FedBN model...")

    model = load_client_model(
        client_id=0,
        num_classes=len(dataset.classes)
    )

    model.eval()

    transform = get_train_transforms()

    def predict_image(img, name):

        pil_image = Image.fromarray(img)

        input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        top_probability, top_index = torch.max(probabilities, dim=1)

        predicted_class = dataset.classes[top_index.item()]

        print(f"\n{name}")
        print("Predicted Class:", predicted_class)
        print(f"Probability: {top_probability.item() * 100:.2f}%")

        return predicted_class

    print("\n========== BACKGROUND DEPENDENCE ANALYSIS ==========")

    predict_image(image, "ORIGINAL IMAGE")
    predict_image(leaf_only, "LEAF ONLY")
    predict_image(background_only, "BACKGROUND ONLY")

    print("\n====================================================")

    # Save visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(leaf_only)
    plt.title("Leaf Only")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(background_only)
    plt.title("Background Only")
    plt.axis("off")

    output_dir = RESULTS_DIR / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"image_{test_index}_segmentation.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("\nSegmentation completed!")
    print("True Class:", dataset.classes[true_label])
    print("Saved:", output_path)


if __name__ == "__main__":
    segment_leaf(test_index=40)