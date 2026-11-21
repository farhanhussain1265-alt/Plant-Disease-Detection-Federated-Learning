import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import Subset
from backend.config.config import (
    DEVICE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NUM_CLIENTS
)
from backend.models.cnn_model import PlantDiseaseModel
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(
            self.save_activations
        )

    def save_activations(self, module, input, output):

        self.activations = output

        output.register_hook(self.save_gradients)

    def save_gradients(self, gradient):

        self.gradients = gradient

    def generate(self, image, target_class):

        self.model.zero_grad()

        # Forward pass
        output = self.model(image)

        # Get score for the selected class
        score = output[:, target_class]

        # Backward pass
        score.backward()

        # Calculate channel-wise importance weights
        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = torch.sum(
            weights * self.activations,
            dim=1
        )

        # Remove negative activations
        cam = F.relu(cam)

        # Resize heatmap to original image dimensions
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam = cam.detach().cpu().numpy()

        # Normalize heatmap
        cam = cam - np.min(cam)

        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam
def load_client_model(client_id, num_classes):

    print(f"\nLoading Client {client_id} Model...")

    model = PlantDiseaseModel(num_classes=num_classes)

    shared_path = CHECKPOINT_DIR / "fedbn_shared_parameters.pth"

    bn_path = CHECKPOINT_DIR / f"fedbn_client_{client_id}_bn.pth"

    shared_state = torch.load(
        shared_path,
        map_location=DEVICE
    )

    bn_state = torch.load(
        bn_path,
        map_location=DEVICE
    )

    model_state = model.state_dict()

    model_state.update(shared_state)
    model_state.update(bn_state)

    model.load_state_dict(model_state)

    model.to(DEVICE)
    model.eval()

    print("Client Model Loaded Successfully!")

    return model


def generate_gradcam(client_id=0, test_index=0):

    print("\nStarting Grad-CAM...")

    # Load dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    class_names = dataset.classes

    # Recreate the same train-test split
    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    # Divide test data among clients
    total_size = len(test_dataset)

    base_size = total_size // NUM_CLIENTS
    remainder = total_size % NUM_CLIENTS

    client_size = base_size + (
        1 if client_id < remainder else 0
    )

    start = client_id * base_size + min(client_id, remainder)

    if test_index < 0 or test_index >= client_size:
        raise IndexError(
            f"test_index must be between 0 and {client_size - 1}"
        )

    client_test_indices = list(
        range(start, start + client_size)
    )

    client_test_dataset = Subset(
        test_dataset,
        client_test_indices
    )

    # Get selected image
    image, true_label = client_test_dataset[test_index]

    image = image.unsqueeze(0).to(DEVICE)

    # Load personalized FedBN model
    model = load_client_model(
        client_id,
        len(class_names)
    )

    # Make prediction
    # Get prediction and class probabilities
    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        prediction = outputs.argmax(dim=1).item()

        true_probability = probabilities[0][true_label].item()
        predicted_probability = probabilities[0][prediction].item()

        # Get top 5 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities[0], k=5
        )

    print("\n========== PREDICTION ANALYSIS ==========")

    print("True Class:", dataset.classes[true_label])
    print("Predicted Class:", dataset.classes[prediction])

    print(f"True Class Probability: {true_probability * 100:.2f}%")
    print(f"Predicted Class Probability: {predicted_probability * 100:.2f}%")

    print("\nTop 5 Predictions:")

    for probability, class_index in zip(top_probabilities, top_indices):
        print(
            f"{dataset.classes[class_index.item()]}: "
            f"{probability.item() * 100:.2f}%"
        )

    print("=========================================\n")

    # Initialize Grad-CAM
    target_layer = model.model.layer3[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    # Generate heatmap for predicted class
    # Generate Grad-CAM for predicted class
    predicted_heatmap = gradcam.generate(
        image,
        target_class=prediction
    )

    # Generate Grad-CAM for true class
    true_heatmap = gradcam.generate(
        image,
        target_class=true_label
    )

    # Convert normalized image back to display format
    original_image = image.squeeze(0).detach().cpu()

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    original_image = original_image * std + mean

    original_image = original_image.permute(
        1, 2, 0
    ).numpy()

    original_image = np.clip(
        original_image,
        0,
        1
    )

    # Save visualization
    # Save comparison visualization

    output_dir = RESULTS_DIR / "gradcam"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    # Original image
    axes[0].imshow(original_image)

    axes[0].set_title(
        f"Original Image\nTrue: {class_names[true_label]}"
    )

    axes[0].axis("off")

    # Predicted class Grad-CAM
    axes[1].imshow(original_image)

    axes[1].imshow(
        predicted_heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[1].set_title(
        f"Predicted Class Grad-CAM\n{class_names[prediction]}"
    )

    axes[1].axis("off")

    # True class Grad-CAM
    axes[2].imshow(original_image)

    axes[2].imshow(
        true_heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        f"True Class Grad-CAM\n{class_names[true_label]}"
    )

    axes[2].axis("off")

    plt.tight_layout()

    output_path = output_dir / (
        f"client_{client_id}_image_{test_index}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nComparison Visualization Saved:", output_path)


if __name__ == "__main__":

    generate_gradcam(
        client_id=0,
        test_index=20
    )