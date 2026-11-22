import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from backend.config.config import DEVICE, RESULTS_DIR, NUM_CLIENTS
from backend.models.cnn_model import PlantDiseaseModel
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


# Fair experiment checkpoint directory
FAIR_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "fair_comparison"
)


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.activations = None
        self.gradients = None

        self.target_layer = target_layer

        self.hook = target_layer.register_forward_hook(
            self.save_activations
        )

    def save_activations(self, module, inputs, output):

        self.activations = output

        output.register_hook(self.save_gradients)

    def save_gradients(self, gradient):

        self.gradients = gradient

    def generate(self, image, target_class):

        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            output = self.model(image)

            score = output[:, target_class].sum()

            score.backward()

        weights = torch.mean(
            self.gradients,
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * self.activations,
            dim=1
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - np.min(cam)

        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam

    def close(self):

        self.hook.remove()


def load_model(method, num_classes, client_id=0):

    model = PlantDiseaseModel(num_classes=num_classes)

    if method == "fedavg":

        checkpoint_path = (
            FAIR_CHECKPOINT_DIR
            / "fedavg"
            / "fedavg_final_model.pth"
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(state_dict)

    elif method == "fedbn":

        if client_id < 0 or client_id >= NUM_CLIENTS:
            raise ValueError("Invalid FedBN client ID.")

        shared_path = (
            FAIR_CHECKPOINT_DIR
            / "fedbn"
            / "fedbn_final_shared_parameters.pth"
        )

        bn_path = (
            FAIR_CHECKPOINT_DIR
            / "fedbn"
            / f"fedbn_client_{client_id}_bn.pth"
        )

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

    else:

        raise ValueError(
            "Method must be either 'fedavg' or 'fedbn'."
        )

    model.to(DEVICE)
    model.eval()

    print(f"{method.upper()} model loaded successfully!")

    return model


def generate_gradcam(method="fedavg", client_id=0, test_index=20):

    print("\nStarting Grad-CAM...")

    dataset = load_dataset(
        transform=get_train_transforms()
    )

    class_names = dataset.classes

    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    total_size = len(test_dataset)

    base_size = total_size // NUM_CLIENTS
    remainder = total_size % NUM_CLIENTS

    if client_id < 0 or client_id >= NUM_CLIENTS:
        raise ValueError("Invalid client ID.")

    client_size = base_size + (
        1 if client_id < remainder else 0
    )

    start = client_id * base_size + min(client_id, remainder)

    if test_index < 0 or test_index >= client_size:
        raise IndexError(
            f"test_index must be between 0 and {client_size - 1}"
        )

    selected_index = start + test_index

    image, true_label = test_dataset[selected_index]

    image = image.unsqueeze(0).to(DEVICE)

    model = load_model(
        method,
        len(class_names),
        client_id
    )

    # Prediction
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        prediction = outputs.argmax(dim=1).item()

        predicted_probability = probabilities[0][prediction].item()

        true_probability = probabilities[0][true_label].item()

        top_probabilities, top_indices = torch.topk(
            probabilities[0],
            k=5
        )

    print("\n========== PREDICTION ANALYSIS ==========")

    print("Method:", method.upper())
    print("Client ID:", client_id)
    print("True Class:", class_names[true_label])
    print("Predicted Class:", class_names[prediction])

    print(
        f"Predicted Confidence: "
        f"{predicted_probability * 100:.2f}%"
    )

    print(
        f"True Class Probability: "
        f"{true_probability * 100:.2f}%"
    )

    print("\nTop 5 Predictions:")

    for probability, class_index in zip(
        top_probabilities,
        top_indices
    ):

        print(
            f"{class_names[class_index.item()]}: "
            f"{probability.item() * 100:.2f}%"
        )

    print("=========================================")

    # Grad-CAM
    target_layer = model.model.layer4[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    predicted_heatmap = gradcam.generate(
        image,
        prediction
    )

    true_heatmap = gradcam.generate(
        image,
        true_label
    )

    gradcam.close()

    # Restore original image
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
    output_dir = RESULTS_DIR / "fair_comparison" / "gradcam"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    axes[0].imshow(original_image)

    axes[0].set_title(
        f"Original Image\nTrue: {class_names[true_label]}"
    )

    axes[0].axis("off")

    axes[1].imshow(original_image)

    axes[1].imshow(
        predicted_heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[1].set_title(
        f"{method.upper()} Predicted Grad-CAM\n"
        f"{class_names[prediction]}"
    )

    axes[1].axis("off")
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
        f"{method}_client_{client_id}_image_{test_index}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nGrad-CAM visualization saved:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["fedavg", "fedbn"],
        default="fedavg"
    )

    parser.add_argument(
        "--client",
        type=int,
        default=0
    )

    parser.add_argument(
        "--index",
        type=int,
        default=20
    )

    args = parser.parse_args()

    generate_gradcam(
        method=args.method,
        client_id=args.client,
        test_index=args.index
    )