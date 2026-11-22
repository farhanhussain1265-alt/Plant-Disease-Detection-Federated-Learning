
import matplotlib
matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report as generate_classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from backend.config.config import (
    DEVICE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    RESULTS_DIR
)

from backend.models.cnn_model import PlantDiseaseModel
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


def evaluate_classification():

    print("\nStarting Classification Evaluation...")

    # 1. Load the dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    # 2. Recreate the same test split
    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 3. Initialize the model
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # 4. Load the trained federated model
    model_path = CHECKPOINT_DIR / "federated_resnet18.pth"

    if not model_path.exists():
        print("Federated model checkpoint not found!")
        return

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    print("\nFederated model loaded successfully!")

    # 5. Generate predictions
    actual_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            actual_labels.extend(labels.numpy())
            predicted_labels.extend(
                predictions.cpu().numpy()
            )

    # 6. Generate classification report
    report = generate_classification_report(
        actual_labels,
        predicted_labels,
        labels=list(range(len(dataset.classes))),
        target_names=dataset.classes,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = RESULTS_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    print("Classification report saved:", report_path)

    # 7. Generate confusion matrix
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(range(len(dataset.classes)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=dataset.classes
    )

    fig, ax = plt.subplots(figsize=(18, 16))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format="d",
        colorbar=False
    )

    plt.title("Plant Disease Detection - Confusion Matrix")
    plt.tight_layout()

    confusion_dir = RESULTS_DIR / "confusion_matrix"

    confusion_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    confusion_path = confusion_dir / "confusion_matrix.png"

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion Matrix saved:", confusion_path)

    print("\nClassification Evaluation Completed!")


if __name__ == "__main__":
    evaluate_classification()