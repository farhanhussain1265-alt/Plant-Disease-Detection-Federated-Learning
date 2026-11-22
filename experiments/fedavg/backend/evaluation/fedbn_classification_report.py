import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

from backend.config.config import (
    DEVICE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NUM_CLIENTS,
    BATCH_SIZE
)

from backend.models.cnn_model import PlantDiseaseModel
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


def load_client_model(client_id, num_classes):

    print(f"\nLoading Client {client_id} Model...")

    # Initialize ResNet18
    model = PlantDiseaseModel(
        num_classes=num_classes
    )

    # Load shared global parameters
    shared_path = (
        CHECKPOINT_DIR / "fedbn_shared_parameters.pth"
    )

    shared_state = torch.load(
        shared_path,
        map_location=DEVICE
    )

    # Load client-specific BatchNorm parameters
    bn_path = (
        CHECKPOINT_DIR / f"fedbn_client_{client_id}_bn.pth"
    )

    bn_state = torch.load(
        bn_path,
        map_location=DEVICE
    )

    # Combine shared parameters and local BatchNorm parameters
    model_state = model.state_dict()

    model_state.update(shared_state)
    model_state.update(bn_state)

    # Load complete model state
    model.load_state_dict(model_state)

    model.to(DEVICE)
    model.eval()

    print(f"Client {client_id} Model Loaded Successfully!")

    return model


def evaluate_client(model, test_dataset):

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    return all_labels, all_predictions


def generate_classification_report():

    print("\nStarting FedBN Classification Evaluation...")

    # Load complete dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    class_names = dataset.classes

    # Recreate the same deterministic test split
    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    all_true_labels = []
    all_predictions = []

    # Divide test dataset among clients
    total_size = len(test_dataset)

    base_size = total_size // NUM_CLIENTS
    remainder = total_size % NUM_CLIENTS

    for client_id in range(NUM_CLIENTS):

        start = (
            client_id * base_size
            + min(client_id, remainder)
        )

        client_size = base_size + (
            1 if client_id < remainder else 0
        )

        test_indices = list(
            range(start, start + client_size)
        )

        client_test_dataset = Subset(
            test_dataset,
            test_indices
        )

        print("\n------------------------------")
        print(f"Evaluating Client {client_id}")
        print("------------------------------")

        # Reconstruct personalized model
        model = load_client_model(
            client_id,
            len(class_names)
        )

        # Evaluate client model
        labels, predictions = evaluate_client(
            model,
            client_test_dataset
        )

        all_true_labels.extend(labels)
        all_predictions.extend(predictions)

        print(
            f"Client {client_id} Test Images:",
            len(labels)
        )

        print(
            f"Client {client_id} Accuracy:",
            round(
                accuracy_score(labels, predictions) * 100,
                2
            ),
            "%"
        )

    # Generate overall classification report
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = classification_report(
        all_true_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    overall_accuracy = accuracy_score(
        all_true_labels,
        all_predictions
    )

    print("\n========== FEDBN CLASSIFICATION REPORT ==========")

    print(report)

    print(
        "Overall Test Accuracy:",
        round(overall_accuracy * 100, 2),
        "%"
    )

    print(
        "Total Test Images:",
        len(all_true_labels)
    )

    # Save classification report
    report_path = (
        RESULTS_DIR / "fedbn_classification_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("FEDBN CLASSIFICATION REPORT\n\n")

        file.write(report)

        file.write(
            f"\nOverall Test Accuracy: "
            f"{overall_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Total Test Images: {len(all_true_labels)}\n"
        )

    print("\nClassification Report Saved:", report_path)

    # Generate confusion matrix
    cm = confusion_matrix(
        all_true_labels,
        all_predictions,
        labels=list(range(len(class_names)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(18, 16))

    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True
    )

    plt.title("FedBN - Plant Disease Confusion Matrix")

    plt.tight_layout()

    matrix_dir = RESULTS_DIR / "confusion_matrix"

    matrix_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix_path = (
        matrix_dir / "fedbn_confusion_matrix.png"
    )

    plt.savefig(
        matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion Matrix Saved:", matrix_path)

    print("\nFedBN Evaluation Completed Successfully!")


if __name__ == "__main__":

    generate_classification_report()