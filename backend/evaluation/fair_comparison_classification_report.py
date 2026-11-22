import matplotlib
matplotlib.use("Agg")

import torch
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
    BATCH_SIZE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NUM_CLIENTS
)

from backend.models.cnn_model import PlantDiseaseModel
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


# --------------------------------------------------
# PATHS
# --------------------------------------------------

FAIR_CHECKPOINT_DIR = CHECKPOINT_DIR / "fair_comparison"

OUTPUT_DIR = RESULTS_DIR / "fair_comparison" / "classification_reports"


# --------------------------------------------------
# LOAD FEDAVG MODEL
# --------------------------------------------------

def load_fedavg_model(num_classes):

    print("\nLoading Fair FedAvg Model...")

    model = PlantDiseaseModel(
        num_classes=num_classes
    )

    model_path = (
        FAIR_CHECKPOINT_DIR
        / "fedavg"
        / "fedavg_final_model.pth"
    )

    state = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    print("Fair FedAvg Model Loaded Successfully!")

    return model


# --------------------------------------------------
# LOAD PERSONALIZED FEDBN MODEL
# --------------------------------------------------

def load_fedbn_model(client_id, num_classes):

    print(f"\nLoading FedBN Client {client_id} Model...")

    model = PlantDiseaseModel(
        num_classes=num_classes
    )

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
        map_location="cpu"
    )

    bn_state = torch.load(
        bn_path,
        map_location="cpu"
    )

    model_state = model.state_dict()

    model_state.update(shared_state)
    model_state.update(bn_state)

    model.load_state_dict(model_state)

    model.to(DEVICE)
    model.eval()

    print(f"FedBN Client {client_id} Model Loaded Successfully!")

    return model


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, test_dataset):

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

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

            actual_labels.extend(
                labels.numpy()
            )

            predicted_labels.extend(
                predictions.cpu().numpy()
            )

    return actual_labels, predicted_labels


# --------------------------------------------------
# GENERATE REPORT AND CONFUSION MATRIX
# --------------------------------------------------

def save_evaluation_results(
    method_name,
    actual_labels,
    predicted_labels,
    class_names
):

    method_dir = OUTPUT_DIR / method_name

    method_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels = list(range(len(class_names)))

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    print(f"\n========== {method_name.upper()} REPORT ==========")

    print(report)

    print(
        f"Overall Accuracy: {accuracy * 100:.2f}%"
    )

    # Save classification report
    report_path = method_dir / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write(f"{method_name.upper()} CLASSIFICATION REPORT\n\n")
        file.write(report)
        file.write(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Total Test Images: {len(actual_labels)}\n")

    # Generate confusion matrix
    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(18, 16))

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format="d",
        colorbar=False
    )

    plt.title(f"{method_name.upper()} - Confusion Matrix")
    plt.tight_layout()

    matrix_path = method_dir / "confusion_matrix.png"

    plt.savefig(
        matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Classification report saved:", report_path)
    print("Confusion matrix saved:", matrix_path)

    return accuracy


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("\nStarting Fair Model Evaluation...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    class_names = dataset.classes

    print("\nNumber of Classes:", len(class_names))

    # Recreate deterministic test split
    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    print("Total Test Images:", len(test_dataset))

    # ----------------------------------------------
    # FEDAVG EVALUATION
    # ----------------------------------------------

    fedavg_model = load_fedavg_model(
        len(class_names)
    )

    fedavg_labels, fedavg_predictions = evaluate_model(
        fedavg_model,
        test_dataset
    )

    fedavg_accuracy = save_evaluation_results(
        "fedavg",
        fedavg_labels,
        fedavg_predictions,
        class_names
    )

    # ----------------------------------------------
    # FEDBN EVALUATION
    # ----------------------------------------------

    fedbn_labels = []
    fedbn_predictions = []

    total_size = len(test_dataset)

    base_size = total_size // NUM_CLIENTS
    remainder = total_size % NUM_CLIENTS

    start = 0

    for client_id in range(NUM_CLIENTS):

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

        print(f"\nEvaluating FedBN Client {client_id}...")

        model = load_fedbn_model(
            client_id,
            len(class_names)
        )

        labels, predictions = evaluate_model(
            model,
            client_test_dataset
        )

        fedbn_labels.extend(labels)
        fedbn_predictions.extend(predictions)

        client_accuracy = accuracy_score(
            labels,
            predictions
        )

        print(
            f"Client {client_id} Accuracy: "
            f"{client_accuracy * 100:.2f}%"
        )

        start += client_size

    fedbn_accuracy = save_evaluation_results(
        "fedbn",
        fedbn_labels,
        fedbn_predictions,
        class_names
    )

    # ----------------------------------------------
    # FINAL COMPARISON
    # ----------------------------------------------

    print("\n========== FINAL COMPARISON ==========")

    print(
        f"FedAvg Accuracy: {fedavg_accuracy * 100:.2f}%"
    )

    print(
        f"FedBN Accuracy: {fedbn_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy Difference: "
        f"{(fedavg_accuracy - fedbn_accuracy) * 100:.2f} "
        f"percentage points"
    )

    print("\nFair Classification Evaluation Completed!")


if __name__ == "__main__":

    main()