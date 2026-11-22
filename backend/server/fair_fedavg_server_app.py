import csv
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from backend.config.config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NUM_CLIENTS,
    DEVICE
)
from backend.models.cnn_model import PlantDiseaseModel
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data
from backend.utils.metrics import evaluate_model


app = ServerApp()


# --------------------------------------------------
# Centralized evaluation on the common test dataset
# --------------------------------------------------

def global_evaluate(
    server_round: int,
    arrays: ArrayRecord
) -> MetricRecord:

    print(f"\nEvaluating global model - Round {server_round}")

    # Load the same test split used by the experiment
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    # Create model and load received parameters
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    model.load_state_dict(
        arrays.to_torch_state_dict()
    )

    model.to(DEVICE)

    # Evaluate global model
    test_loss, test_accuracy = evaluate_model(
        model,
        test_dataset
    )

    print("Round:", server_round)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Save round-specific checkpoint
    output_checkpoint_dir = (
        CHECKPOINT_DIR / "fair_comparison" / "fedavg"
    )
    output_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        output_checkpoint_dir /
        f"fedavg_round_{server_round}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    # Save evaluation metrics
    output_results_dir = (
        RESULTS_DIR / "fair_comparison" / "fedavg"
    )
    output_results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_file = output_results_dir / "metrics.csv"

    file_exists = metrics_file.exists()

    with open(
        metrics_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "round",
                "global_test_loss",
                "global_test_accuracy"
            ])

        writer.writerow([
            server_round,
            test_loss,
            test_accuracy
        ])

    return MetricRecord({
        "global_test_loss": float(test_loss),
        "global_test_accuracy": float(test_accuracy)
    })


# --------------------------------------------------
# Main ServerApp
# --------------------------------------------------

@app.main()
def main(grid: Grid, context: Context) -> None:

    print("\n===================================")
    print(" FAIR FEDAVG EXPERIMENT STARTED")
    print("===================================")

    # Read experiment configuration
    num_rounds = context.run_config["num-server-rounds"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    learning_rate = context.run_config["learning-rate"]

    print("Number of clients:", NUM_CLIENTS)
    print("Federated rounds:", num_rounds)
    print("Local epochs:", local_epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    # Load the common initial model checkpoint
    initial_checkpoint = (
        CHECKPOINT_DIR / "fair_initial_resnet18.pth"
    )

    if not initial_checkpoint.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {initial_checkpoint}"
        )

    initial_state = torch.load(
        initial_checkpoint,
        map_location="cpu",
        weights_only=True
    )

    initial_arrays = ArrayRecord(initial_state)

    print("\nShared initial checkpoint loaded successfully.")

    # Configure FedAvg
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_available_nodes=NUM_CLIENTS
    )

    # Start federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({
            "local-epochs": local_epochs,
            "batch-size": batch_size,
            "learning-rate": learning_rate
        }),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate
    )

    # Save final global model
    final_state = result.arrays.to_torch_state_dict()

    final_checkpoint_dir = (
        CHECKPOINT_DIR / "fair_comparison" / "fedavg"
    )
    final_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_checkpoint_path = (
        final_checkpoint_dir / "fedavg_final_model.pth"
    )

    torch.save(
        final_state,
        final_checkpoint_path
    )

    print("\n===================================")
    print(" FAIR FEDAVG EXPERIMENT COMPLETED")
    print("===================================")
    print("Final model:", final_checkpoint_path)
