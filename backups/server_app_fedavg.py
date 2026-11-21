import torch
import csv
from backend.config.config import RESULTS_DIR

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord
)

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg

from backend.config.config import CHECKPOINT_DIR


from backend.models.cnn_model import PlantDiseaseModel

from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data

from backend.utils.metrics import evaluate_model


# Create Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    print("\nStarting Flower Server...")

    # Read configuration
    num_rounds = context.run_config["num-server-rounds"]

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]

    # Initialize a fresh metrics CSV for this experiment
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_file = RESULTS_DIR / "federated_metrics.csv"

    with open(metrics_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "round",
            "global_test_loss",
            "global_test_accuracy"
        ])

    print("New experiment metrics file initialized!")

    # Initialize global model
    global_model = PlantDiseaseModel(
        num_classes=15
    )

    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )

    # Configure FedAvg strategy
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0
    )

    # Start federated training
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,

        train_config=ConfigRecord({
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
            "batch-size": batch_size
        }),

        num_rounds=num_rounds,
        evaluate_fn=global_evaluate
    )

    # Save final global model
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = CHECKPOINT_DIR / "federated_resnet18.pth"

    torch.save(
        result.arrays.to_torch_state_dict(),
        model_path
    )

    print("\nFederated Training Completed!")
    print("Final Model Saved:", model_path)


def global_evaluate(server_round: int, arrays: ArrayRecord):

    print(f"\nGlobal Evaluation - Round {server_round}")

    # Load global model parameters
    model = PlantDiseaseModel(
        num_classes=15
    )

    model.load_state_dict(
        arrays.to_torch_state_dict()
    )

    # Load dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    # Get centralized test dataset
    _, test_dataset = distribute_data(dataset)

    # Evaluate global model
    loss, accuracy = evaluate_model(
        model,
        test_dataset
    )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_file = RESULTS_DIR / "federated_metrics.csv"

    file_exists = metrics_file.exists()

    with open(metrics_file, "a", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "round",
                "global_test_loss",
                "global_test_accuracy"
            ])

        writer.writerow([
            server_round,
            loss,
            accuracy
        ])

    print("Global metrics saved:", metrics_file)


    return MetricRecord({
        "global_test_loss": float(loss),
        "global_test_accuracy": float(accuracy)
    })