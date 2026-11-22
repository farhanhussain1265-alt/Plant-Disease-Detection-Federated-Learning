import csv
import torch
import torch.nn as nn

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from backend.config.config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    NUM_CLIENTS
)

from backend.models.cnn_model import PlantDiseaseModel


app = ServerApp()


def get_batchnorm_keys(model):
    """Identify all BatchNorm parameters and buffers."""

    keys = set()

    for module_name, module in model.named_modules():

        if isinstance(module, nn.modules.batchnorm._BatchNorm):

            prefix = f"{module_name}." if module_name else ""

            for key in module.state_dict().keys():
                keys.add(prefix + key)

    return keys


@app.main()
def main(grid: Grid, context: Context) -> None:

    print("\n===================================")
    print(" FAIR FEDBN EXPERIMENT STARTED")
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

    # Load the common initial checkpoint
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

    # Identify BatchNorm tensors
    model = PlantDiseaseModel(num_classes=15)

    batchnorm_keys = get_batchnorm_keys(model)

    # Keep only shared parameters
    shared_initial_state = {
        key: value
        for key, value in initial_state.items()
        if key not in batchnorm_keys
    }

    print("\nInitial checkpoint loaded.")
    print("Total initial tensors:", len(initial_state))
    print("Local BatchNorm tensors:", len(batchnorm_keys))
    print("Shared tensors:", len(shared_initial_state))

    initial_arrays = ArrayRecord(
        shared_initial_state
    )

    # Configure FedBN
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        weighted_by_key="num-examples"
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
        num_rounds=num_rounds
    )

    # Save final shared parameters
    output_checkpoint_dir = (
        CHECKPOINT_DIR / "fair_comparison" / "fedbn"
    )

    output_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_checkpoint = (
        output_checkpoint_dir / "fedbn_final_shared_parameters.pth"
    )

    torch.save(
        result.arrays.to_torch_state_dict(),
        final_checkpoint
    )

    # Save evaluation metrics
    output_results_dir = (
        RESULTS_DIR / "fair_comparison" / "fedbn"
    )

    output_results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_file = output_results_dir / "metrics.csv"

    with open(
        metrics_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "round",
            "test_loss",
            "test_accuracy"
        ])

        for round_number, metrics in sorted(
            result.evaluate_metrics_clientapp.items()
        ):

            writer.writerow([
                round_number,
                metrics.get("test_loss"),
                metrics.get("test_accuracy")
            ])

    print("\n===================================")
    print(" FAIR FEDBN EXPERIMENT COMPLETED")
    print("===================================")

    print("Final shared model:", final_checkpoint)
    print("Evaluation metrics:", metrics_file)

    print("\nFederated evaluation results:")

    for round_number, metrics in sorted(
        result.evaluate_metrics_clientapp.items()
    ):

        print(
            f"Round {round_number}: "
            f"Loss={metrics.get('test_loss')}, "
            f"Accuracy={metrics.get('test_accuracy')}"
        )
