import torch
import torch.nn as nn

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context
)

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg

from backend.config.config import CHECKPOINT_DIR
from backend.models.cnn_model import PlantDiseaseModel


app = ServerApp()


def get_batchnorm_keys(model):

    batchnorm_layers = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]

    bn_keys = {
        key
        for key in model.state_dict()
        if any(
            key.startswith(f"{name}.")
            for name in batchnorm_layers
        )
    }

    return bn_keys


@app.main()
def main(grid: Grid, context: Context):

    print("\nStarting Flower FedBN Server...")

    num_rounds = context.run_config["num-server-rounds"]

    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]

    # Initialize global model
    global_model = PlantDiseaseModel(num_classes=15)

    # Identify BatchNorm parameters
    bn_keys = get_batchnorm_keys(global_model)

    # Keep only shared parameters
    global_state = global_model.state_dict()

    shared_state = {
        key: value
        for key, value in global_state.items()
        if key not in bn_keys
    }

    initial_arrays = ArrayRecord(shared_state)

    print("BatchNorm parameters excluded from aggregation!")
    print("Shared parameters:", len(shared_state))
    print("Local BatchNorm parameters:", len(bn_keys))

    # Configure FedAvg for shared parameters
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0
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

        num_rounds=num_rounds
    )

    # Save shared global parameters separately
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = CHECKPOINT_DIR / "fedbn_shared_parameters.pth"

    torch.save(
        result.arrays.to_torch_state_dict(),
        model_path
    )

    print("\nFedBN Training Completed!")
    print("Shared Model Parameters Saved:", model_path)
    print("Client-specific BatchNorm states remain local.")