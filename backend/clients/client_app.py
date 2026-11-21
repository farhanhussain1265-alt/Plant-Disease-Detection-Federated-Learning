import torch.nn as nn
from torch.utils.data import Subset
import torch

from backend.config.config import CHECKPOINT_DIR
from backend.config.config import NUM_CLIENTS
from backend.utils.metrics import evaluate_model

from flwr.clientapp import ClientApp
from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict
)

from backend.models.cnn_model import PlantDiseaseModel
from backend.clients.train import train_model

from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


app = ClientApp()


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


@app.train()
def train(message: Message, context: Context):

    print("\nFedBN Client Training Started!")

    config = message.content["config"]

    # Load local dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    client_datasets, _ = distribute_data(dataset)

    partition_id = int(context.node_config["partition-id"])

    client_dataset = client_datasets[partition_id]

    print("Client ID:", partition_id)
    print("Training Images:", len(client_dataset))

    # Initialize model
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # Identify BatchNorm parameters
    bn_keys = get_batchnorm_keys(model)

    # Receive shared global parameters
    received_parameters = (
        message.content["arrays"].to_torch_state_dict()
    )

    # Load shared parameters without replacing local BatchNorm
    model_state = model.state_dict()

    for key, value in received_parameters.items():

        if key not in bn_keys:
            model_state[key] = value

    # Restore this client's BatchNorm state
    if "local_bn" in context.state:

        local_bn_state = (
            context.state["local_bn"].to_torch_state_dict()
        )

        model_state.update(local_bn_state)

        print("Previous local BatchNorm state restored!")

    else:

        print("Initializing local BatchNorm state...")

    model.load_state_dict(model_state)

    # Train locally
    loss, accuracy = train_model(
        model,
        client_dataset,
        local_epochs=config["local-epochs"],
        learning_rate=config["learning-rate"],
        batch_size=config["batch-size"]
    )

    # Separate shared parameters and local BatchNorm parameters
    updated_state = model.state_dict()

    shared_state = {
        key: value
        for key, value in updated_state.items()
        if key not in bn_keys
    }

    local_bn_state = {
        key: value
        for key, value in updated_state.items()
        if key in bn_keys
    }

    # Save local BatchNorm state for the next round
    # Save local BatchNorm state for the next round
    context.state["local_bn"] = ArrayRecord(
        local_bn_state
    )

    # Persist local BatchNorm parameters on disk
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    bn_checkpoint_path = (
        CHECKPOINT_DIR / f"fedbn_client_{partition_id}_bn.pth"
    )

    torch.save(
        local_bn_state,
        bn_checkpoint_path
    )

    print("Local BatchNorm checkpoint saved:", bn_checkpoint_path)

    # Send only shared parameters to the server
    updated_parameters = ArrayRecord(
        shared_state
    )

    metrics = MetricRecord({
        "train_loss": float(loss),
        "train_accuracy": float(accuracy),
        "num-examples": len(client_dataset)
    })

    content = RecordDict({
        "arrays": updated_parameters,
        "metrics": metrics
    })

    print("Local BatchNorm state saved!")
    print("Shared parameters sent to server!")

    return Message(
        content=content,
        reply_to=message
    )

@app.evaluate()
def evaluate(message: Message, context: Context):

    print("\nFedBN Client Evaluation Started!")

    # Load dataset and recreate the test split
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    # Identify this client
    partition_id = int(context.node_config["partition-id"])

    # Divide test data among clients
    total_size = len(test_dataset)
    base_size = total_size // NUM_CLIENTS
    remainder = total_size % NUM_CLIENTS

    start = partition_id * base_size + min(
        partition_id, remainder
    )

    client_size = base_size + (
        1 if partition_id < remainder else 0
    )

    test_indices = list(
        range(start, start + client_size)
    )

    client_test_dataset = Subset(
        test_dataset,
        test_indices
    )

    # Initialize model
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # Load shared global parameters
    received_parameters = (
        message.content["arrays"].to_torch_state_dict()
    )

    model_state = model.state_dict()

    model_state.update(received_parameters)

    # Restore local BatchNorm state
    if "local_bn" in context.state:

        local_bn_state = (
            context.state["local_bn"].to_torch_state_dict()
        )

        model_state.update(local_bn_state)

    else:
        print("Warning: Local BatchNorm state not found!")

    model.load_state_dict(model_state)

    # Evaluate using this client's local BatchNorm
    loss, accuracy = evaluate_model(
        model,
        client_test_dataset
    )

    metrics = MetricRecord({
        "eval_loss": float(loss),
        "eval_accuracy": float(accuracy),
        "num-examples": len(client_test_dataset)
    })

    print("Client Evaluation Completed!")

    return Message(
        content=RecordDict({
            "metrics": metrics
        }),
        reply_to=message
    )