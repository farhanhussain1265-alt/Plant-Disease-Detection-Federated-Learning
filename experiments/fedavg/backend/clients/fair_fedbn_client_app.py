import torch
import torch.nn as nn

from flwr.clientapp import ClientApp
from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict
)

from backend.config.config import CHECKPOINT_DIR
from backend.models.cnn_model import PlantDiseaseModel
from backend.clients.train import train_model
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


app = ClientApp()


def get_batchnorm_keys(model):
    """Return all BatchNorm parameter and buffer keys."""

    keys = set()

    for module_name, module in model.named_modules():

        if isinstance(module, nn.modules.batchnorm._BatchNorm):

            prefix = f"{module_name}." if module_name else ""

            for key in module.state_dict().keys():
                keys.add(prefix + key)

    return keys


@app.train()
def train(message: Message, context: Context):

    print("\n================================")
    print("FAIR FEDBN CLIENT TRAINING")
    print("================================")

    config = message.content["config"]

    # Identify client
    partition_id = int(context.node_config["partition-id"])

    print("Client ID:", partition_id)

    # Load dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    # Recreate identical training partitions
    client_datasets, _ = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    client_dataset = client_datasets[partition_id]

    print("Training Images:", len(client_dataset))

    # Create model architecture
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # Identify BatchNorm parameters
    batchnorm_keys = get_batchnorm_keys(model)

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

    # Start with identical initial model parameters
    model.load_state_dict(initial_state)

    # Receive shared parameters from server
    shared_parameters = (
        message.content["arrays"].to_torch_state_dict()
    )

    # Update only non-BatchNorm parameters
    model_state = model.state_dict()

    model_state.update(shared_parameters)

    # Restore previous local BatchNorm parameters
    if "local_bn" in context.state:

        local_bn_state = context.state["local_bn"].to_torch_state_dict()

        model_state.update(local_bn_state)

        print("Previous local BatchNorm state restored.")

    else:

        print("Initial local BatchNorm state loaded.")

    model.load_state_dict(model_state)

    # Set reproducible local data-shuffling seed
    seed = 42 + partition_id

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Local training
    loss, accuracy = train_model(
        model,
        client_dataset,
        local_epochs=config["local-epochs"],
        learning_rate=config["learning-rate"],
        batch_size=config["batch-size"]
    )

    # Separate shared and local parameters
    trained_state = model.state_dict()

    shared_state = {
        key: value
        for key, value in trained_state.items()
        if key not in batchnorm_keys
    }

    local_bn_state = {
        key: value.detach().cpu().clone()
        for key, value in trained_state.items()
        if key in batchnorm_keys
    }

    # Preserve local BatchNorm state between rounds
    context.state["local_bn"] = ArrayRecord(local_bn_state)

    # Save client-specific BatchNorm checkpoint
    checkpoint_dir = (
        CHECKPOINT_DIR / "fair_comparison" / "fedbn"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir /
        f"fedbn_client_{partition_id}_bn.pth"
    )

    torch.save(
        local_bn_state,
        checkpoint_path
    )

    # Send only shared parameters to server
    updated_parameters = ArrayRecord(
        shared_state
    )

    # Training metrics
    metrics = MetricRecord({
        "train_loss": float(loss),
        "train_accuracy": float(accuracy),
        "num-examples": len(client_dataset)
    })

    content = RecordDict({
        "arrays": updated_parameters,
        "metrics": metrics
    })

    print("Training Loss:", loss)
    print("Training Accuracy:", accuracy)
    print("Local BatchNorm checkpoint saved.")
    print("Shared parameters sent to server.")
    print("FedBN client training completed.")

    return Message(
        content=content,
        reply_to=message
    )


@app.evaluate()
def evaluate(message: Message, context: Context):

    print("\n================================")
    print("FAIR FEDBN CLIENT EVALUATION")
    print("================================")

    partition_id = int(context.node_config["partition-id"])

    print("Client ID:", partition_id)

    # Load dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    # Recreate the common test split
    _, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    # Divide test data among clients
    total_size = len(test_dataset)
    base_size = total_size // 3
    remainder = total_size % 3

    start = partition_id * base_size + min(
        partition_id, remainder
    )

    client_size = base_size + (
        1 if partition_id < remainder else 0
    )

    end = start + client_size

    from torch.utils.data import Subset

    client_test_dataset = Subset(
        test_dataset,
        range(start, end)
    )

    print("Evaluation Images:", len(client_test_dataset))

    # Create model
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # Load common initial checkpoint
    initial_checkpoint = (
        CHECKPOINT_DIR / "fair_initial_resnet18.pth"
    )

    initial_state = torch.load(
        initial_checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(initial_state)

    # Load shared parameters from server
    shared_parameters = (
        message.content["arrays"].to_torch_state_dict()
    )

    model_state = model.state_dict()
    model_state.update(shared_parameters)

    # Restore personalized BatchNorm parameters
    if "local_bn" not in context.state:
        raise RuntimeError(
            f"Client {partition_id} has no local BatchNorm state."
        )

    model_state.update(context.state["local_bn"].to_torch_state_dict())

    model.load_state_dict(model_state)

    # Evaluate personalized client model
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    from backend.utils.metrics import evaluate_model

    test_loss, test_accuracy = evaluate_model(
        model,
        client_test_dataset
    )

    print("Client Test Loss:", test_loss)
    print("Client Test Accuracy:", test_accuracy)

    metrics = MetricRecord({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "num-examples": len(client_test_dataset)
    })

    return Message(
        content=RecordDict({
            "metrics": metrics
        }),
        reply_to=message
    )
