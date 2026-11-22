import torch

from flwr.clientapp import ClientApp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict

from backend.models.cnn_model import PlantDiseaseModel
from backend.clients.train import train_model
from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data


app = ClientApp()


@app.train()
def train(message: Message, context: Context):

    print("\nFair FedAvg Client Training Started!")

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

    # Initialize model architecture
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # Receive global model parameters
    global_parameters = (
        message.content["arrays"].to_torch_state_dict()
    )

    # Load global parameters into local model
    model.load_state_dict(global_parameters)

    # Set reproducible local data-shuffling seed
    seed = 42 + partition_id

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Train local model
    loss, accuracy = train_model(
        model,
        client_dataset,
        local_epochs=config["local-epochs"],
        learning_rate=config["learning-rate"],
        batch_size=config["batch-size"]
    )

    # Prepare updated model parameters
    updated_parameters = ArrayRecord(
        model.state_dict()
    )

    # Prepare training metrics
    metrics = MetricRecord({
        "train_loss": float(loss),
        "train_accuracy": float(accuracy),
        "num-examples": len(client_dataset)
    })

    content = RecordDict({
        "arrays": updated_parameters,
        "metrics": metrics
    })

    print("FedAvg Client Training Completed!")
    print("Updated parameters sent to server.")

    return Message(
        content=content,
        reply_to=message
    )
