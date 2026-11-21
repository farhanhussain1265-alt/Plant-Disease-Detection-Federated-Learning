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


# Create Flower ClientApp
app = ClientApp()


@app.train()
def train(message: Message, context: Context):

    print("\nFlower Client Training Started!")

    # Read client configuration
    config = message.content["config"]

    # Load PlantVillage dataset
    dataset = load_dataset(
        transform=get_train_transforms()
    )

    # Distribute data among clients
    client_datasets, _ = distribute_data(dataset)

    # Identify the current client
    partition_id = int(context.node_config["partition-id"])

    client_dataset = client_datasets[partition_id]

    print("Client ID:", partition_id)
    print("Training Images:", len(client_dataset))

    # Initialize ResNet18 model
    model = PlantDiseaseModel(
        num_classes=len(dataset.classes)
    )

    # Receive global model parameters
    global_parameters = message.content["arrays"]

    model.load_state_dict(
        global_parameters.to_torch_state_dict()
    )

    # Local training
    loss, accuracy = train_model(
        model,
        client_dataset,
        local_epochs=config["local-epochs"],
        learning_rate=config["learning-rate"],
        batch_size=config["batch-size"]
    )

    # Convert updated model into Flower ArrayRecord
    updated_parameters = ArrayRecord(
        model.state_dict()
    )

    # Prepare training metrics
    metrics = MetricRecord({
        "train_loss": float(loss),
        "train_accuracy": float(accuracy),
        "num-examples": len(client_dataset)
    })

    # Prepare response
    content = RecordDict({
        "arrays": updated_parameters,
        "metrics": metrics
    })

    return Message(
        content=content,
        reply_to=message
    )