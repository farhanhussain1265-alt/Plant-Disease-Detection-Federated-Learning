from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data

from backend.models.cnn_model import PlantDiseaseModel
from backend.clients.train import train_model

from backend.config.config import DEVICE


# 1. Load dataset with preprocessing
dataset = load_dataset(
    transform=get_train_transforms()
)

# 2. Distribute data among clients
client_datasets, test_dataset = distribute_data(dataset)

# 3. Select Client 1
client_dataset = client_datasets[0]

print("\nSelected Client: 1")
print("Client Training Images:", len(client_dataset))

# 4. Initialize ResNet18 model
model = PlantDiseaseModel(num_classes=len(dataset.classes))

# 5. Train the model locally
print("\nStarting Local Training...")

loss, accuracy = train_model(
    model,
    client_dataset
)

print("\nTraining Test Successful!")
print("Final Loss:", loss)
print("Final Accuracy:", accuracy)
print("Training Device:", DEVICE)