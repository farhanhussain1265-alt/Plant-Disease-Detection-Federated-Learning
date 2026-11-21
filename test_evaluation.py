from backend.data.dataset_loader import load_dataset
from backend.data.preprocessing import get_train_transforms
from backend.data.data_distribution import distribute_data

from backend.models.cnn_model import PlantDiseaseModel
from backend.clients.train import train_model
from backend.utils.metrics import evaluate_model


# 1. Load dataset
dataset = load_dataset(
    transform=get_train_transforms()
)

# 2. Distribute dataset
client_datasets, test_dataset = distribute_data(dataset)

# 3. Initialize model
model = PlantDiseaseModel(
    num_classes=len(dataset.classes)
)

# 4. Train model using Client 1 data
print("\nStarting Client 1 Training...")

train_loss, train_accuracy = train_model(
    model,
    client_datasets[0]
)

# 5. Evaluate on global test dataset
print("\nStarting Global Test Evaluation...")

test_loss, test_accuracy = evaluate_model(
    model,
    test_dataset
)

print("\nEvaluation Test Successful!")