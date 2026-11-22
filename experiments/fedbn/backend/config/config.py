from pathlib import Path
import torch
import os


# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]


# Dataset directories
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
TEST_DATA_DIR = DATASET_DIR / "test"

# Actual PlantVillage dataset location
PLANTVILLAGE_DIR = Path(
    os.getenv(
        "PLANTVILLAGE_DATA_DIR",
        r"C:\Users\farha\Desktop\MAJOR PROJECT\dataset\raw\PlantVillage"
    )
)

# Model and result directories
CHECKPOINT_DIR = Path(
    os.getenv(
        "PROJECT_CHECKPOINT_DIR",
        r"C:\Users\farha\Desktop\MAJOR PROJECT\checkpoints"
    )
)

RESULTS_DIR = Path(
    os.getenv(
        "PROJECT_RESULTS_DIR",
        r"C:\Users\farha\Desktop\MAJOR PROJECT\results"
    )
)

# Federated Learning configuration
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 16


# Image configuration
IMAGE_SIZE = 128
NUM_CHANNELS = 3


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model configuration
MODEL_NAME = "ResNet18"
LEARNING_RATE = 0.001


print("Configuration loaded successfully!")
print("Project Root:", ROOT_DIR)
print("Training Device:", DEVICE)