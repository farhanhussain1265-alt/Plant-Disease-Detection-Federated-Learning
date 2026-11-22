from torchvision import datasets
from backend.config.config import PLANTVILLAGE_DIR


def load_dataset(transform=None):

    dataset = datasets.ImageFolder(
        root=str(PLANTVILLAGE_DIR),
        transform=transform
    )

    print("Dataset loading....")
    print("Dataset Path:", PLANTVILLAGE_DIR)
    print("Total Images:", len(dataset))
    print("Total Classes:", len(dataset.classes))
    print("Dataset loaded successfully!")

    print("\nClass Names:")
    for class_name in dataset.classes:
        print(class_name)

    return dataset


if __name__ == "__main__":
    dataset = load_dataset()