from collections import Counter

from backend.data.dataset_loader import load_dataset
from backend.data.data_distribution import distribute_data


def check_client_distribution():

    # Load dataset without image transformations
    dataset = load_dataset()

    # Distribute dataset among 3 clients
    client_datasets, test_dataset = distribute_data(
        dataset,
        test_ratio=0.2,
        seed=42
    )

    print("\n========== CLIENT CLASS DISTRIBUTION ==========")

    for client_id, client_dataset in enumerate(client_datasets):

        # Recover original dataset indices
        original_indices = [
            client_dataset.dataset.indices[index]
            for index in client_dataset.indices
        ]

        # Get class labels using original indices
        labels = [
            dataset.targets[index]
            for index in original_indices
        ]

        class_counts = Counter(labels)

        print(f"\nCLIENT {client_id + 1}")
        print("-" * 45)

        for class_id, class_name in enumerate(dataset.classes):

            count = class_counts.get(class_id, 0)

            print(f"{class_name}: {count}")

        print("Total images:", len(labels))

    print("\n========== DISTRIBUTION CHECK COMPLETED ==========")


if __name__ == "__main__":
    check_client_distribution()