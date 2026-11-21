import torch

from torch.utils.data import random_split, Subset

from backend.config.config import NUM_CLIENTS


def distribute_data(dataset, test_ratio=0.2, seed=42):
    print("Data distribution started...")
    # Step 1: Calculate dataset sizes
    total_size = len(dataset)

    test_size = round(total_size * test_ratio)
    train_size = total_size - test_size

    # Step 2: Split dataset into training and testing
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Step 3: Shuffle training indices
    indices = torch.randperm(
        len(train_dataset),
        generator=torch.Generator().manual_seed(seed)
    ).tolist()

    # Step 4: Distribute training data among clients
    client_datasets = []
    start = 0

    for client_id in range(NUM_CLIENTS):

        client_size = train_size // NUM_CLIENTS

        if client_id < train_size % NUM_CLIENTS:
            client_size += 1

        client_indices = indices[start:start + client_size]

        client_dataset = Subset(
            train_dataset,
            client_indices
        )

        client_datasets.append(client_dataset)

        start += client_size

    # Step 5: Display distribution information
    print("\nData Distribution Completed!")

    print("Total Images:", total_size)
    print("Training Images:", len(train_dataset))
    print("Testing Images:", len(test_dataset))

    for client_id, client_dataset in enumerate(client_datasets):
        print(f"Client {client_id + 1} Images:", len(client_dataset))

    print("Data distribution done!")
    return client_datasets, test_dataset