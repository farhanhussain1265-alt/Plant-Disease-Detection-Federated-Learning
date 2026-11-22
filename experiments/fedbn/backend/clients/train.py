import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from backend.config.config import (
    DEVICE,
    BATCH_SIZE,
    LOCAL_EPOCHS,
    LEARNING_RATE
)


def train_model(
    model,
    train_dataset,
    local_epochs=None,
    learning_rate=None,
    batch_size=None
):

    # Use default configuration if values are not provided
    local_epochs = local_epochs or LOCAL_EPOCHS
    learning_rate = learning_rate or LEARNING_RATE
    batch_size = batch_size or BATCH_SIZE

    # Load local training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.to(DEVICE)
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(local_epochs):

        epoch_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        total_loss = epoch_loss / len(train_dataset)

        print(
            f"Epoch [{epoch + 1}/{local_epochs}] "
            f"Loss: {total_loss:.4f}"
        )

    accuracy = correct / total if total > 0 else 0

    print("\nLocal Training Completed!")
    print("Training Loss:", round(total_loss, 4))
    print("Training Accuracy:", round(accuracy * 100, 2), "%")

    return total_loss, accuracy