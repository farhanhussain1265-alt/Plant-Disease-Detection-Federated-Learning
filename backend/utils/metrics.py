import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from backend.config.config import DEVICE, BATCH_SIZE


def evaluate_model(model, test_dataset):

    # Load test data in batches
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Generate predictions
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            # Get predicted class
            predictions = torch.argmax(outputs, dim=1)

            # Calculate correct predictions
            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    # Calculate final metrics
    test_loss = total_loss / total
    test_accuracy = correct / total

    print("\nModel Evaluation Completed!")
    print("Test Loss:", round(test_loss, 4))
    print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
    print("Correct Predictions:", correct)
    print("Total Predictions:", total)

    return test_loss, test_accuracy