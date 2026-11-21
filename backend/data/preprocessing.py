from torchvision import transforms
from backend.config.config import IMAGE_SIZE


def get_train_transforms():
    print("Transforming....")
    transform = transforms.Compose([

        # Resize image
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Convert image into PyTorch tensor
        transforms.ToTensor(),

        # Normalize image
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])
    print("Transforming Done!")
    return transform