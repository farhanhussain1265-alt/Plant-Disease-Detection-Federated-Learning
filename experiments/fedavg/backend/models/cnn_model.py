import torch
import torch.nn as nn

from torchvision import models


class PlantDiseaseModel(nn.Module):

    def __init__(self, num_classes):

        super(PlantDiseaseModel, self).__init__()

        # Load ResNet18 architecture
        self.model = models.resnet18(weights=None)

        # Get the number of input features
        input_features = self.model.fc.in_features

        # Replace the final classification layer
        self.model.fc = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)