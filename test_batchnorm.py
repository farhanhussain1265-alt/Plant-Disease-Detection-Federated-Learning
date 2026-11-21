import torch.nn as nn

from backend.models.cnn_model import PlantDiseaseModel


model = PlantDiseaseModel(num_classes=15)

print("\n========== BATCH NORMALIZATION INSPECTION ==========")

batchnorm_count = 0

for name, layer in model.named_modules():

    if isinstance(layer, nn.BatchNorm2d):

        batchnorm_count += 1

        print(f"\nLayer: {name}")
        print("Running Mean Shape:", layer.running_mean.shape)
        print("Running Variance Shape:", layer.running_var.shape)
        print("Number of Features:", layer.num_features)

print("\nTotal BatchNorm Layers:", batchnorm_count)

print("\n========== INSPECTION COMPLETED ==========")