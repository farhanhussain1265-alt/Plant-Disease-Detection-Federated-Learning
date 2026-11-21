# 🌿 Plant Disease Detection Using Federated Learning

## 1. Project Overview

Plant Disease Detection Using Federated Learning is a deep learning-based application that identifies plant diseases from leaf images.

The project uses a ResNet18 Convolutional Neural Network (CNN) and Flower Federated Learning framework to simulate collaborative model training across three clients.

Instead of aggregating all model parameters, the project implements Federated Batch Normalization (FedBN), which keeps Batch Normalization parameters local to each client while aggregating the remaining model parameters.

A Streamlit web application allows users to upload plant leaf images, obtain disease predictions, and visualize model attention using Grad-CAM.

## 2. Objectives

- Develop an automated plant disease classification system.
- Implement federated learning using the Flower framework.
- Compare FedAvg and FedBN approaches.
- Preserve client-specific Batch Normalization parameters.
- Apply Grad-CAM for model explainability.
- Develop an interactive web application for disease prediction.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| CNN Architecture | ResNet18 |
| Federated Learning | Flower |
| Federated Strategy | FedAvg, FedBN |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Evaluation | Scikit-learn |
| Explainability | Grad-CAM |
| Frontend | Streamlit |

## 4. Dataset

**Dataset:** PlantVillage

- Total Images: 20,638
- Number of Classes: 15
- Training Images: 16,510
- Testing Images: 4,128
- Number of Simulated Clients: 3

The dataset contains healthy and diseased leaf images from pepper, potato, and tomato plants.

## 5. System Architecture

The project follows a simulated federated learning architecture.

1. The PlantVillage dataset is divided into training and testing sets.
2. Training data is distributed across three simulated clients.
3. Each client trains a local ResNet18 model.
4. Flower coordinates communication between clients and the server.
5. The server aggregates shared model parameters.
6. Batch Normalization parameters remain client-specific in FedBN.
7. The trained models are evaluated using the test dataset.
8. The final personalized models are used for disease prediction.

## 6. Federated Learning Approaches

### FedAvg

Federated Averaging aggregates model parameters from participating clients to produce a global model.

### FedBN

Federated Batch Normalization keeps Batch Normalization parameters local to individual clients while aggregating the remaining model parameters.

This allows clients to retain personalized normalization statistics.

## 7. Experimental Results

### FedAvg vs FedBN

| Evaluation Metric | FedAvg | FedBN |
|---|---:|---:|
| Test Accuracy | 6.66% | 81.81% |
| Macro Precision | 0.02 | 0.79 |
| Macro Recall | 0.08 | 0.79 |
| Macro F1 Score | 0.02 | 0.79 |
| Weighted F1 Score | 0.02 | 0.82 |

**Observed accuracy difference: 75.15 percentage points.**

### Important Experimental Limitation

The FedAvg baseline was trained for one communication round, whereas FedBN was trained for five communication rounds.

Therefore, these results represent the observed outcomes of two experiments with unequal training budgets. The improvement cannot be attributed solely to the FedBN method.

## 8. Explainable AI

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize image regions that contribute to a model's prediction.

The application generates heatmaps for predicted disease classes, helping users inspect the model's attention.

Grad-CAM provides qualitative explanations and does not prove that the model relies exclusively on biological disease symptoms.

## 9. Web Application Features

The Streamlit application provides:

- Plant leaf image upload.
- Disease prediction using personalized FedBN models.
- Top-five prediction scores.
- Grad-CAM visualization.
- Model performance dashboard.
- Confusion matrix and classification report.
- FedAvg vs FedBN comparison.
- Project and model information.

## 10. Project Structure

```text
MAJOR PROJECT/
│
├── backend/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── clients/
│   ├── server/
│   ├── evaluation/
│   ├── explainability/
│   └── utils/
│
├── frontend/
│   ├── __init__.py
│   └── app.py
│
├── dataset/
│   └── raw/
│       └── PlantVillage/
│
├── checkpoints/
│
├── results/
│   ├── accuracy/
│   ├── loss/
│   ├── confusion_matrix/
│   ├── gradcam/
│   ├── federated_metrics.csv
│   ├── global_accuracy.png
│   └── global_loss.png
│
├── backups/
│
├── main.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

## 11. Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MAJOR-PROJECT
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Prepare the dataset

Download the PlantVillage dataset and place its class folders inside:

```text
dataset/raw/PlantVillage/
```

### Step 6: Run the application

```bash
python -m streamlit run frontend/app.py
```

Open the local application at:

```text
http://localhost:8501
```

## 12. Model Checkpoints

The application requires the following trained FedBN checkpoints:

```text
checkpoints/
├── fedbn_shared_parameters.pth
├── fedbn_client_0_bn.pth
├── fedbn_client_1_bn.pth
└── fedbn_client_2_bn.pth
```

These files are required for personalized client-model inference.

## 13. Limitations

- The experiments use simulated clients on a single computer rather than physically separate devices.
- The dataset contains curated PlantVillage images.
- Performance on real-world agricultural images may differ.
- The FedAvg and FedBN experiments used different communication-round budgets.
- The model supports only the 15 classes represented in the dataset.
- Grad-CAM explanations are qualitative rather than causal.

## 14. Future Improvements

- Conduct controlled FedAvg and FedBN experiments with equal training rounds.
- Explore non-IID data distributions.
- Evaluate performance on real-world plant images.
- Improve disease classification using pretrained architectures.
- Deploy the application to a cloud platform.
- Explore additional explainability techniques.

## 15. Conclusion

This project demonstrates the implementation of plant disease classification using deep learning and federated learning.

By integrating PyTorch, ResNet18, Flower, FedBN, and Grad-CAM, the system provides an experimental framework for collaborative model training and explainable plant disease prediction.

The Streamlit interface makes the trained models accessible through an interactive image-based prediction application.