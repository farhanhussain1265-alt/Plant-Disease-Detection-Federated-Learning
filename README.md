# Plant Disease Detection Using Federated Learning

A deep learning project that uses Federated Learning to detect plant diseases from leaf images. This project implements and compares two federated learning algorithms: **FedAvg (Federated Averaging)** and **FedBN (Federated Batch Normalization)** using PyTorch and Flower.

## 1. Project Overview

Plant diseases can significantly affect agricultural productivity. This project explores how Federated Learning can be used for plant disease classification while keeping training data distributed across multiple clients.

A ResNet18-based image classification model is trained using three simulated federated clients.

The project compares FedAvg and FedBN under the same experimental conditions and provides a Streamlit web application for model inference, performance visualization, and Grad-CAM explainability.

## 2. Technologies Used

- Python 3.11
- PyTorch
- Torchvision
- Flower
- ResNet18
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenCV
- Streamlit
- Grad-CAM

## 3. Dataset

**Dataset:** PlantVillage

- Total images: 20,638
- Number of classes: 15
- Training images: 16,510
- Testing images: 4,128
- Train-test split: 80:20
- Random seed: 42
- Image size: 128 × 128

The dataset contains healthy and diseased plant leaf images from different categories.

The training dataset is distributed among three simulated federated clients.

**Note:** The clients are simulated on a single computer using Flower. They are not three separate physical devices.

## 4. Federated Learning Algorithms

### FedAvg

Federated Averaging trains models locally on individual clients and aggregates their model parameters on the server to produce a global model.

### FedBN

Federated Batch Normalization keeps Batch Normalization parameters local to each client while aggregating the remaining model parameters.

Both algorithms use the same initial model, dataset split, training configuration, and evaluation dataset for a fair comparison.

## 5. Experimental Configuration

| Parameter | Value |
|---|---|
| Model | ResNet18 |
| Number of clients | 3 |
| Communication rounds | 5 |
| Local epochs | 1 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Image size | 128 × 128 |
| Dataset split seed | 42 |
| Federated learning framework | Flower |

## 6. Experimental Results

The following results were obtained from the final five-round experiment.

| Metric | FedAvg | FedBN |
|---|---:|---:|
| Test Accuracy | 89.34% | 78.44% |
| Test Loss | 0.3208 | 0.6490 |
| Macro Precision | 0.88 | 0.77 |
| Macro Recall | 0.88 | 0.78 |
| Macro F1-Score | 0.87 | 0.77 |
| Weighted F1-Score | 0.89 | 0.78 |

### Key Observation

FedAvg achieved an accuracy of 89.34%, while FedBN achieved 78.44%.

FedAvg performed approximately 10.90 percentage points better than FedBN under the experimental data partition and training configuration used in this study.

These results show that FedBN did not outperform FedAvg in this particular experiment.

## 7. Project Structure

```text
Plant-Disease-Detection-Federated-Learning/
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
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── experiments/
│   ├── fedavg/
│   └── fedbn/
│
├── frontend/
│   └── app.py
│
├── checkpoints/
│
├── results/
│
├── backups/
│
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/farhanhussain1265-alt/Plant-Disease-Detection-Federated-Learning.git

cd Plant-Disease-Detection-Federated-Learning
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Download the PlantVillage dataset and place its class folders inside:

```text
dataset/raw