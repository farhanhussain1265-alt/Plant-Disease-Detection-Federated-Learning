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

Both algorithms use the same initial model, dataset split, training partitions, training configuration, and evaluation dataset for a fair comparison.

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
│   └── fair_comparison/
│       ├── fedavg/
│       └── fedbn/
│
├── results/
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

Download the PlantVillage dataset and place its 15 class folders inside:

```text
dataset/raw/PlantVillage/
```

The directory should contain the 15 plant disease class folders.

### Step 5: Download Pretrained Models

The trained model checkpoints are available on Google Drive.

**[Download Pretrained Models](https://drive.google.com/file/d/1qbWozuwNWW0BHO3fT7lsCs1JsLFnPPfR/view?usp=drive_link)**

Download the ZIP file and extract it.

The extracted ZIP contains two folders: `fedavg` and `fedbn`.

Copy these two folders into:

```text
checkpoints/fair_comparison/
```

The final checkpoint structure should look like this:

```text
checkpoints/
└── fair_comparison/
    ├── fedavg/
    │   └── fedavg_final_model.pth
    │
    └── fedbn/
        ├── fedbn_final_shared_parameters.pth
        ├── fedbn_client_0_bn.pth
        ├── fedbn_client_1_bn.pth
        └── fedbn_client_2_bn.pth
```

These pretrained checkpoints allow you to use the trained models without running the federated training experiments again.

## 9. Running the Federated Learning Experiments

To run the FedAvg experiment:

```bash
flwr build --app experiments/fedavg
```

```bash
flwr run experiments/fedavg --stream
```

To run the FedBN experiment:

```bash
flwr build --app experiments/fedbn
```

```bash
flwr run experiments/fedbn --stream
```

**Note:** Running these commands trains the models again. You do not need to run them just to use the pretrained models.

## 10. Model Evaluation

Generate the classification reports:

```bash
python -m backend.evaluation.fair_comparison_classification_report
```

Generate evaluation visualizations:

```bash
python -m backend.evaluation.fair_comparison_visualizations
```

The evaluation includes:

- Classification reports
- Precision, recall, and F1-score
- Confusion matrices
- Performance comparisons

## 11. Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented to visualize the important image regions influencing the model's predictions.

It helps users understand which parts of a leaf image contribute to a disease classification.

Grad-CAM is available for both FedAvg and FedBN models.

## 12. Running the Web Application

Start the Streamlit application:

```bash
streamlit run frontend/app.py
```

The application provides:

- Plant disease prediction using FedAvg and FedBN.
- Model performance comparison.
- Classification reports.
- Confusion matrices.
- Grad-CAM visualizations.

## 13. Future Improvements

- Evaluate different data distributions across clients.
- Experiment with additional federated learning algorithms.
- Improve model generalization.
- Explore larger image resolutions and data augmentation.
- Extend the system to real-world distributed clients.

## 14. Author

**Farhan Hussain**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: https://github.com/farhanhussain1265-alt

LinkedIn: https://linkedin.com/in/farhanhussain-ai

---

**Project:** Plant Disease Detection Using Federated Learning

**Framework:** PyTorch + Flower

**Interface:** Streamlit