import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import datasets

import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config.config import (
    DEVICE,
    PLANTVILLAGE_DIR,
    RESULTS_DIR,
    NUM_CLIENTS
)

from backend.data.preprocessing import get_train_transforms

from backend.explainability.gradcam import (
    GradCAM,
    load_model
)


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAIR_RESULTS_DIR = RESULTS_DIR / "fair_comparison"

CLASSIFICATION_DIR = (
    FAIR_RESULTS_DIR / "classification_reports"
)

VISUALIZATION_DIR = (
    FAIR_RESULTS_DIR / "visualizations"
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# ==================================================
# CLASS NAMES
# ==================================================

@st.cache_data
def get_class_names():

    dataset = datasets.ImageFolder(
        root=str(PLANTVILLAGE_DIR)
    )

    return dataset.classes


class_names = get_class_names()


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def get_model(method, client_id):

    model = load_model(
        method=method,
        num_classes=len(class_names),
        client_id=client_id
    )

    return model


# ==================================================
# PREDICTION
# ==================================================

def predict_disease(model, uploaded_image):

    transform = get_train_transforms()

    image = uploaded_image.convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        prediction = outputs.argmax(dim=1).item()

        top_probabilities, top_indices = torch.topk(
            probabilities[0],
            k=5
        )

    return (
        image_tensor,
        prediction,
        probabilities,
        top_probabilities,
        top_indices
    )


# ==================================================
# GRAD-CAM
# ==================================================

def generate_gradcam(model, image_tensor, target_class):

    target_layer = model.model.layer4[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    try:

        heatmap = gradcam.generate(
            image_tensor,
            target_class
        )

    finally:

        gradcam.close()

    return heatmap


# ==================================================
# HEADER
# ==================================================

st.title("🌿 Plant Disease Detection Using Federated Learning")

st.markdown(
    """
    ### AI-powered plant disease classification

    Identify plant diseases using deep learning and federated learning.

    This application uses **ResNet18, PyTorch, Flower, FedAvg,
    FedBN and Grad-CAM**.
    """
)

st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🌱 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Home",
        "Disease Detection",
        "Model Performance",
        "FedAvg vs FedBN",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Project Information")

st.sidebar.write("Architecture: ResNet18")
st.sidebar.write("Framework: Flower")
st.sidebar.write("Clients: 3 simulated clients")
st.sidebar.write("Classes: 15")
st.sidebar.write(f"Device: {DEVICE}")


# ==================================================
# HOME
# ==================================================

if page == "Home":

    st.header("Welcome to Plant Disease Detection")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Disease Classes",
        "15"
    )

    col2.metric(
        "Federated Clients",
        "3"
    )

    col3.metric(
        "FedAvg Accuracy",
        "89.34%"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write(
        """
        Plant diseases significantly affect agricultural productivity.

        This project uses deep learning and federated learning
        to identify plant diseases from leaf images.

        Three simulated clients participate in federated training
        using the Flower framework.

        The project compares two federated learning methods:
        FedAvg and FedBN.
        """
    )

    st.subheader("Main Features")

    st.markdown(
        """
        - 🌿 Plant disease classification
        - 🧠 ResNet18 deep learning architecture
        - 🔄 Flower federated learning
        - ⚖️ FedAvg and FedBN comparison
        - 🔥 Grad-CAM explainability
        - 📊 Performance visualization
        """
    )

    st.divider()

    st.subheader("Experimental Results")

    col1, col2 = st.columns(2)

    col1.metric(
        "FedAvg Test Accuracy",
        "89.34%"
    )

    col2.metric(
        "FedBN Test Accuracy",
        "78.44%"
    )


# ==================================================
# DISEASE DETECTION
# ==================================================

elif page == "Disease Detection":

    st.header("🌿 Plant Disease Detection")

    st.write(
        "Upload a plant leaf image to identify its disease."
    )

    method = st.selectbox(
        "Select Federated Learning Method",
        ["fedavg", "fedbn"],
        format_func=lambda x: x.upper()
    )

    client_id = 0

    if method == "fedbn":

        client_id = st.selectbox(
            "Select Personalized Client",
            [0, 1, 2],
            format_func=lambda x: f"Client {x + 1}"
        )

    else:

        st.info(
            "FedAvg uses one global model shared across clients."
        )

    uploaded_file = st.file_uploader(
        "Upload Plant Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            uploaded_image,
            caption="Uploaded Plant Leaf",
            width=400
        )

        if st.button(
            "Predict Disease",
            type="primary"
        ):

            try:

                with st.spinner("Analyzing plant image..."):

                    model = get_model(
                        method,
                        client_id
                    )

                    (
                        image_tensor,
                        prediction,
                        probabilities,
                        top_probabilities,
                        top_indices
                    ) = predict_disease(
                        model,
                        uploaded_image
                    )

                predicted_class = class_names[prediction]

                confidence = (
                    probabilities[0][prediction].item() * 100
                )

                st.divider()

                st.subheader("Prediction Result")

                st.success(
                    f"Predicted Disease: {predicted_class}"
                )

                st.metric(
                    "Prediction Confidence Score",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "Softmax confidence score; not necessarily a calibrated probability."
                )

                st.subheader("Top 5 Predictions")

                for probability, index in zip(
                    top_probabilities,
                    top_indices
                ):

                    class_name = class_names[index.item()]

                    percentage = probability.item() * 100

                    st.write(
                        f"**{class_name}** — {percentage:.2f}%"
                    )

                    st.progress(
                        float(probability.item())
                    )

                st.divider()

                st.subheader("🔥 Grad-CAM Explainability")

                with st.spinner("Generating Grad-CAM..."):

                    heatmap = generate_gradcam(
                        model,
                        image_tensor,
                        prediction
                    )

                original = np.array(uploaded_image)

                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(12, 5)
                )

                axes[0].imshow(original)

                axes[0].set_title(
                    "Original Image"
                )

                axes[0].axis("off")

                axes[1].imshow(original)

                axes[1].imshow(
                    heatmap,
                    cmap="jet",
                    alpha=0.45
                )

                axes[1].set_title(
                    "Predicted Class Grad-CAM"
                )

                axes[1].axis("off")

                plt.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                st.info(
                    """
                    Grad-CAM highlights image regions that influenced
                    the model's prediction. It does not prove that
                    the model identified the biological cause
                    of the disease.
                    """
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.header("📊 Model Performance Dashboard")

    st.subheader("Final Experimental Results")

    col1, col2 = st.columns(2)

    col1.metric(
        "FedAvg Accuracy",
        "89.34%"
    )

    col2.metric(
        "FedBN Accuracy",
        "78.44%"
    )

    st.divider()

    # Accuracy comparison

    accuracy_path = (
        VISUALIZATION_DIR / "accuracy_comparison.png"
    )

    if accuracy_path.exists():

        st.subheader("Accuracy Across Communication Rounds")

        st.image(
            str(accuracy_path),
            width="stretch"
        )

    # Loss comparison

    loss_path = (
        VISUALIZATION_DIR / "loss_comparison.png"
    )

    if loss_path.exists():

        st.subheader("Test Loss Across Communication Rounds")

        st.image(
            str(loss_path),
            width="stretch"
        )

    # Final accuracy

    final_path = (
        VISUALIZATION_DIR / "final_accuracy_comparison.png"
    )

    if final_path.exists():

        st.subheader("Final Accuracy Comparison")

        st.image(
            str(final_path),
            width="stretch"
        )

    st.divider()

    # Classification reports

    st.subheader("FedAvg Classification Report")

    fedavg_report = (
        CLASSIFICATION_DIR
        / "fedavg"
        / "classification_report.txt"
    )

    if fedavg_report.exists():

        st.code(
            fedavg_report.read_text(encoding="utf-8")
        )

    st.subheader("FedBN Classification Report")

    fedbn_report = (
        CLASSIFICATION_DIR
        / "fedbn"
        / "classification_report.txt"
    )

    if fedbn_report.exists():

        st.code(
            fedbn_report.read_text(encoding="utf-8")
        )

    # Confusion matrices

    st.divider()

    st.subheader("Confusion Matrices")

    col1, col2 = st.columns(2)

    fedavg_cm = (
        CLASSIFICATION_DIR
        / "fedavg"
        / "confusion_matrix.png"
    )

    fedbn_cm = (
        CLASSIFICATION_DIR
        / "fedbn"
        / "confusion_matrix.png"
    )

    with col1:

        st.write("FedAvg")

        if fedavg_cm.exists():

            st.image(str(fedavg_cm))

    with col2:

        st.write("FedBN")

        if fedbn_cm.exists():

            st.image(str(fedbn_cm))


# ==================================================
# FEDAVG VS FEDBN
# ==================================================

elif page == "FedAvg vs FedBN":

    st.header("⚖️ FedAvg vs FedBN Comparison")

    st.write(
        "Controlled comparison using the same initial weights, "
        "data partition, test set and training configuration."
    )

    comparison_data = {
        "Metric": [
            "Test Accuracy",
            "Test Loss",
            "Communication Rounds",
            "Number of Clients"
        ],

        "FedAvg": [
            "89.34%",
            "0.3208",
            "5",
            "3"
        ],

        "FedBN": [
            "78.44%",
            "0.6490",
            "5",
            "3"
        ]
    }

    st.table(comparison_data)

    st.divider()

    st.metric(
        "FedAvg Accuracy Advantage",
        "10.90 percentage points"
    )

    st.subheader("Experimental Observation")

    st.write(
        """
        FedAvg achieved higher test accuracy than FedBN
        under the experimental configuration used in this study.

        This result indicates that FedAvg performed better
        on the selected PlantVillage data partition and
        training configuration.
        """
    )

    st.warning(
        """
        FedBN did not outperform FedAvg in this experiment.
        This finding should not be generalized to all datasets
        or federated learning environments.
        """
    )

    st.caption(
        "Both methods used five communication rounds, three clients, "
        "one local epoch per round, batch size 16 and learning rate 0.001."
    )


# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.header("About the Project")

    st.subheader("Project Title")

    st.write(
        "Plant Disease Detection Using Federated Learning"
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - PyTorch
        - Torchvision
        - Flower Federated Learning
        - ResNet18
        - Streamlit
        - Matplotlib
        - Grad-CAM
        """
    )

    st.subheader("Dataset")

    st.write(
        """
        PlantVillage dataset containing 20,638 images
        across 15 plant disease and healthy-leaf classes.
        """
    )

    st.subheader("Federated Learning Architecture")

    st.write(
        """
        Three simulated clients participate in federated training.

        FedAvg aggregates model parameters across clients.

        FedBN aggregates shared parameters while keeping
        Batch Normalization parameters client-specific.
        """
    )

    st.subheader("Important Note")

    st.info(
        """
        This is a research and educational prototype.

        The three clients are simulated on one computer.
        Real-world field performance may differ from
        performance on the PlantVillage dataset.
        """
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Plant Disease Detection | Federated Learning | PyTorch + Flower"
)