import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets

from backend.config.config import (
    DEVICE,
    PLANTVILLAGE_DIR,
    RESULTS_DIR
)

from backend.data.preprocessing import get_train_transforms

from backend.explainability.gradcam import (
    GradCAM,
    load_client_model
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

@st.cache_data
def get_class_names():

    dataset = datasets.ImageFolder(
        root=str(PLANTVILLAGE_DIR)
    )

    return dataset.classes


class_names = get_class_names()


# --------------------------------------------------
# LOAD FEDBN MODEL
# --------------------------------------------------

@st.cache_resource
def get_model(client_id):

    model = load_client_model(
        client_id=client_id,
        num_classes=len(class_names)
    )

    return model


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

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


# --------------------------------------------------
# GRAD-CAM FUNCTION
# --------------------------------------------------

def generate_gradcam(model, image_tensor, target_class):

    target_layer = model.model.layer3[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    heatmap = gradcam.generate(
        image_tensor,
        target_class=target_class
    )

    # Remove hook after generating the heatmap
    target_layer._forward_hooks.clear()

    return heatmap


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🌿 Plant Disease Detection Using Federated Learning")

st.markdown(
    """
    ### AI-powered plant disease classification

    This application uses **ResNet18, Flower Federated Learning,
    and FedBN** to identify plant diseases from leaf images.

    Upload a leaf image to predict its disease and visualize
    the model's attention using Grad-CAM.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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

st.sidebar.write("### Model Information")

st.sidebar.write("Architecture: ResNet18")
st.sidebar.write("Federated Framework: Flower")
st.sidebar.write("Aggregation: FedBN")
st.sidebar.write("Number of Clients: 3")
st.sidebar.write("Number of Classes: 15")

st.sidebar.write(f"Device: {DEVICE}")


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.header("Welcome to Plant Disease Detection")

    col1, col2, col3 = st.columns(3)

    col1.metric("Plant Disease Classes", "15")

    col2.metric("Federated Clients", "3")

    col3.metric("FedBN Test Accuracy", "81.81%")

    st.divider()

    st.subheader("Project Overview")

    st.write(
        """
        Plant diseases can significantly affect agricultural
        productivity.

        This project uses deep learning and federated learning
        to classify plant diseases from leaf images.

        Instead of training one model using centralized client
        data, three simulated clients participate in federated
        training through the Flower framework.
        """
    )

    st.subheader("Main Features")

    st.markdown(
        """
        - 🌿 Plant disease image classification
        - 🧠 ResNet18 deep learning architecture
        - 🔄 Flower-based federated learning
        - 🏥 FedBN personalized Batch Normalization
        - 🔍 Grad-CAM explainability
        - 📊 Model performance evaluation
        """
    )


# ==================================================
# DISEASE DETECTION PAGE
# ==================================================

elif page == "Disease Detection":

    st.header("🌿 Plant Disease Detection")

    st.write(
        "Upload an image of a plant leaf to identify its disease."
    )

    client_id = st.selectbox(
        "Select Personalized Client Model",
        [0, 1, 2],
        format_func=lambda x: f"Client {x + 1}"
    )

    uploaded_file = st.file_uploader(
        "Upload Plant Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        uploaded_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            uploaded_image,
            caption="Uploaded Plant Leaf",
            width=400
        )

        if st.button("Predict Disease", type="primary"):

            try:

                with st.spinner("Loading model and analyzing image..."):

                    model = get_model(client_id)

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

                confidence = probabilities[0][prediction].item() * 100

                st.divider()

                st.subheader("Prediction Result")

                st.success(
                    f"Predicted Disease: {predicted_class}"
                )

                st.metric(
                    "Prediction Score",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "Softmax score; it is not necessarily a calibrated probability."
                )

                # Top 5 predictions

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

                # Grad-CAM

                st.divider()

                st.subheader("🔍 Grad-CAM Explainability")

                with st.spinner("Generating Grad-CAM visualization..."):

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

                axes[0].set_title("Original Image")

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
                    "Grad-CAM highlights image regions that influenced "
                    "the model's prediction. It does not prove that "
                    "the model has identified the biological cause "
                    "of the disease."
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ==================================================
# MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.header("📊 Model Performance Dashboard")

    st.subheader("FedBN Evaluation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "81.81%")

    col2.metric("Macro F1 Score", "0.79")

    col3.metric("Weighted F1 Score", "0.82")

    st.divider()

    # Accuracy graph

    accuracy_path = RESULTS_DIR / "global_accuracy.png"

    if accuracy_path.exists():

        st.subheader("Federated Training Accuracy")

        st.image(
            str(accuracy_path),
            width="stretch"
        )

    # Loss graph

    loss_path = RESULTS_DIR / "global_loss.png"

    if loss_path.exists():

        st.subheader("Federated Training Loss")

        st.image(
            str(loss_path),
            width="stretch"
        )

    # Confusion matrix

    confusion_path = (
        RESULTS_DIR
        / "confusion_matrix"
        / "fedbn_confusion_matrix.png"
    )

    if confusion_path.exists():

        st.subheader("FedBN Confusion Matrix")

        st.image(
            str(confusion_path),
            width="stretch"
        )

    # Classification report

    report_path = RESULTS_DIR / "fedbn_classification_report.txt"

    if report_path.exists():

        st.subheader("Classification Report")

        with open(report_path, "r") as file:

            report = file.read()

        st.code(report)


# ==================================================
# FEDAVG VS FEDBN PAGE
# ==================================================

elif page == "FedAvg vs FedBN":

    st.header("⚖️ FedAvg vs FedBN Comparison")

    st.write(
        "Comparison of the observed classification results."
    )

    comparison_data = {
        "Metric": [
            "Test Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1 Score",
            "Weighted F1 Score"
        ],

        "FedAvg": [
            "6.66%",
            "0.02",
            "0.08",
            "0.02",
            "0.02"
        ],

        "FedBN": [
            "81.81%",
            "0.79",
            "0.79",
            "0.79",
            "0.82"
        ]
    }

    st.table(comparison_data)

    st.divider()

    st.metric(
        "Observed Accuracy Difference",
        "75.15 percentage points"
    )

    st.warning(
        """
        Experimental limitation:

        The FedAvg baseline was trained for one communication
        round, whereas FedBN was trained for five rounds.

        Therefore, this is an observed comparison, not a controlled
        experiment with equal training budgets. The improvement
        cannot be attributed solely to the FedBN method.
        """
    )


# ==================================================
# ABOUT PROJECT PAGE
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
        Three simulated clients train personalized models.
        Shared model parameters are aggregated by the Flower
        server, while Batch Normalization parameters remain
        client-specific.
        """
    )

    st.subheader("Important Note")

    st.info(
        """
        This application is a research and educational prototype.
        Its performance on real-world field images may differ
        from its performance on the PlantVillage test dataset.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Plant Disease Detection | Federated Learning | PyTorch + Flower"
)