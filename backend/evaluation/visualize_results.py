import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

from backend.config.config import RESULTS_DIR


def visualize_results():

    metrics_file = RESULTS_DIR / "federated_metrics.csv"

    if not metrics_file.exists():
        print("Metrics CSV file not found!")
        return

    df = pd.read_csv(metrics_file)

    if df.empty:
        print("No experiment results available!")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Global Accuracy Graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["round"],
        df["global_test_accuracy"] * 100,
        marker="o",
        label="Global Test Accuracy"
    )

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning - Global Accuracy")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        RESULTS_DIR / "global_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Global Loss Graph
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["round"],
        df["global_test_loss"],
        marker="o",
        label="Global Test Loss"
    )

    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    plt.title("Federated Learning - Global Loss")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        RESULTS_DIR / "global_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nVisualization completed successfully!")
    print("Accuracy Graph:", RESULTS_DIR / "global_accuracy.png")
    print("Loss Graph:", RESULTS_DIR / "global_loss.png")


if __name__ == "__main__":
    visualize_results()