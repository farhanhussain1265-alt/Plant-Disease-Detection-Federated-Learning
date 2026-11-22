import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from backend.config.config import RESULTS_DIR


# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

FAIR_RESULTS_DIR = RESULTS_DIR / "fair_comparison"

FEDAVG_METRICS_PATH = FAIR_RESULTS_DIR / "fedavg" / "metrics.csv"
FEDBN_METRICS_PATH = FAIR_RESULTS_DIR / "fedbn" / "metrics.csv"
FINAL_COMPARISON_PATH = FAIR_RESULTS_DIR / "final_comparison.csv"

OUTPUT_DIR = FAIR_RESULTS_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load experimental results
# --------------------------------------------------

fedavg = pd.read_csv(FEDAVG_METRICS_PATH)
fedbn = pd.read_csv(FEDBN_METRICS_PATH)
comparison = pd.read_csv(FINAL_COMPARISON_PATH)

# Exclude FedAvg round 0 for fair round-by-round comparison
fedavg = fedavg[fedavg["round"] >= 1].copy()

print("Experimental metrics loaded successfully!")


# --------------------------------------------------
# 3. Accuracy comparison
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    fedavg["round"],
    fedavg["global_test_accuracy"] * 100,
    marker="o",
    linewidth=2,
    label="FedAvg"
)

plt.plot(
    fedbn["round"],
    fedbn["test_accuracy"] * 100,
    marker="s",
    linewidth=2,
    label="FedBN"
)

plt.title("FedAvg vs FedBN - Test Accuracy")
plt.xlabel("Communication Rounds")
plt.ylabel("Test Accuracy (%)")
plt.xticks(range(1, 6))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300)
plt.close()


# --------------------------------------------------
# 4. Test loss comparison
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    fedavg["round"],
    fedavg["global_test_loss"],
    marker="o",
    linewidth=2,
    label="FedAvg"
)

plt.plot(
    fedbn["round"],
    fedbn["test_loss"],
    marker="s",
    linewidth=2,
    label="FedBN"
)

plt.title("FedAvg vs FedBN - Test Loss")
plt.xlabel("Communication Rounds")
plt.ylabel("Test Loss")
plt.xticks(range(1, 6))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "loss_comparison.png", dpi=300)
plt.close()


# --------------------------------------------------
# 5. Final accuracy comparison
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.bar(
    comparison["Method"],
    comparison["Final Accuracy (%)"]
)

plt.title("Final Accuracy: FedAvg vs FedBN")
plt.xlabel("Federated Learning Method")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(comparison["Final Accuracy (%)"]):
    plt.text(
        index,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "final_accuracy_comparison.png", dpi=300)
plt.close()


# --------------------------------------------------
# 6. Completion message
# --------------------------------------------------

print("\nAll visualizations generated successfully!")

print("\nSaved files:")

for file in OUTPUT_DIR.iterdir():
    print(file)