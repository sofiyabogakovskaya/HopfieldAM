import json
import os
import matplotlib.pyplot as plt

def plot_metrics(filename="logs/metrics.json"):
    """Plots training loss and validation accuracy from the log file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run training first!")
        return

    # Load logged metrics
    with open(filename, "r") as f:
        metrics = json.load(f)

    # Extract loss and accuracy
    epochs = [entry["epoch"] for entry in metrics if "epoch" in entry]
    train_loss = [entry["loss"] for entry in metrics if "loss" in entry]
    val_acc = [entry["val_accuracy"] for entry in metrics if "val_accuracy" in entry]
    # epochs = sorted(int(k) for k in metrics.keys())  # Convert keys to integers for sorting
    # train_loss = [metrics[str(epoch)]["train_loss"] for epoch in epochs]
    # val_acc = [metrics[str(epoch)]["val_accuracy"] for epoch in epochs]

    if not epochs or not train_loss or not val_acc:
        print("Error: Metrics file does not contain valid training data.")
        return

    # Create the figure
    fig, ax1 = plt.subplots()

    # Plot training loss (left axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_loss, "r-", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Plot validation accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy (%)", color="tab:blue")
    ax2.plot(epochs, [a * 100 for a in val_acc], "b-", label="Validation Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Titles and layout
    fig.suptitle("Training Loss and Validation Accuracy")
    fig.tight_layout()

    plt.show()

def plot_energy(E):
    for e in E[:10]:
        plt.legend()
        plt.plot(e);

# If running standalone:
if __name__ == "__main__":
    plot_metrics()

