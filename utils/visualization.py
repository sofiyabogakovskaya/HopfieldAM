import json
import os
import matplotlib.pyplot as plt

def plot_metrics(filename="logs/metrics.json", save_plot=True, plot_filename="logs/training_plot.png"):
    """plots training loss and validation accuracy from the log file."""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run training first!")
        return

    # load logged metrics
    with open(filename, "r") as f:
        metrics = json.load(f)

    # extract loss and accuracy
    epochs = [entry["epoch"] for entry in metrics if "epoch" in entry]
    train_loss = [entry["loss"] for entry in metrics if "loss" in entry]
    val_acc = [entry["val_accuracy"] for entry in metrics if "val_accuracy" in entry]

    if not epochs or not train_loss or not val_acc:
        print("Error: Metrics file does not contain valid training data.")
        return

    fig, ax1 = plt.subplots()

    # plot training loss (left axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_loss, "r-", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # plot validation accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy (%)", color="tab:blue")
    ax2.plot(epochs, [a * 100 for a in val_acc], "b-", label="Validation Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Training Loss and Validation Accuracy")
    fig.tight_layout()

    # save the plot if requested
    if save_plot:
        os.makedirs("logs", exist_ok=True) 
        plt.savefig(plot_filename)
        print(f"plot saved to {plot_filename}")
    
    # plt.show()

if __name__ == "__main__":
    plot_metrics()


def plot_energy(E):
    for e in E[:10]:
        plt.legend()
        plt.plot(e);
