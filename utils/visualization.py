import json
import os
import matplotlib.pyplot as plt

def plot_metrics(run_id, save_plot=True):
    """plots training loss and validation accuracy from the log file."""

    metrics_path = f"experiments/{run_id}/metrics.json"
    output_dir = f"experiments/{run_id}"
    plot_path = os.path.join(output_dir, "training_plot.png")

    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found. Please ensure metrics have been logged.")
        return
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    train_losses = metrics.get("train_losses")
    val_accuracies = metrics.get("val_accuracies")

    if not (isinstance(train_losses, list) and isinstance(val_accuracies, list)):
        print("Error: 'train_losses' and 'val_accuracies' must be lists in metrics.json.")
        return

    if len(train_losses) == 0 or len(val_accuracies) == 0:
        print("Error: No data found in 'train_losses' or 'val_accuracies'.")
        return

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots()

    # training loss (left axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.plot(epochs, train_losses, 'r-', label='Train Loss')
    ax1.tick_params(axis="y")

    # validation accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.plot(epochs, [a * 100 for a in val_accuracies], 'b-', label='Val Accuracy')
    ax2.tick_params(axis="y")

    # title and layout
    plt.title("Training Loss and Validation Accuracy over Epochs")
    fig.tight_layout()

    # save plot if requested
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    # Show plot
    plt.show()




# def plot_metrics(filename="logs/metrics.json", save_plot=True, plot_filename="logs/training_plot.png"):
#     """plots training loss and validation accuracy from the log file."""
#     if not os.path.exists(filename):
#         print(f"Error: {filename} not found. Run training first!")
#         return

#     # load logged metrics
#     with open(filename, "r") as f:
#         metrics = json.load(f)

#     # extract loss and accuracy
#     epochs = [entry["epoch"] for entry in metrics if "epoch" in entry]
#     train_loss = [entry["loss"] for entry in metrics if "loss" in entry]
#     val_acc = [entry["val_accuracy"] for entry in metrics if "val_accuracy" in entry]

#     if not epochs or not train_loss or not val_acc:
#         print("Error: Metrics file does not contain valid training data.")
#         return

#     fig, ax1 = plt.subplots()

#     # plot training loss (left axis)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Loss", color="tab:red")
#     ax1.plot(epochs, train_loss, "r-", label="Train Loss")
#     ax1.tick_params(axis="y", labelcolor="tab:red")

#     # plot validation accuracy (right axis)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("Validation Accuracy (%)", color="tab:blue")
#     ax2.plot(epochs, [a * 100 for a in val_acc], "b-", label="Validation Accuracy")
#     ax2.tick_params(axis="y", labelcolor="tab:blue")

#     fig.suptitle("Training Loss and Validation Accuracy")
#     fig.tight_layout()

#     # save the plot if requested
#     if save_plot:
#         os.makedirs("logs", exist_ok=True) 
#         plt.savefig(plot_filename)
#         print(f"plot saved to {plot_filename}")
    
#     # plt.show()

if __name__ == "__main__":
    plot_metrics()


def plot_energy(E):
    for e in E[:10]:
        plt.legend()
        plt.plot(e);
