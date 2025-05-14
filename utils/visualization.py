import json
import os

import jax.numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt

from utils.integrate_trajectory import get_energy


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
    ax1.legend()

    # validation accuracy (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.plot(epochs, [a * 100 for a in val_accuracies], 'b-', label='Val Accuracy')
    ax2.tick_params(axis="y")
    ax2.legend()

    # title and layout
    plt.title(f"{run_id}: training loss and validation accuracy over epochs")
    fig.tight_layout()
    plt.grid(alpha=0.4)

    # save plot if requested
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    plt.show()
    plt.close("all")

if __name__ == "__main__":
    plot_metrics()


def plot_energy(run_id, model, X_batch, y_batch, dt, t1, samples, save_plot=True):    
    X, y, ts, batch_E = get_energy(model, X_batch, y_batch, dt, t1, samples)

    output_dir = f"experiments/{run_id}"
    plot_path = os.path.join(output_dir, f"energy_plot_{samples}samples.png")

    # plot, coloring by digit
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("tab10")  # 10 distinct colors

    for digit in range(10):
        idx = jnp.where(y == digit)[0]
        for i in idx:
            plt.plot(ts, batch_E[i],
                    color=colors(digit),
                    label=str(digit) if i == idx[0] else None,
                    alpha=0.7)

    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title(f"{run_id}: Energy vs Time for first {samples} samples (t ∈ [0,{t1}])")
    plt.legend(title="Digit", ncol=5, fontsize="small")
    plt.grid(alpha=0.4)
    plt.ylim(-50, 100)
    plt.ylim(bottom=min(batch_E.flatten()) * 1.1)  # show negative energies clearly

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    plt.show()
    plt.close("all")
        

