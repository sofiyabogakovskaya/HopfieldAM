import json
import os

import jax.numpy as jnp

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np

from utils.integrate_trajectory import get_energy
from utils.energy_derivative import get_energy_dot


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



def plot_energy(run_id, model, X_batch, y_batch, dt, t1, samples, save_plot=True, ylim=None, xlim=None, plot_mean=False, plot_only_mean=False, plot_path=None):  
      
    X, y, ts, batch_E = get_energy(model, X_batch, y_batch, dt, t1, samples)
    output_dir = f"experiments/{run_id}"
    
    if plot_path is None:
        plot_path = os.path.join(output_dir, f"energy_plot_{samples}samples.png")

    # plot, coloring by digit
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap("tab10")  # 10 distinct colors

    for digit in range(10):
        idx = jnp.where(y == digit)[0]
        for i in idx:
            if not plot_only_mean:
                plt.plot(ts, batch_E[i],
                        color=colors(digit),
                        label=str(digit) if i == idx[0] else None,
                        alpha=0.7)
        # plt.plot(ts, ba)
        if plot_mean:
            E_mean_for_digit = jnp.mean(batch_E[idx], axis=0)

            if batch_E[idx].shape[0] > 0:
                plt.plot(ts, E_mean_for_digit,
                        linestyle="--",
                        color=colors(digit),
                        label=f"{digit}: <E>",
                        alpha=0.7)

    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.grid(alpha=0.4)
    plt.title(f"{run_id}: Energy vs Time for {samples} samples, (t ∈ [0,{t1}])")
    plt.legend(ncol=5, fontsize=10)
    
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    else:
        plt.ylim(bottom=min(batch_E.flatten()) * 1.1)  # show negative energies clearly
        
    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {plot_path}")

    plt.show()
    plt.close("all")
        

def plot_energy_dot(run_id, model, X_batch, y_batch, dt, t1, samples, save_plot=False):    
    X, y, ts, batch_E = get_energy_dot(model, X_batch, y_batch, dt, t1, samples)

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
    plt.ylabel("Energy deriviative")
    plt.title(f"{run_id}: Energy derivative for {samples} samples (t ∈ [0,{t1}])")
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


def create_batch_to_compute_energy(numbers: list, each_number_samples: int, dataset_name="mnist", split="test"):
    # load full dataset as arrays
    ds = tfds.load(dataset_name, split=split, as_supervised=True, batch_size=-1)
    images, labels = ds[0].numpy(), ds[1].numpy()  # Convert to numpy arrays
    
    selected_images = []
    selected_labels = []
    
    for number in numbers:
        # find indices for current number
        mask = labels == number
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(f"Number {number} not found in {split} split")
        
        # sample with replacement if needed
        replace = len(indices) < each_number_samples
        selected_indices = np.random.choice(indices, size=each_number_samples, replace=replace)
        
        selected_images.append(images[selected_indices])
        selected_labels.append(labels[selected_indices])
    
    # combine and shuffle
    combined_images = np.concatenate(selected_images, axis=0)
    combined_labels = np.concatenate(selected_labels, axis=0)
    
    # shuffle the batch
    shuffle_idx = np.random.permutation(len(combined_images))
    
    X = jnp.array(combined_images[shuffle_idx]).squeeze()
    X = X.reshape(X.shape[0], -1)
    y = jnp.array(combined_labels[shuffle_idx])
    
    return X, y