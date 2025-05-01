import json
import os

import equinox as eqx

from datetime import datetime


def new_run_id(prefix="run"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log_experiment(run_id, model, opt_state, config: dict, metrics: dict):
    os.makedirs(f"experiments/{run_id}", exist_ok=True)
    
    with open(f"experiments/{run_id}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(f"experiments/{run_id}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    eqx.tree_serialise_leaves(f"experiments/{run_id}/model.eqx", model)
    eqx.tree_serialise_leaves(f"experiments/{run_id}/opt_state.eqx", opt_state)


def log_summary(run_id, config: dict, metrics: dict, log_path="experiments/results.log"):
    epochs = config.get("epochs", 0)
    batch_size = config.get("batch_size", 0)
    lr = config.get("lr", config.get("learning_rate", 0))
    dt = config.get("dt", 0)
    t1 = config.get("t1", 0)
    N_classes = config.get("N_classes", 0)
    loss = metrics.get("test_loss", 0)
    acc = metrics.get("test_accuracy", 0) * 100  

    line1 = f"{run_id} | loss: {loss:.4f} | acc: {acc:.1f}% | epochs: {epochs} | "
    line2 = f"batch_size: {batch_size} | lr: {lr} | dt: {dt} | t1: {t1} | N_classes: {N_classes}\n"
    summary_line = line1 + line2
 
    with open(log_path, "a") as f:
        f.write(summary_line)


LOG_DIR = "logs"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.json")
TRAINING_LOG = os.path.join(LOG_DIR, "training.log")

def log_message(message):
    """Prints and saves log messages."""
    print(message)
    with open("logs/training.log", "a") as f:
        f.write(message + "\n")


def log_metrics(metrics, filename="logs/metrics.json"):
    """Saves metrics like accuracy, loss, etc., in a JSON file.""" 

    os.makedirs("logs", exist_ok=True)

    # load existing data if file exists
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(metrics)  # append new metrics

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def clear_logs():
    """deletes logs/metrics.json and logs/training.log before training."""
    os.makedirs("logs", exist_ok=True)

    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
        print(f"Cleared: {METRICS_FILE}")

    if os.path.exists(TRAINING_LOG):
        os.remove(TRAINING_LOG)
        print(f"Cleared: {TRAINING_LOG}")

if __name__ == "__main__":
    clear_logs() 