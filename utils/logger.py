import json
import os

LOG_DIR = "logs"
METRICS_FILE = os.path.join(LOG_DIR, "metrics.json")
TRAINING_LOG = os.path.join(LOG_DIR, "training.log")

def log_message(message):
    """Prints and saves log messages."""
    print(message)
    with open("logs/training.log", "a") as f:
        f.write(message + "\n")


# def clear_first(filename="logs/metrics.json", clear=True):
#     if clear and os.path.exists(filename):
#         os.remove(filename)


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
    """Deletes both logs/metrics.json and logs/training.log before training."""
    os.makedirs("logs", exist_ok=True)

    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
        print(f"Cleared: {METRICS_FILE}")

    if os.path.exists(TRAINING_LOG):
        os.remove(TRAINING_LOG)
        print(f"Cleared: {TRAINING_LOG}")

if __name__ == "__main__":
    clear_logs() 