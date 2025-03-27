import json
import os

def log_message(message):
    """Prints and saves log messages."""
    print(message)
    with open("logs/training.log", "a") as f:
        f.write(message + "\n")

def log_metrics(metrics, filename="logs/metrics.json"):
    """Saves metrics like accuracy, loss, etc., in a JSON file."""
    os.makedirs("logs", exist_ok=True)
    
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(metrics)  # Append new metrics

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)