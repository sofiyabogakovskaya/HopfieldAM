import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

def get_dataloader(dataset_name="mnist", batch_size=32):
    """Loads dataset and returns train, validation, and test loaders."""
    # Load dataset from TensorFlow Datasets
    ds = tfds.load(dataset_name, as_supervised=True, batch_size=-1)  # Load all at once

    # Convert to NumPy arrays
    X_train, y_train = tfds.as_numpy(ds["train"])
    X_test, y_test = tfds.as_numpy(ds["test"])

    # Normalize images to [0, 1]
    X_train, X_test = X_train.astype(jnp.float32) / 255.0, X_test.astype(jnp.float32) / 255.0

    # Split validation set (e.g., 10% of training data)
    split_idx = int(0.9 * len(X_train))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # Function to yield batches
    def dataloader(X, y, batch_size):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # Shuffle dataset each epoch
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            yield X[batch_idx], y[batch_idx]

    return (
        lambda: dataloader(X_train, y_train, batch_size),
        lambda: dataloader(X_val, y_val, batch_size),
        lambda: dataloader(X_test, y_test, batch_size),
    )
