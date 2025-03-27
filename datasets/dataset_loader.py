import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

def get_dataloader(dataset_name="mnist", batch_size=32, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Loads dataset, performs train-validation-test split, and returns data loaders."""
    
    ds = tfds.load(dataset_name, as_supervised=True, batch_size=-1)  

    X, y = tfds.as_numpy(ds["train"])  

    X = X.astype(jnp.float32) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio/(1 - test_ratio), 
                                                       random_state=random_seed, stratify=y_train)

    # Function to yield batches
    def dataloader(X, y, batch_size):
        """Yields batches of (X, y) from the dataset."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # Shuffle dataset each epoch
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i : i + batch_size]
            yield X[batch_idx], y[batch_idx]

    return (
        lambda: dataloader(X_train, y_train, batch_size),  # Train loader
        lambda: dataloader(X_val, y_val, batch_size),  # Validation loader
        lambda: dataloader(X_test, y_test, batch_size),  # Test loader
    )

# import jax.numpy as jnp

# from sklearn.model_selection import train_test_split

# from numpy import genfromtxt


# def load_data():
#     my_data = genfromtxt('/content/sample_data/mnist_test.csv', delimiter=',')
#     features = jnp.array(my_data[:, 1:], dtype=jnp.float32) / 255.0
#     targets = jnp.array(my_data[:, 0], dtype=jnp.int32)
#     return features, targets

# def split_data(features, targets):
#     X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test


# def split_in_batches(X_train, y_train, batch_size):
#     num_samples = X_train.shape[0]
#     num_batches = num_samples // batch_size

#     # TODO: make dataloader to only save single batch at a time in RAM
#     X_train = X_train[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
#     y_train = y_train[: num_batches * batch_size].reshape(num_batches, batch_size)

#     return list(zip(X_train, y_train))


# def get_dataloader(batch_size=64):
#     features, targets = load_data()
#     # TODO: batching for test data, because it can be big and not fit into GPU
#     # also make batching with dataloader object for test
#     X_train, X_test, y_train, y_test = split_data(features, targets)
#     data_batches = split_in_batches(X_train, y_train, batch_size)
#     return data_batches, X_test, y_test


