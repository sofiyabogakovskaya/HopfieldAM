import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

def get_dataloader(dataset_name="mnist", batch_size=32, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Loads dataset, performs train-validation-test split, and returns data loaders."""
    
    ds = tfds.load(dataset_name, as_supervised=True, batch_size=-1)  

    X, y = tfds.as_numpy(ds["train"])  
    X = X.reshape(X.shape[0], -1)
    X = X.astype(jnp.float32) / 255.0
    
    print(X.shape)

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
        lambda: dataloader(X_train, y_train, batch_size), 
        lambda: dataloader(X_val, y_val, batch_size), 
        lambda: dataloader(X_test, y_test, batch_size),  
    )
