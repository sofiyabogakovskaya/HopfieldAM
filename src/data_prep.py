import jax.numpy as jnp

from sklearn.model_selection import train_test_split

from numpy import genfromtxt


def load_data():
    my_data = genfromtxt('/content/sample_data/mnist_test.csv', delimiter=',')
    features = jnp.array(my_data[:, 1:], dtype=jnp.float32) / 255.0
    targets = jnp.array(my_data[:, 0], dtype=jnp.int32)
    return features, targets

def split_data(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def split_in_batches(X_train, y_train, batch_size):
    num_samples = X_train.shape[0]
    num_batches = num_samples // batch_size

    # TODO: make dataloader to only save single batch at a time in RAM
    X_train = X_train[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    y_train = y_train[: num_batches * batch_size].reshape(num_batches, batch_size)

    return list(zip(X_train, y_train))