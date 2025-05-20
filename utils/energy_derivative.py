from jax.nn import relu

import numpy as np

def relu_derivative(y):
    """Computes the derivative of ReLU w.r.t. y."""
    return np.diag((y > 0).astype(float))

def energy_derivative(model, x):
    y = x / (np.linalg.norm(x))
    g = model.g(y)
    W = model.W
    I = np.eye(len(x))
    E_dot = (-W @ g + y).T @ (relu_derivative(y)) @ (I - y.T @ y) @ (W @ g)
    return E_dot