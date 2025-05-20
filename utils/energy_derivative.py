from jax import vmap
from jax.nn import relu
import jax.numpy as jnp

from utils.integrate_trajectory import integrate_trajectory

def relu_derivative(y):
    """computes the derivative of ReLU w.r.t. y"""
    return jnp.diag((y > 0).astype(float))

def energy_derivative(model, x):
    """computes approximate energy derivative over y (normalized x) for ReLU case"""
    y = x / (jnp.linalg.norm(x))
    g = model.g(y)
    W = model.W
    I = jnp.eye(len(x))
    E_dot = (-W @ g + y).T @ (relu_derivative(y)) @ (I - jnp.outer(y, y)) @ (W @ g)
    return E_dot

def get_energy_dot(model, X_batch, y_batch, dt, t1, samples):
    N_steps = int(t1 / dt)
    ts = jnp.linspace(0.0, t1, N_steps + 1)

    X = X_batch[:samples]
    y = y_batch[:samples]

    batch_trajs = vmap(integrate_trajectory, in_axes=(None, 0, None, None, None))(
        model, X, dt, t1, ts
        )

    batch_E_dot = vmap(lambda traj: vmap(energy_derivative, in_axes=(None, 0))(model, traj))(batch_trajs)
    return X, y, ts, batch_E_dot