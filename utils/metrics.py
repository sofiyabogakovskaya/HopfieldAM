import jax.numpy as jnp
from jax.nn import log_softmax

from utils.integrate import integrate

def accuracy(model, val_loader, dt, N_steps, N_classes):
    correct, total = 0, 0

    for x_batch, y_batch in val_loader():
        x_T = integrate(model, x_batch, dt, N_steps)[-N_classes:]
        log_proba = log_softmax(x_T)
        predictions = jnp.argmax(log_proba, axis=-1)

        correct += jnp.sum(predictions == y_batch)
        total += len(y_batch)

    return correct / total  


# def accuracy(model, x, y, dt, N_steps, N_classes):
#     x_T = integrate(model, x, dt, N_steps)[-N_classes:]
#     log_proba = log_softmax(x_T)
#     return jnp.argmax(log_proba) == y