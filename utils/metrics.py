import jax.numpy as jnp
from jax.nn import log_softmax
from jax import grad, random, config, vmap, jit

from utils.integrate import integrate 

import time
from tqdm import tqdm

def accuracy(model, x, y, dt, N_steps, N_classes):
    x_T = integrate(model, x, dt, N_steps)[-N_classes:]
    log_proba = log_softmax(x_T)
    return jnp.argmax(log_proba) == y

def batch_accuracy(model, val_loader, dt, N_steps, N_classes):
    total_accuracy = 0.0
    num_batches = 0
    for x_batch, y_batch in val_loader():
         batch_acc = jnp.mean(
             vmap(accuracy, in_axes=(None, 0, 0, None, None, None))(model, x_batch, y_batch, dt, N_steps, N_classes)
             )
         total_accuracy += batch_acc
         num_batches += 1
    return total_accuracy / num_batches  