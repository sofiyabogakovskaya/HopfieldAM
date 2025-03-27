import jax.numpy as jnp
import equinox as eqx
import diffrax

from jax import grad, random, config, vmap, jit
from jax.nn import relu, gelu, sigmoid, softmax, log_softmax

from utils.integrate import integrate

def loss_last10(model, x, y, dt, N_steps, N_classes):
    '''last ten elements loss'''
    x_T = integrate(model, x, dt, N_steps)[-N_classes:]
    log_proba = log_softmax(x_T)
    return -log_proba[y]

def batch_loss_last10(model, x, y, dt, N_steps, N_classes):
    return jnp.mean(vmap(loss_last10, in_axes=(None, 0, 0, None, None, None))(model, x, y, dt, N_steps, N_classes))