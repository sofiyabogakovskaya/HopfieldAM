import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.nn import relu

class Hopfield(eqx.Module):
    W: jnp.array
    b: jnp.array

    def __init__(self, N_neurons, key):
        self.W = random.normal(key, (N_neurons, N_neurons)) / jnp.sqrt(N_neurons)
        self.b = jnp.zeros((N_neurons, ))

    def __call__(self, t, x, args):
        r = relu(x)
        return (self.W + self.W.T) @ r / 2 - x + self.b

    def energy(self, x):
        r = relu(x)
        E = (x - self.b) @ r - jnp.sum(r**2) / 2 - r @ (self.W + self.W.T) @ r / 4
        return E