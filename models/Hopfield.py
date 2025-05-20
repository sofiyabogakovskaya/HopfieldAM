import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.nn import relu, sigmoid, gelu, softplus, silu
from typing import Callable

import numpy as np

class Hopfield(eqx.Module):
    W: jnp.array
    b: jnp.array
    g: Callable = eqx.static_field()

    def __init__(self, N_neurons, key, g: Callable = relu):
        self.W = random.normal(key, (N_neurons, N_neurons)) / jnp.sqrt(N_neurons)
        self.b = jnp.zeros((N_neurons, ))
        self.g = g

    def __call__(self, t, x, args):
        g = self.g(x)
        return (self.W + self.W.T) @ g / 2 - x + self.b

    def energy(self, x):
        g = self.g(x)
        E = (x - self.b) @ g - jnp.sum(g**2) / 2 - g @ (self.W + self.W.T) @ g / 4
        return E
    