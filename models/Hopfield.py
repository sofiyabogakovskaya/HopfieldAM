import jax.numpy as jnp
import equinox as eqx

from jax import random
from jax.nn import relu, sigmoid, gelu, softplus, tanh
from jax.scipy.special import spence  # note: spence(x) = Li₂(1 - x)
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

    def lagrangian(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.g is relu:
            r = self.g(x)
            return jnp.sum(r**2) / 2
        elif self.g is sigmoid:
            return jnp.sum(softplus(x))
        elif self.g is tanh:
            return jnp.sum(jnp.log(jnp.cosh(x)))
        elif self.g is softplus:
            return x * jnp.log1p(jnp.exp(x)) + spence(1 + jnp.exp(x)) + + (jnp.pi**2) / 12
        # elif self.g is gelu:
        #     
        else:
            raise NotImplementedError(f"lagrangian not implemented for activation {self.g}")


    def energy(self, x, g: Callable = relu):
        g = self.g(x)
        L = self.lagrangian(x)
        E = (x - self.b) @ g - L - g @ (self.W + self.W.T) @ g / 4
        return E
    