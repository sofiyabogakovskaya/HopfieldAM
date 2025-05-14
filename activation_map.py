from jax.nn import relu, sigmoid, tanh, silu, gelu, softplus

ACTIVATION_MAP = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "silu": silu,
    "gelu": gelu,
    "softplus": softplus
}
