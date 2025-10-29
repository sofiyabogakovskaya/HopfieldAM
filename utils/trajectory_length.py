import jax.numpy as jnp

def trajectory_length(x):
    diffs = jnp.diff(x, axis=0)
    step_norms = jnp.linalg.norm(diffs, axis=-1)
    lengths = jnp.sum(step_norms, axis=0)
    return lengths