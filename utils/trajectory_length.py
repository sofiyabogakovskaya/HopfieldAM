import jax.numpy as jnp

def trajectory_length(x):
    diffs = jnp.diff(x, axis=0)
    step_norms = jnp.linalg.norm(diffs, axis=-1)
    length = jnp.sum(step_norms, axis=0)
    return length 

def trajectory_direction(x):
    x_0 = x[0]
    x_T = x[-1]
    dist_vec = jnp.linalg.norm(x_T - x_0)
    dist = jnp.sum(dist_vec)
    return dist

def how_straight(traj_length, traj_dir):
    return traj_length - traj_dir