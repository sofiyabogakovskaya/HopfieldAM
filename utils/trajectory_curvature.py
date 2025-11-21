import jax.numpy as jnp
from jax import vmap, jvp, jit

from typing import Callable, Tuple, Dict



def derivative_based_curvature(Y: jnp.ndarray, 
                               dt: float, 
                               speed_eps: float = 1e-10
                               ) -> jnp.ndarray:
    '''
    Finite-difference curvature estimator on Y (T, d).
    Returns kappa array of length T (NaN at first and last index).
    '''
    T = Y.shape[0]
    # central differences
    v = (Y[2:] - Y[:-2]) / (2.0 * dt)        
    a = (Y[2:] - 2.0 * Y[1:-1] + Y[:-2]) / (dt**2) 
    speed_squared = jnp.sum(v * v, axis=1)        
    # avoid division by zero
    denom = jnp.maximum(speed_squared, speed_eps)
    # projection of a onto v
    dot_v_a = jnp.sum(v * a, axis=1)         
    proj = (dot_v_a / denom)[:, None] * v   
    a_perp = a - proj                         
    kappa_core = jnp.sqrt(jnp.sum(a_perp * a_perp, axis=1)) / denom 
    # assemble into length-T with NaNs at boundaries
    out = jnp.full((T,), jnp.nan)
    out = out.at[1:-1].set(kappa_core)
    return out


def derivative_based_curvature_(model: Callable,
                                X_traj: jnp.ndarray,
                                dt: float,
                                t1: float
                                ) -> jnp.ndarray:
    
    N_steps = int(t1 / dt)
    ts = jnp.linspace(0.0, t1, N_steps + 1)
    ts = jnp.asarray(ts)

    def model_wrapped(ti, xi):
        return model(ti, xi, None)

    V = vmap(model_wrapped)(ts, X_traj) # (T,N)

    # JVP acceleration (we use the chai rule to calculate the accelaration)
    def acceleration(ti, xi, vi):
        z = jnp.concatenate([xi, jnp.array([ti])])
        dz = jnp.concatenate([vi, jnp.array([1.0])])
        def F(zflat):
            x_ = zflat[:-1]
            t_ = zflat[-1]
            return model_wrapped(t_, x_)
        _, a = jvp(F, (z,), (dz,))
        return a

    acceleration_jit = jit(acceleration)

    A = vmap(acceleration_jit)(ts, X_traj, V)   # (T, N)

    # curvature: kappa = || a_perp || / ||v||^2
    V_square = jnp.sum(V * V, axis=1)   # (T,)
    dot_VA = jnp.sum(V * A, axis=1)   # (T,)

    projection = (dot_VA / V_square)[:, None] * V   # (T, N)
    A_perp = A - projection
    A_perp_norm = jnp.linalg.norm(A_perp, axis=1)

    kappa = A_perp_norm / V_square

    return kappa