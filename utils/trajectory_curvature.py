import jax.numpy as jnp


def derivative_based_curvature(Y: jnp.ndarray, dt: float, speed_eps: float = 1e-10) -> jnp.ndarray:
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

def triangle_based_curvature(Y: jnp.ndarray, length_eps: float = 1e-12) -> jnp.ndarray:
    '''
    three-point curvature estimate using triangle area formula.
    For each triple (i-1, i, i+1) compute curvature and place at index i.
    Returns length-T array with NaN at first & last indices.
    '''
    T = Y.shape[0]
    
    p0 = Y[:-2]   
    p1 = Y[1:-1]
    p2 = Y[2:]
    u = p1 - p0
    v = p2 - p0
    u2 = jnp.sum(u * u, axis=1)
    v2 = jnp.sum(v * v, axis=1)
    uv = jnp.sum(u * v, axis=1)
    # squared area * 4 = 4 * (0.5*A)^2 * 4 = (u2*v2 - uv^2)
    inside = jnp.maximum(u2 * v2 - uv * uv, 0.0)
    area = 0.5 * jnp.sqrt(inside)  # area of triangle
    l1 = jnp.sqrt(jnp.sum((p1 - p0) ** 2, axis=1))
    l2 = jnp.sqrt(jnp.sum((p2 - p1) ** 2, axis=1))
    l3 = jnp.sqrt(jnp.sum((p2 - p0) ** 2, axis=1))
    denom = l1 * l2 * l3
    denom = jnp.where(denom <= length_eps, jnp.nan, denom)
    kappa_core = 4.0 * area / denom
    out = jnp.full((T,), jnp.nan)
    out = out.at[1:-1].set(kappa_core)
    return out

