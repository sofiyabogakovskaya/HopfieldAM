import jax.numpy as jnp
from jax import vmap
from typing import Optional, Tuple, Dict

def moving_average_smooth(X: jnp.ndarray, window: int) -> jnp.ndarray:
    '''
    simple moving average along time axis. window must be odd.
    '''
    assert window >= 1 and window % 2 == 1
    K = jnp.ones((window,), dtype=X.dtype) / window
    # convolve each channel
    def conv_col(col):
        return jnp.convolve(col, K, mode='same')
    return vmap(conv_col, in_axes=1, out_axes=1)(X)

def gaussian_smooth(X: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Gaussian smoothing along time axis (sigma>0)."""
    if sigma <= 0.0:
        return X
    radius = int(jnp.ceil(3.0 * sigma))
    xs = jnp.arange(-radius, radius + 1)
    K = jnp.exp(-(xs**2) / (2.0 * sigma**2))
    K = K / jnp.sum(K)
    def conv_col(col):
        return jnp.convolve(col, K, mode='same')
    return vmap(conv_col, in_axes=1, out_axes=1)(X)

# -------------------------
# Local PCA projection
# -------------------------
def local_pca_project_center(X: jnp.ndarray, window: int, pca_k: int) -> jnp.ndarray:
    """
    For each time index i, compute PCA on window centered at i and return
    the projection of the center point onto the top-pca_k components.
    X: (T, N)
    window: odd integer >=3
    returns: Y (T, pca_k)
    """
    assert window >= 3 and window % 2 == 1
    T, N = X.shape
    w = (window - 1) // 2
    padded = jnp.pad(X, ((w, w), (0, 0)), mode='reflect')
    W = window

    def extract_window(i):
        # dynamic_slice args: (array, start_indices, slice_sizes)
        return jax.lax.dynamic_slice(padded, (i, 0), (W, N))  # (W, N)

    indices = jnp.arange(T)
    windows = vmap(extract_window)(indices)  # (T, W, N)

    def project_window(win):
        # win: (W, N)
        mean = jnp.mean(win, axis=0)
        centered = win - mean  # (W, N)
        # SVD of the centered window (small W, N)
        # full_matrices=False gives U:(W,r), S:(r,), Vt:(r,N)
        _, _, Vt = jnp.linalg.svd(centered, full_matrices=False)
        # handle case where rank < pca_k
        # Vt shape: (r, N), r = min(W, N)
        r = Vt.shape[0]
        k = jnp.minimum(pca_k, r)
        comps = Vt[:k, :]  # (k, N)
        center = win[w, :]  # central point
        proj_coords = (center - mean) @ comps.T  # (k,)
        # If k < pca_k, pad with zeros
        if k < pca_k:
            pad = jnp.zeros((pca_k - k,), dtype=proj_coords.dtype)
            proj_coords = jnp.concatenate([proj_coords, pad], axis=0)
        return proj_coords

    Y = vmap(project_window)(windows)  # (T, pca_k)
    return Y

# -------------------------
# Curvature estimators
# -------------------------
def derivative_based_curvature(Y: jnp.ndarray, dt: float, speed_eps: float = 1e-10) -> jnp.ndarray:
    """
    Finite-difference curvature estimator on Y (T, d).
    Returns kappa array of length T (NaN at first and last index).
    """
    T, d = Y.shape
    if T < 3:
        return jnp.full((T,), jnp.nan)
    # central differences
    v = (Y[2:] - Y[:-2]) / (2.0 * dt)          # (T-2, d)
    a = (Y[2:] - 2.0 * Y[1:-1] + Y[:-2]) / (dt**2)  # (T-2, d)
    speed2 = jnp.sum(v * v, axis=1)            # (T-2,)
    # avoid division by zero
    denom = jnp.maximum(speed2, speed_eps)
    # projection of a onto v
    dot_v_a = jnp.sum(v * a, axis=1)           # (T-2,)
    proj = (dot_v_a / denom)[:, None] * v      # (T-2, d)
    a_perp = a - proj                           # (T-2, d)
    kappa_core = jnp.sqrt(jnp.sum(a_perp * a_perp, axis=1)) / denom  # (T-2,)
    # assemble into length-T with NaNs at boundaries
    out = jnp.full((T,), jnp.nan)
    out = out.at[1:-1].set(kappa_core)
    return out

def triangle_based_curvature(Y: jnp.ndarray, length_eps: float = 1e-12) -> jnp.ndarray:
    """
    Three-point curvature estimate using triangle area formula.
    For each triple (i-1, i, i+1) compute curvature and place at index i.
    Returns length-T array with NaN at first & last indices.
    """
    T, d = Y.shape
    if T < 3:
        return jnp.full((T,), jnp.nan)
    p0 = Y[:-2]   # (T-2, d)
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

# -------------------------
# Main wrapper
# -------------------------
def compute_curvature_jax(
    X: jnp.ndarray,
    dt: float = 1.0,
    method: str = "fd",                 # "fd" (finite-diff) or "tri" (triangle)
    smoothing: Optional[str] = None,    # None, "moving", or "gaussian"
    smoothing_param: Optional[float] = None,  # window (int) for moving, sigma for gaussian
    local_pca_window: Optional[int] = None,   # odd integer window size; if None or 0 => no PCA
    pca_k: int = 2,                     # dims to keep for PCA (2 recommended)
    speed_eps: float = 1e-10
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute curvature of trajectory X (T, N) using JAX.
    Returns (kappa, info) where kappa is length-T array (NaN where undefined),
    and info contains parameters and masks.
    """
    X = jnp.asarray(X)
    T, N = X.shape
    info = {
        "orig_shape": (T, N),
        "method": method,
        "smoothing": smoothing,
        "smoothing_param": smoothing_param,
        "local_pca_window": local_pca_window,
        "pca_k": pca_k,
    }

    # 1) smoothing
    Y = X
    if smoothing is not None:
        if smoothing == "moving":
            if smoothing_param is None:
                raise ValueError("smoothing_param (odd int) required for moving average")
            window = int(smoothing_param)
            if window % 2 == 0:
                raise ValueError("moving average window must be odd")
            Y = moving_average_smooth(Y, window)
        elif smoothing == "gaussian":
            if smoothing_param is None:
                raise ValueError("smoothing_param (sigma float) required for gaussian smoothing")
            sigma = float(smoothing_param)
            Y = gaussian_smooth(Y, sigma)
        else:
            raise ValueError(f"Unknown smoothing mode: {smoothing}")

    # 2) local PCA projection (optional)
    used_pca = False
    if local_pca_window is not None and local_pca_window > 0:
        if local_pca_window % 2 == 0:
            raise ValueError("local_pca_window must be odd")
        if pca_k < 1:
            raise ValueError("pca_k must be >= 1")
        Y = local_pca_project_center(Y, local_pca_window, pca_k)  # (T, pca_k)
        used_pca = True

    # 3) choose curvature estimator
    if method == "fd":
        # finite-difference derivative-based
        kappa = derivative_based_curvature(Y, dt=dt, speed_eps=speed_eps)
    elif method == "tri":
        kappa = triangle_based_curvature(Y)
    else:
        raise ValueError("method must be 'fd' or 'tri'")

    info.update({
        "used_pca": used_pca,
        "projected_dim": Y.shape[1],
    })

    return kappa, info


'''
# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import numpy as onp
    # toy noisy helix in high-D (embed 3D helix into 50D random directions)
    key = jax.random.PRNGKey(0)
    T = 500
    t = jnp.linspace(0, 6 * jnp.pi, T)
    helix3 = jnp.stack([jnp.cos(t), jnp.sin(t), t / (6*jnp.pi)], axis=1)  # (T,3)
    # embed into 50 dims
    N = 50
    A = jax.random.normal(key, (3, N))
    X = helix3 @ A + 0.02 * jax.random.normal(key, (T, N))
    # compute curvature: gaussian smoothing sigma=1.5, local PCA window 31, keep 3 dims
    kappa, info = compute_curvature_jax(
        X,
        dt = float(t[1] - t[0]),
        method="fd",
        smoothing="gaussian",
        smoothing_param=1.5,
        local_pca_window=31,
        pca_k=3,
    )
    print("info:", info)
    print("kappa shape:", kappa.shape)
'''