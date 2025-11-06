from typing import Optional, Tuple
import math
import os

from jax import jit
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Literal, Optional
Method = Literal["minmax", "zscore"]


@jit
def _jax_min_max(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return (min, max) computed via JAX. Useful for consistent dtype/acceleration."""
    return jnp.min(x), jnp.max(x)


def normalize_per_step(X, method: Method = "minmax", eps: float = 1e-8):
    """
    X: jax array of shape (T, D)
    method: 'minmax' or 'zscore'
    returns jax array of same shape and dtype (float)
    """
    X = jnp.asarray(X, dtype=jnp.float32)
    if method == "minmax":
        mins = jnp.min(X, axis=1, keepdims=True)
        maxs = jnp.max(X, axis=1, keepdims=True)
        denom = maxs - mins
        # avoid div by zero for constant frames
        denom = jnp.where(denom < eps, 1.0, denom)
        Xn = (X - mins) / denom
        return Xn
    elif method == "zscore":
        mu = jnp.mean(X, axis=1, keepdims=True)
        sigma = jnp.std(X, axis=1, keepdims=True)
        sigma = jnp.where(sigma < eps, 1.0, sigma)
        Xn = (X - mu) / sigma
        return Xn
    else:
        raise ValueError(f"Unknown method: {method}")


def create_gif(
    X,
    out_path: str = "mnist_evolution.gif",
    fps: int = 20,
    cmap: str = "gray",
    save_colors = True,
    sample_stride: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (4, 4),
    dpi: int = 80,
    resize: Optional[Tuple[int, int]] = None,
):
    """
    Make and save a GIF showing how a flattened image evolves over time.

    Parameters
    ----------
    X : array-like
        Input array with shape (T, D) where D is a perfect square (e.g. 784).
        Accepts numpy arrays or JAX arrays.
    out_path : str
        Output filename (should end with .gif for Pillow writer).
    fps : int
        Frames per second for the saved GIF.
    cmap : str
        Matplotlib colormap to use (default: 'gray').
    sample_stride : Optional[int]
        If provided, use X[::sample_stride] frames only (useful to speed-up long sequences).
    vmin, vmax : Optional[float]
        Value range for imshow. If None they are computed from the data using JAX.
    figsize : tuple
        Figure size in inches passed to plt.figure().
    dpi : int
        DPI for saving the figure.
    resize : Optional[tuple]
        If provided, reshape frames to (resize[0], resize[1]) after inference (useful to scale up).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object (also saved to `out_path`).
    """

    # Convert to numpy for matplotlib, but allow JAX arrays as input
    is_jax = hasattr(X, "device_buffer") or isinstance(X, jnp.ndarray)
    if is_jax:
        X_np = np.array(X)
    else:
        X_np = np.asarray(X)

    if X_np.ndim != 2:
        raise ValueError(f"Expected X with shape (T, D), got shape {X_np.shape}")

    T, D = X_np.shape
    side = int(math.isqrt(D))
    if side * side != D:
        raise ValueError(f"Expected D to be a perfect square (like 784), got D={D}")
    
    if save_colors == True:
        x_0 = X_np[0]
        x_0_norm = jnp.linalg.norm(x_0)
        X_np_norm = normalize_per_step(jnp.array(X_np), method="minmax")   # each frame scaled to [0,1]
        X_np_scaled = X_np_norm * x_0_norm
        X_np = X_np_scaled

    frames = X_np.reshape(T, side, side)

    if sample_stride is not None and sample_stride > 1:
        frames = frames[::sample_stride]

    # use JAX to compute min/max if not provided
    if vmin is None or vmax is None:
        frames_jax = jnp.array(frames)
        jmin, jmax = _jax_min_max(frames_jax)
        vmin = float(jmin) if vmin is None else vmin
        vmax = float(jmax) if vmax is None else vmax

    # optional resize using numpy (fast and simple)
    if resize is not None:
        import PIL.Image as PilImage

        new_h, new_w = resize
        resized = np.empty((frames.shape[0], new_h, new_w), dtype=frames.dtype)
        for i in range(frames.shape[0]):
            img = PilImage.fromarray(frames[i])
            img = img.resize((new_w, new_h), resample=PilImage.NEAREST)
            resized[i] = np.asarray(img)
        frames = resized
        side = new_h

    # Matplotlib animation
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    ax.set_axis_off()

    im = ax.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    def _update(i):
        im.set_data(frames[i])
        return (im,)

    anim = animation.FuncAnimation(fig, _update, frames=frames.shape[0], interval=1000 / fps, blit=True)

    # ensure output dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save as GIF using PillowWriter
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)

    plt.close(fig)

    print(f"Saved GIF to: {out_path} (frames={frames.shape[0]}, size={side}x{side})")
    return anim
