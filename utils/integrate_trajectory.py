import diffrax
import jax.numpy as jnp
from jax import grad, random, config, vmap, jit

def integrate_trajectory(model, x, dt, t1, ts):
    """
    Be careful with saveat=diffrax.SaveAt(ts=ts) in the code below
    It returns the same shape of energy as input t

    """
    x = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        diffrax.Dopri5(),
        t0=0,
        t1=t1,
        dt0=dt,
        y0=x,
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
        saveat=diffrax.SaveAt(ts=ts),   # Would return energy in all ts given
        args=None
    ).ys
    return x

def get_energy(model, X_batch, y_batch, dt, t1, samples):
    N_steps = int(t1 / dt)
    ts = jnp.linspace(0.0, t1, N_steps + 1)

    X = X_batch[:samples]
    y = y_batch[:samples]

    batch_trajs = vmap(integrate_trajectory, in_axes=(None, 0, None, None, None))(
        model, X, dt, t1, ts
        )

    batch_E = vmap(lambda traj: vmap(model.energy)(traj))(batch_trajs)
    return X, y, ts, batch_E