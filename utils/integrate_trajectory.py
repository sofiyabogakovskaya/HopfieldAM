import diffrax
from jax import grad, random, config, vmap, jit

def integrate_trajectory(model, x, dt, t1, ts):
    x = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        diffrax.Dopri5(),
        t0=0,
        t1=t1,
        dt0=dt,
        y0=x,
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
        saveat=diffrax.SaveAt(ts=ts),
        args=None
    ).ys
    return x
