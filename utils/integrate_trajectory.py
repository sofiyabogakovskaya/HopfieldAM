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

def get_energy(model, data_batches, dt, t1, ts):
    # TODO: fix data_batches thing, we only have a dataloader
    batch_trajectory = vmap(integrate_trajectory, in_axes=(None, 0, None, None, None))(model, data_batches[0][0], dt, t1, ts)
    E = vmap(vmap(model.energy))(batch_trajectory)
    return E