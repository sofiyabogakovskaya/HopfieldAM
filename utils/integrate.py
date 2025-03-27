import diffrax

def integrate(model, x, dt, N_steps):
    x = diffrax.diffeqsolve(
      diffrax.ODETerm(model),
      diffrax.Dopri5(), #Runge-Kutta adaptive solver (Dormand-Prince 5th order method)
      t0=0,
      t1=1,
      dt0=dt,
      y0=x,
      stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
      args=None
    ).ys[-1]
    return x