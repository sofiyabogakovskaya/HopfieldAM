import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax

from sklearn.model_selection import train_test_split

from jax import grad, random, config, vmap, jit
from jax.nn import relu, gelu, sigmoid, softmax, log_softmax
from numpy import genfromtxt

from src.Hopfield_model import Hopfield
from src.data_prep import load_data, split_data, split_in_batches

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

def loss_fn(model, x, y, dt, N_steps, N_classes):
    # last ten elements loss
    x_T = integrate(model, x, dt, N_steps)[-N_classes:]
    log_proba = log_softmax(x_T)
    return -log_proba[y]

def loss_fn1(model, x, y, dt, N_steps, N_classes):
    # linear layer loss
    key5 = random.PRNGKey(19)
    classification_layer = eqx.nn.Linear(64, 10, key5)
    x_T = integrate(model, x, dt, N_steps)
    classes = classification_layer(x_T)
    log_proba = log_softmax(classes)
    return -log_proba[y]

def batch_loss(model, x, y, dt, N_steps, N_classes):
    return jnp.mean(vmap(loss_fn1, in_axes=(None, 0, 0, None, None, None))(model, x, y, dt, N_steps, N_classes))

def accuracy(model, x, y, dt, N_steps, N_classes):
    x_T = integrate(model, x, dt, N_steps)[-N_classes:]
    log_proba = log_softmax(x_T)
    return jnp.argmax(log_proba) == y

@eqx.filter_jit
def update(model, x, y, opt_state, optimizer, dt, N_steps, N_classes):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y, dt, N_steps, N_classes)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss



def train(epochs=1000, learning_rate=1e-3, batch_size=64, dt=1e-2, N_steps=100, N_classes=10):
    key = random.PRNGKey(33)
    features, targets = load_data()
    # TODO: batching for test data, because it can be big and not fit into GPU
    # also make batching with dataloader object for test
    X_train, X_test, y_train, y_test = split_data(features, targets)
    data_batches = split_in_batches(X_train, y_train, batch_size)
    # print(data_batches[0][0].shape[1])
    model = Hopfield(data_batches[0][0].shape[1], key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    accs = []

    for epoch in range(epochs):
        for x, y in data_batches:
            model, opt_state, loss = update(model, x, y, opt_state, optimizer, dt, N_steps, N_classes)
        # acc = jnp.mean(vmap(accuracy, in_axes=(None, 0, 0, None, None, None))(model, x, y, dt, N_steps, N_classes))
        acc = jnp.mean(
              vmap(accuracy, in_axes=(None, 0, 0, None, None, None))(model, X_test, y_test, dt, N_steps, N_classes)
            )
        # TODO: maybe put vmaped accuracy in a separate function?

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        accs.append(acc)

    return model, data_batches, accs