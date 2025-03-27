import jax.numpy as jnp
import equinox as eqx
import optax
import time


from jax import grad, random, config, vmap, jit

from tqdm import tqdm

from utils.logger import log_message, log_metrics
from utils.metrics import accuracy
# from losses.integrate import integrate


@eqx.filter_jit
def update(model, x, y, batch_loss, opt_state, optimizer, dt, N_steps, N_classes):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y, dt, N_steps, N_classes)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(model, train_loader, val_loader, batch_loss, optimizer, opt_state, epochs, dt, N_steps, N_classes):

    accs = []

    for epoch in tqdm(range(epochs), desc="training epoch..."):
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in train_loader(): 
            model, opt_state, loss_value = update(model, x_batch, y_batch, batch_loss, opt_state, optimizer, dt, N_steps, N_classes)
            total_loss += loss_value
            num_batches += 1

        avg_loss = total_loss / num_batches

        val_acc = accuracy(model, val_loader)
        accs.append(val_acc)

        log_message(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.2%}")
        log_metrics({"epoch": epoch + 1, "loss": float(avg_loss), "val_accuracy": float(val_acc)})

    return model, accs





# def train(epochs=1000, learning_rate=1e-3, batch_size=64, dt=1e-2, N_steps=100, N_classes=10):
#     key = random.PRNGKey(33)
#     features, targets = load_data()
#     # TODO: batching for test data, because it can be big and not fit into GPU
#     # also make batching with dataloader object for test
#     X_train, X_test, y_train, y_test = split_data(features, targets)
#     data_batches = split_in_batches(X_train, y_train, batch_size)
#     # print(data_batches[0][0].shape[1])
#     model = Hopfield(data_batches[0][0].shape[1], key)
#     optimizer = optax.adam(learning_rate)
#     opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
#     accs = []

#     for epoch in range(epochs):
#         for x, y in data_batches:
#             model, opt_state, loss = update(model, x, y, opt_state, optimizer, dt, N_steps, N_classes)
#         # acc = jnp.mean(vmap(accuracy, in_axes=(None, 0, 0, None, None, None))(model, x, y, dt, N_steps, N_classes))
#         acc = jnp.mean(
#               vmap(accuracy, in_axes=(None, 0, 0, None, None, None))(model, X_test, y_test, dt, N_steps, N_classes)
#             )
#         # TODO: maybe put vmaped accuracy in a separate function?

#         print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
#         accs.append(acc)

#     return model, data_batches, accs