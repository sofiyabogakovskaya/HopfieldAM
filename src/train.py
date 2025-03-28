import jax.numpy as jnp
import equinox as eqx
import time

from jax import grad, random, config, vmap, jit

from tqdm import tqdm

from utils.logger import log_message, log_metrics
from utils.metrics import accuracy

@eqx.filter_jit
def update(model, x, y, batch_loss, opt_state, optimizer, dt, N_steps, N_classes):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y, dt, N_steps, N_classes)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train(model, train_loader, val_loader, batch_loss, optimizer, opt_state, epochs=1, dt, N_steps, N_classes):

    # accs = []

    # for epoch in range(epochs):
    #     total_loss = 0.0
    #     num_batches = 0
    #     for x_batch, y_batch in tqdm(train_loader(), desc="batch training..."):
    #         model, opt_state, loss_value = update(model, x_batch, y_batch, batch_loss, opt_state, optimizer, dt, N_steps, N_classes)
    #         total_loss += loss_value
    #         num_batches += 1

    #     avg_loss = total_loss / num_batches

    # val_acc = accuracy(model, val_loader, dt, N_steps, N_classes)
    # correct, total = 0, 0
    # for x_batch, y_batch in tqdm(val_loader(), desc="evaluating accuracy..."):
    #     print([len[a] for a in x_batch])
    #     acc = accuracy(model, x_batch, y_batch, dt, N_steps, N_classes)
    #     correct += jnp.sum(acc)
    #     total += len(y_batch)
    # val_acc = correct / total

          
    # accs.append(val_acc)

    # log_message(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.2%}")
    # log_metrics({"epoch": epoch + 1, "loss": float(avg_loss), "val_accuracy": float(val_acc)})

    return model
    # return model, accs
