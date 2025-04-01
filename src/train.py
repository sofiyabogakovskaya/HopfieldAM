import jax.numpy as jnp
import equinox as eqx
import time

from jax import grad, random, config, vmap, jit

from tqdm import tqdm

from utils.logger import log_message, log_metrics, clear_logs
from utils.metrics import batch_accuracy

@eqx.filter_jit
def update(model, x, y, batch_loss, opt_state, optimizer, dt, N_steps, N_classes):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y, dt, N_steps, N_classes)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train(model, train_loader, val_loader, batch_loss, optimizer, opt_state, epochs, dt, N_steps, N_classes):
    clear_logs()
    avg_loss = 1
    val_acc = 0.01 
    for epoch in tqdm(range(epochs), desc="epoch training..."):
        # total_loss = 0.0
        # num_batches = 0
        # for x_batch, y_batch in train_loader():
        #     model, opt_state, loss_value = update(model, x_batch, y_batch, batch_loss, opt_state, optimizer, dt, N_steps, N_classes)
        #     total_loss += loss_value
        #     num_batches += 1   
        # avg_loss = total_loss / num_batches
        avg_loss += 1
        val_acc += 0.02
        # val_acc = batch_accuracy(model, val_loader, dt, N_steps, N_classes)   

        # print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.2%}")
        log_message(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.2%}")
        log_metrics({"epoch": epoch + 1, "loss": float(avg_loss), "val_accuracy": float(val_acc)})

    return model
