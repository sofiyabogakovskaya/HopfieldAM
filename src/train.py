import jax.numpy as jnp
import equinox as eqx
import time

from jax import grad, random, config, vmap, jit

from tqdm import tqdm

from config import CONFIG
from utils.logger import log_message, log_metrics, clear_logs
from utils.logger import log_experiment, new_run_id
from utils.metrics import batch_accuracy

@eqx.filter_jit
def update(model, x, y, batch_loss, opt_state, optimizer, dt, t1, N_classes):
    loss, grads = eqx.filter_value_and_grad(batch_loss)(model, x, y, dt, t1, N_classes)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train(model, train_loader, val_loader, batch_loss, optimizer, opt_state, epochs, dt, t1, N_classes):
    clear_logs()
    run_id = new_run_id()
    train_losses = []
    train_accuracies = []
    for epoch in tqdm(range(epochs), desc="epoch training..."):
        total_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in train_loader():
            model, opt_state, loss_value = update(model, x_batch, y_batch, batch_loss, opt_state, optimizer, dt, t1, N_classes)
            total_loss += loss_value
            num_batches += 1   
        avg_loss = total_loss / num_batches
        val_acc = batch_accuracy(model, val_loader, dt, t1, N_classes)  

        train_losses.append(float(avg_loss))
        train_accuracies.append(float(val_acc)) 

        log_message(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Accuracy: {val_acc:.2%}")
        log_metrics({"epoch": epoch + 1, "loss": float(avg_loss), "val_accuracy": float(val_acc)})
    
    print(train_losses)
    print(train_accuracies)
    print(type(train_losses))
    print(type(train_losses[-1]))
    
    log_experiment(run_id, model, opt_state, CONFIG, {
                "train_loss": train_losses,
                "train_accuracy": train_accuracies
            })

    return model
