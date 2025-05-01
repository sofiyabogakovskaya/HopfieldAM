import jax.numpy as jnp
from utils.metrics import batch_accuracy

def test(model, test_loader, batch_loss, dt, t1, N_classes):
    total_loss = 0.0
    num_batches = 0
    for x_batch, y_batch in test_loader():
        loss_value = batch_loss(model, x_batch, y_batch, dt, t1, N_classes)
        total_loss += loss_value
        num_batches += 1   
    test_loss = total_loss / num_batches
    test_loss = round(float(test_loss), 4)
    test_accuracy = batch_accuracy(model, test_loader, dt, t1, N_classes)
    test_accuracy = round(float(test_accuracy), 4)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy:.2%}")
    return test_loss, test_accuracy
