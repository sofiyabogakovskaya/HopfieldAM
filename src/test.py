import jax.numpy as jnp
from utils.metrics import batch_accuracy

def test(model, test_loader, dt, t1, N_classes):
    """Runs the final evaluation on the test dataset."""
    acc = batch_accuracy(model, test_loader, dt, t1, N_classes)
    print(f"Test Accuracy: {acc:.2%}")
    return acc
