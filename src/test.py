import jax.numpy as jnp
from utils.metrics import batch_accuracy

def test(run_id, model, test_loader, dt, t1, N_classes):
    test_accuracy = batch_accuracy(model, test_loader, dt, t1, N_classes)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    return test_accuracy
