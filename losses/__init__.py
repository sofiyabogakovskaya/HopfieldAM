from .loss_last10 import batch_loss_last10


def get_batch_loss(name):
    losses = {
        "batch_loss_last10": batch_loss_last10
    }
    return losses.get(name, batch_loss_last10)  
