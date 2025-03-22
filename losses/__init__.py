from .loss1 import loss1
from .loss2 import loss2

def get_loss(name):
    losses = {
        "loss1": loss1,
        "loss2": loss2,
    }
    return losses.get(name, loss1)  # Default to loss1
