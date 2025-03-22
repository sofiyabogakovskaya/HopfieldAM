from .model1 import Model1
from .model2 import Model2

def get_model(name):
    models = {
        "model1": Model1(),
        "model2": Model2(),
    }
    return models.get(name, Model1())  # Default to Model1
