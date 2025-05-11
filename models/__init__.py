from .Hopfield import Hopfield

def get_model(model_name, key, **kwargs):
    """Factory function to initialize models with custom parameters."""
    if model_name == "Hopfield":
        return Hopfield(key=key, **kwargs)  
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

# def get_model(name):
#     models = {
#         "Hopfield": Hopfield()
#         # "model2": Model2(),
#     }
#     return models.get(name, Hopfield()) 
