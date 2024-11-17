from src.models.base_model import BaseModel

def is_neural_network(model_class):
    return issubclass(model_class, BaseModel) and hasattr(model_class, 'train') and hasattr(model_class, 'predict')