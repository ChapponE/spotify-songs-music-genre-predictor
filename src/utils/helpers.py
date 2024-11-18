# src/utils/helpers.py

import torch.optim as optim
import torch.nn as nn

def get_optimizer(optimizer_name, parameters, learning_rate):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(parameters, lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' n'est pas supporté.")

def get_loss_function(loss_name):
    loss_name = loss_name.lower()
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function '{loss_name}' n'est pas supportée.")

# src/utils/helpers.py
import inspect
import torch.nn as nn

def is_neural_network(model_class):
    try:
        model = model_class()
        return model.neural_network
    except:
        return False