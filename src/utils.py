import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set seed across python random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, name, lr, momentum=0.9, weight_decay=5e-4):
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay)
    elif name == 'nesterov':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'nadam':
        return torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
