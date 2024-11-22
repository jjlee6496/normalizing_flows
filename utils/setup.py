import torch
import random
import numpy as np

def setup_experiment(config):
    """Setup experiment with proper seed and device"""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    if torch.cuda.is_available() and config.device == "cuda":
        torch.cuda.manual_seed(config.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device