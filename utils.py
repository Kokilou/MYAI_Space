import os
import numpy as np
import torch

def check_file_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
        
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False