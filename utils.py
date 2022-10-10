import yaml
import random
import os
import torch
import numpy as np

def load_config(config_file_path: str):
    with open(config_file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True