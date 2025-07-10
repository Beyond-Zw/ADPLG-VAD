import math
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype = np.uint16)
    return r

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)
    
def save_best_record(test_info, file_path):
    fo = open(file_path, "w")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("auc: {:.4f}\n".format(test_info["auc"][-1]))
    fo.write("ap: {:.4f}\n".format(test_info["ap"][-1]))

def max_min_one_dim_norm(x):
    # x [32,32]
    x_max = torch.max(x, dim=0)[0]
    x_min = torch.min(x, dim=0)[0]
    x_norm = (x - x_min) / (
            x_max - x_min + 1e-8)

    return x_norm

def max_min_two_dim_norm(x):
    # x [32,32]
    x_max = torch.max(x, dim=1)[0].unsqueeze(-1).repeat(1, 32)
    x_min = torch.min(x, dim=1)[0].unsqueeze(-1).repeat(1, 32)
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)

    return x_norm



