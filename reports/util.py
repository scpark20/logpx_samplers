import os
import json
from easydict import EasyDict

import torch
import numpy as np
import torch.nn.functional as F


# Load a config.json file and return it as an EasyDict
def load_config(path):
    with open(os.path.join(path, 'config.json'), 'r') as f:
        return EasyDict(json.load(f))

# Check if two config objects match for all specified keys
def match(c1, c2, keys):
    return all(c1[k] == c2[k] for k in keys)

def write_rmse(ref_dir, sam_dir):
    for i in range(10000):
        out_path = os.path.join(sam_dir, f"{i}_rmse.txt")
        if os.path.exists(out_path):
            continue

        f1 = os.path.join(ref_dir, f"{i}.pt")
        f2 = os.path.join(sam_dir, f"{i}.pt")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            break
        d1 = torch.load(f1, weights_only=True)
        d2 = torch.load(f2, weights_only=True)
        rmse = torch.sqrt(F.mse_loss(d1, d2)).item()

        with open(out_path, 'w') as wf:
            print(ref_dir, sam_dir, i)
            wf.write(f"{rmse}")
    return None

def read_rmse(sam_dir):
    rmses = []
    for i in range(10000):
        out_path = os.path.join(sam_dir, f"{i}_rmse.txt")
        if not os.path.exists(out_path):
            break

        with open(out_path, 'r') as rf:
            rmse = float(rf.readline())
            rmses.append(rmse)
    if len(rmses) == 0:
        return np.nan
    return np.mean(rmses)

def find_config(configs, ref):
    ret_configs = []
    for config in configs:
        if all(config[key] == ref[key] for key in ref.keys()):
            ret_configs.append(config)
    return ret_configs
            

