import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def extract_log_deltas(pt_file):
    log_deltas = torch.load(pt_file)['solver_state_dict']['log_deltas'].data.cpu()
    return log_deltas

def get_timesteps(log_deltas, t0=1.0, tM=0.0):
    deltas = F.softmax(log_deltas, dim=0)          # (S,)
    c = torch.cumsum(deltas, dim=0)                # (S,)
    timesteps = torch.cat([t0*torch.ones(1, device=log_deltas.device), t0 + (tM - t0) * c], dim=0)
    return timesteps  

def invert_timesteps(timesteps, t0=1.0, tM=0.0):
    c = (timesteps[1:] - t0) / (tM - t0)
    deltas = torch.cat([c[:1], c[1:] - c[:-1]])
    deltas = deltas / deltas.sum()
    log_deltas = deltas.log()
    return log_deltas    

def interp(x, size):
    return F.interpolate(x.view(1, 1, -1), size=size, mode='linear', align_corners=True).view(-1)

def interp_weighted_average(x, y, size):
    x_len = len(x)
    y_len = len(y)
    x_weight = (y_len - size) / (y_len - x_len)
    y_weight = (size - x_len) / (y_len - x_len)
    return x_weight*interp(x, size) + y_weight*interp(y, size)
    
def get_interplated_log_deltas(n_steps, pt_file1, pt_file2):
    log_deltas1 = extract_log_deltas(pt_file1)
    timesteps1 = get_timesteps(log_deltas1)

    log_deltas2 = extract_log_deltas(pt_file2)
    timesteps2 = get_timesteps(log_deltas2)

    timesteps3 = interp_weighted_average(timesteps1, timesteps2, n_steps+1)
    return invert_timesteps(timesteps3)

def _resize_weight(weight, n_steps):
    """
    weight: (S, 2, 5)
    n_steps: 목표 step 수
    return: (n_steps, 2, 5)
    """
    S, C1, C2 = weight.shape          # C1=2, C2=5
    x = weight.reshape(S, -1).T       # (C, S)
    x = x.unsqueeze(0)                # (1, C, S)
    x = F.interpolate(x, size=n_steps, mode='linear', align_corners=True)
    x = x.squeeze(0).T.reshape(n_steps, C1, C2)
    return x

def get_interpolated_weight(n_steps, weight1, weight2):
    len1 = len(weight1)
    len2 = len(weight2)
    c1 = (len2 - n_steps) / (len2 - len1)
    c2 = (n_steps - len1) / (len2 - len1)
    w1 = _resize_weight(weight1, n_steps)   # s3 → n_steps
    w2 = _resize_weight(weight2, n_steps)   # s5 → n_steps
    return c1*w1 + c2*w2

def extract_weight(pt_file):
    weight = torch.load(pt_file)['solver_state_dict']['param_extractor.table'].data.cpu()
    return weight    