import argparse
import os
import re
import sys
import json
import math
import torch
import numpy as np
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampling")
    parser.add_argument('--tag',             type=str,   default='tag')
    parser.add_argument('--model',           type=str,   default='SANA')
    parser.add_argument('--solver',          type=str,   default='DPM-Solver')
    parser.add_argument('--algorithm_type',  type=str,   default='data_prediction')
    parser.add_argument('--skip_type',       type=str,   default='time_uniform')
    parser.add_argument('--flow_shift',      type=float, default=1.0)
    parser.add_argument('--NFE',             type=int,   default=10)
    parser.add_argument('--CFG',             type=float, default=4.5)
    parser.add_argument('--order',           type=int,   default=2)
    parser.add_argument('--data',            type=str,   default='MSCOCO2017')
    parser.add_argument('--save_root',       type=str,   default='/data/scpark/samplings/')
    parser.add_argument('--n_samples',       type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=5)
    parser.add_argument('--output_noise',    action='store_true',  default=False)
    parser.add_argument('--output_traj',     action='store_true',  default=False)
    parser.add_argument('--seed_offset',     type=int,   default=0)
    return parser

def parse_args() -> EasyDict:
    parser = build_parser()
    args = parser.parse_args()
    return EasyDict(vars(args))

def get_model(config: EasyDict):
    if config.model == 'SANA':
        from backbones.sana import SANA
        return SANA()
    if config.model == 'PixArt-Sigma':
        from backbones.pixart_sigma import PixArtSigma
        return PixArtSigma()
    if config.model == 'DiT':
        from backbones.dit import DiT
        return DiT()
    if config.model == 'GMDiT':
        GMFLOW = os.path.join("submodules", "GMFlow")
        sys.path.insert(0, GMFLOW)
        from backbones.gmdit import GMDiT
        return GMDiT()
    raise ValueError(f"Unknown model: {config.model}")

def get_solver(config: EasyDict):
    if config.solver == 'Euler':
        from solvers.euler_solver import Euler_Solver
        return Euler_Solver
    if config.solver == 'DPM-Solver':
        from solvers.dpm_solver import DPM_Solver
        return DPM_Solver
    if config.solver == 'UniPC':
        from solvers.unipc_solver import UniPC_Solver
        return UniPC_Solver    
    raise ValueError(f"Unknown solver: {config.solver}")

def get_data(config: EasyDict):
    if config.data == 'MSCOCO2017':
        return np.load('prompts/mscoco2017.npz')['arr_0'].tolist()
    if config.data == 'ImageNet':
        return [i%1000 for i in range(config.n_samples)]
    raise ValueError(f"Unknown data: {config.data}")

def get_sampling_dir(config):
    p, r = config.tag, config.save_root
    os.makedirs(config.save_root, exist_ok=True)
    i = max([int(m.group(1))
             for d in os.listdir(r)
             if (m := re.match(rf'{re.escape(p)}_(\d+)$', d))]
            or [-1])
    sampling_dir = os.path.join(r, f"{p}_{i+1}")
    os.makedirs(sampling_dir, exist_ok=True)        
    return sampling_dir

def save_config(config):
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(dict(config), f, indent=2)

def main():
    config = parse_args()
    config.save_dir = get_sampling_dir(config)
    save_config(config)

    model  = get_model(config)
    Solver = get_solver(config)
    data   = get_data(config)

    n_iters = math.ceil(config.n_samples / config.batch_size)
    for start in tqdm(range(0, config.n_samples, config.batch_size),
                      total=n_iters, desc="Sampling batches"):
        end = min(start + config.batch_size, config.n_samples)
        conds = data[start:end]
        seeds = config.seed_offset + np.arange(start, end, dtype=int)

        noise_schedule = model.get_noise_schedule()
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        noises = model.get_noise(seeds=seeds)
        solver = Solver(noise_schedule, config.NFE, order=config.order, skip_type=config.skip_type, flow_shift=config.flow_shift, algorithm_type=config.algorithm_type)

        outputs = solver.sample(noises, model_fn, output_traj=config.output_traj)
        samples = outputs['samples'].detach().cpu()
        if config.output_noise:
            noises = noises.detach().cpu()
        if config.output_traj:
            trajs = outputs['trajs'].detach().cpu()    

        for index in range(start, end):
            output = {'sample': samples[index-start],
                      'cond': conds[index-start]
                      }
            if config.output_noise:
                output['noise'] = noises[index-start]
            if config.output_traj:
                output['traj'] = trajs[index-start]
            torch.save(output, os.path.join(config.save_dir, f"{index}.pt"))

if __name__ == '__main__':
    main()