import argparse
import os
import sys
import math
import torch
import numpy as np
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampling")
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
    parser.add_argument('--n_samples',       type=int,   default=1000)
    parser.add_argument('--batch_size',      type=int,   default=5)
    return parser

def parse_args() -> EasyDict:
    parser = build_parser()
    args = parser.parse_args()
    return EasyDict(vars(args))

def get_sampling_dir(config: EasyDict) -> Path:
    parts = [
        config.data,
        config.solver,
        config.algorithm_type,
        config.skip_type,
        f"FS{config.flow_shift}",
        f"NFE{config.NFE}",
        f"CFG{config.CFG}",
        f"ORDER{config.order}",
    ]
    name = config.model + "".join(f"({p})" for p in parts)
    return Path(config.save_root) / name

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
    if config.data == 'Imagenet':
        return [i%1000 for i in range(10000)]
    raise ValueError(f"Unknown data: {config.data}")

def main():
    config = parse_args()
    config.save_dir = get_sampling_dir(config)
    os.makedirs(config.save_dir, exist_ok=True)

    model  = get_model(config)
    Solver = get_solver(config)
    data   = get_data(config)

    n_iters = math.ceil(config.n_samples / config.batch_size)
    for start in tqdm(range(0, config.n_samples, config.batch_size),
                      total=n_iters, desc="Sampling batches"):
        end = min(start + config.batch_size, config.n_samples)
        conds = data[start:end]
        seeds = list(range(start, end))

        model_fn, noise_schedule, latents = model.get_model_fn(
            pos_conds=conds,
            guidance_scale=config.CFG,
            seeds=seeds
        )
        solver = Solver(
            model_fn,
            noise_schedule,
            algorithm_type=config.algorithm_type
        )
        samples = solver.sample(
            latents,
            steps=config.NFE,
            order=config.order,
            skip_type=config.skip_type,
            flow_shift=config.flow_shift
        ).data.cpu()

        for index in range(start, end):
            torch.save(samples[index-start], config.save_dir / f"{index}.pt")

if __name__ == '__main__':
    main()