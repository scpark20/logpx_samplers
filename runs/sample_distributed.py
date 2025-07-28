#DPM-Solver의 torch.distributed와 torch.multiprocessing을 사용하여 분산 샘플링을 수행하는 코드

import argparse
import os
import sys
import math
import torch
import numpy as np
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

from .sample import get_sampling_dir, get_solver, get_data

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
    parser.add_argument("--port",            type=str, default="12355")
    return parser

def parse_args() -> EasyDict:
    parser = build_parser()
    args = parser.parse_args()
    return EasyDict(vars(args))

def get_model(config: EasyDict, rank: int):
    if config.model == 'SANA':
        from backbones.sana import SANA
        return SANA(device=f"cuda:{rank}")
    if config.model == 'PixArt-Sigma':
        from backbones.pixart_sigma import PixArtSigma
        return PixArtSigma(device=f"cuda:{rank}")
    if config.model == 'DiT':
        from backbones.dit import DiT
        return DiT(device=f"cuda:{rank}")
    if config.model == 'GMDiT':
        GMFLOW = os.path.join("submodules", "GMFlow")
        sys.path.insert(0, GMFLOW)
        from backbones.gmdit import GMDiT
        return GMDiT(device=f"cuda:{rank}")
    raise ValueError(f"Unknown model: {config.model}")

def get_data_range(rank, world_size, total_samples):
        samples_per_process = total_samples // world_size
        remainder = total_samples % world_size
        
        # 나머지를 앞쪽 프로세스들에 분배
        if rank < remainder:
            start_idx = rank * (samples_per_process + 1)
            end_idx = start_idx + samples_per_process + 1
        else:
            start_idx = rank * samples_per_process + remainder
            end_idx = start_idx + samples_per_process
        
        return start_idx, end_idx

def main():
    config = parse_args()
    config.save_dir = get_sampling_dir(config)
    os.makedirs(config.save_dir, exist_ok=True)
    
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    mp.spawn(sample, nprocs=world_size, args=(world_size, config))
    

def sample(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.port

    print(f"Rank {rank} initialized")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model  = get_model(config, rank)
    Solver = get_solver(config)
    data   = get_data(config)

    
    start_idx, end_idx = get_data_range(rank, world_size, config.n_samples)
    process_samples = end_idx - start_idx
    n_rounds = math.ceil(process_samples / config.batch_size)

    print(f"Rank {rank}: processing samples {start_idx}~{end_idx-1} ({process_samples} samples, {n_rounds} rounds)")
    for round in tqdm(range(n_rounds), desc=f"Rank {rank} Sampling"):
        batch_start = round * config.batch_size
        batch_end = min(batch_start + config.batch_size, process_samples)
        
        global_start = start_idx + batch_start
        global_end = start_idx + batch_end
        
        conds = data[global_start:global_end]
        seeds = list(range(global_start, global_end))
        
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
        
        # 저장할 때도 글로벌 인덱스 사용
        for i, global_idx in enumerate(range(global_start, global_end)):
            torch.save(samples[i], config.save_dir / f"{global_idx}.pt")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()