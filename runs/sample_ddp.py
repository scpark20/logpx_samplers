#!/usr/bin/env python
import argparse, os, re, sys, json, math, torch, numpy as np
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
from utils.inception import FIDInception
from utils.clip import CLIPEmbedder

import torch.distributed as dist

# ---------------------- arg parsing ----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run sampling (multi-GPU friendly)")
    p.add_argument('--tag',             type=str,   default='tag')
    p.add_argument('--model',           type=str,   default='SANA')
    p.add_argument('--model_id',        type=str,   default=None)
    p.add_argument('--solver',          type=str,   default='DPM-Solver')
    p.add_argument('--algorithm_type',  type=str,   default='data_prediction')
    p.add_argument('--skip_type',       type=str,   default='time_uniform')
    p.add_argument('--flow_shift',      type=float, default=3.0)
    p.add_argument('--NFE',             type=int,   default=10)
    p.add_argument('--CFG',             type=float, default=4.5)
    p.add_argument('--order',           type=int,   default=2)
    p.add_argument('--data',            type=str,   default='MSCOCO2017')
    p.add_argument('--save_root',       type=str,   default='/data/scpark/samplings/')
    p.add_argument('--n_samples',       type=int,   default=100)
    p.add_argument('--batch_size',      type=int,   default=5)     # per-rank batch size
    p.add_argument('--output_noise',    action='store_true',  default=False)
    p.add_argument('--output_traj',     action='store_true',  default=False)
    p.add_argument('--inception',       action='store_true',  default=False)
    p.add_argument('--sample',          action='store_true',  default=False)
    p.add_argument('--clip',            action='store_true',  default=False)
    p.add_argument('--seed_offset',     type=int,   default=0)
    return p

def parse_args() -> EasyDict:
    return EasyDict(vars(build_parser().parse_args()))

# ---------------------- dist helpers ----------------------
def init_dist():
    """Init torch.distributed (NCCL). Returns (rank, world_size, local_rank)."""
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return rank, world_size, local_rank
    # single process
    return 0, 1, 0

def bcast_str(s: str, src: int = 0) -> str:
    if not dist.is_available() or not dist.is_initialized():
        return s
    obj = [s]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]

# ---------------------- user funcs ----------------------
def get_model(config: EasyDict):
    if config.model == 'SANA':
        from backbones.sana import SANA
        return SANA(model_id=config.model_id) if config.model_id is not None else SANA()
    if config.model == 'PixArt-Sigma':
        from backbones.pixart_sigma import PixArtSigma
        return PixArtSigma()
    if config.model == 'DiT':
        from backbones.dit import DiT
        return DiT()
    if config.model == 'GMDiT':
        GMFLOW = os.path.join("submodules", "GMFlow"); sys.path.insert(0, GMFLOW)
        from backbones.gmdit import GMDiT
        return GMDiT()
    raise ValueError(f"Unknown model: {config.model}")

def get_solver(config: EasyDict):
    if config.solver == 'Euler':
        from solvers.others.euler_solver import Euler_Solver
        return Euler_Solver
    if config.solver == 'DPM-Solver':
        from solvers.others.dpm_solver import DPM_Solver
        return DPM_Solver
    if config.solver == 'UniPC':
        from solvers.others.unipc_solver import UniPC_Solver
        return UniPC_Solver
    raise ValueError(f"Unknown solver: {config.solver}")

def get_data(config: EasyDict):
    if config.data == 'MSCOCO2017':
        return np.load('prompts/mscoco2017.npz')['arr_0'].tolist()
    if config.data == 'ImageNet':
        return [i % 1000 for i in range(config.n_samples)]
    raise ValueError(f"Unknown data: {config.data}")

def get_sampling_dir(config, rank=0):
    p, r = config.tag, config.save_root
    os.makedirs(r, exist_ok=True)
    i = max([int(m.group(1)) for d in os.listdir(r)
             if (m := re.match(rf'{re.escape(p)}_(\d+)$', d))] or [-1])
    sampling_dir = os.path.join(r, f"{p}_{i+1}")
    if rank == 0:
        os.makedirs(sampling_dir, exist_ok=True)
    return sampling_dir

def save_config(config):
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(dict(config), f, indent=2)

# ---------------------- main ----------------------
def main():
    config = parse_args()

    rank, world_size, local_rank = init_dist()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # rank0 decides save_dir, broadcast to others
    save_dir = get_sampling_dir(config, rank=rank)
    config.save_dir = bcast_str(save_dir, src=0)
    if rank == 0:
        save_config(config)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # build per-rank components
    model  = get_model(config)           # assume model internally moves to correct device in its methods
    Solver = get_solver(config)
    data   = get_data(config)

    # Optional feature extractors per-rank
    inception = FIDInception().to(device) if config.inception else None
    clip      = CLIPEmbedder(device=device) if config.clip else None

    # shard indices across ranks (global index -> file name "{index}.pt")
    all_idx   = list(range(config.n_samples))
    my_idx    = all_idx[rank::world_size]

    # simple per-rank progress
    n_iters = math.ceil(len(my_idx) / config.batch_size)
    pbar = tqdm(total=n_iters, desc=f"Rank{rank} sampling", disable=(rank != 0))

    torch.backends.cudnn.benchmark = True
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        k = 0
        while k < len(my_idx):
            batch_indices = my_idx[k : k + config.batch_size]
            k += config.batch_size

            conds = [data[i] for i in batch_indices]
            seeds = config.seed_offset + np.asarray(batch_indices, dtype=int)

            noise_schedule = model.get_noise_schedule()
            model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
            noises = model.get_noise(seeds=seeds)
            solver = Solver(noise_schedule, config.NFE, order=config.order,
                            skip_type=config.skip_type, flow_shift=config.flow_shift,
                            algorithm_type=config.algorithm_type)

            outputs = solver.sample(noises, model_fn, output_traj=config.output_traj)
            samples = outputs['samples']
            if config.inception or config.clip:
                raw_outputs = model.decode_vae(samples, raw_output=True)

            if config.inception:
                inc_feats = inception(raw_outputs, clamp_mode="hard").detach().cpu()
            if config.clip:
                clip_feats = clip.encode_image(raw_outputs).detach().cpu()

            samples_cpu = samples.detach().cpu() if config.sample else None
            noises_cpu  = noises.detach().cpu()  if config.output_noise else None
            trajs_cpu   = outputs['trajs'].detach().cpu() if config.output_traj else None

            # save per global index
            for j, index in enumerate(batch_indices):
                out = {'cond': conds[j]}
                if config.sample:       out['sample'] = samples_cpu[j]
                if config.inception:    out['inception_feature'] = inc_feats[j]
                if config.clip:         out['clip_feature'] = clip_feats[j]
                if config.output_noise: out['noise'] = noises_cpu[j]
                if config.output_traj:  out['traj'] = trajs_cpu[j]
                torch.save(out, os.path.join(config.save_dir, f"{index}.pt"))

            pbar.update(1)

    pbar.close()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print(f"Done. Saved to: {config.save_dir}")

if __name__ == '__main__':
    main()
