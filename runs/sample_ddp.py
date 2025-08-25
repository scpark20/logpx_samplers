#!/usr/bin/env python
import argparse, os, re, sys, json, math, torch, numpy as np
from datetime import timedelta
from easydict import EasyDict
from tqdm import tqdm
import torch.distributed as dist

from utils.inception import FIDInception
from utils.clip import CLIPEmbedder

# ---------------------- args ----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sampling (multi-GPU, torchrun)")
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
    p.add_argument('--batch_size',      type=int,   default=5)      # per-rank
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
    """Init torch.distributed. Returns (rank, world, local_rank)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        rank  = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        torch.cuda.set_device(local)
        backend = os.environ.get("DIST_BACKEND", "nccl")
        timeout_s = int(os.environ.get("DIST_TIMEOUT", "300"))
        dist.init_process_group(
            backend=backend, init_method="env://",
            timeout=timedelta(seconds=timeout_s),
            device_id=local,                    # ✅ 경고/행 위험 제거
        )
        return rank, world, local
    return 0, 1, 0

def barrier(local_rank: int):
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

def bcast_obj(obj, src=0):
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=src)
    return container[0]

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
        import numpy as np
        return np.load('prompts/mscoco2017.npz')['arr_0'].tolist()
    if config.data == 'ImageNet':
        return [i % 1000 for i in range(config.n_samples)]
    raise ValueError(f"Unknown data: {config.data}")

def get_sampling_dir(config, rank=0):
    tag, root = config.tag, config.save_root
    os.makedirs(root, exist_ok=True)
    # 다음 인덱스 결정
    i = max([int(m.group(1)) for d in os.listdir(root)
             if (m := re.match(rf'{re.escape(tag)}_(\d+)$', d))] or [-1]) + 1
    path = os.path.join(root, f"{tag}_{i}")
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    return path

def save_config(config):
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(dict(config), f, indent=2)

# ---------------------- main ----------------------
def main():
    config = parse_args()
    rank, world, local = init_dist()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    # rank0만 결정/생성 후 브로드캐스트
    save_dir = get_sampling_dir(config, rank=rank) if rank == 0 else None
    save_dir = bcast_obj(save_dir, src=0)
    config.save_dir = save_dir
    if rank == 0:
        save_config(config)
    barrier(local)

    # 컴포넌트
    model  = get_model(config)
    Solver = get_solver(config)
    data   = get_data(config)

    inception = FIDInception().to(device) if config.inception else None
    clip      = CLIPEmbedder(device=device) if config.clip else None

    # 샤딩 (글로벌 인덱스 유지)
    all_idx = list(range(config.n_samples))
    my_idx  = all_idx[rank::world]  # 각 랭크가 전역 인덱스 일부 담당

    n_iters = math.ceil(len(my_idx) / config.batch_size)
    pbar = tqdm(total=n_iters, desc=f"rank{rank}", disable=(rank != 0))

    torch.backends.cudnn.benchmark = True
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        ptr = 0
        while ptr < len(my_idx):
            batch_indices = my_idx[ptr: ptr + config.batch_size]
            ptr += config.batch_size

            conds = [data[i] for i in batch_indices]
            seeds = config.seed_offset + np.asarray(batch_indices, dtype=int)  # ✅ seed = offset + global_idx

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
            trajs_t     = outputs.get('trajs', None)
            trajs_cpu   = trajs_t.detach().cpu() if (trajs_t is not None and config.output_traj) else None

            for j, gidx in enumerate(batch_indices):
                out = {'cond': conds[j]}
                if config.sample:       out['sample'] = samples_cpu[j]
                if config.inception:    out['inception_feature'] = inc_feats[j]
                if config.clip:         out['clip_feature'] = clip_feats[j]
                if config.output_noise: out['noise'] = noises_cpu[j]
                if config.output_traj:  out['traj'] = trajs_cpu[j]
                torch.save(out, os.path.join(config.save_dir, f"{gidx}.pt"))

            pbar.update(1)

    pbar.close()
    barrier(local)
    if rank == 0:
        print(f"Done. Saved to: {config.save_dir}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
