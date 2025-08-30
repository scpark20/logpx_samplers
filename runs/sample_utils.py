#!/usr/bin/env python
import argparse, os, re, sys, json, math, torch, numpy as np
from easydict import EasyDict
from tqdm import tqdm
from utils.fid import FIDInception
from utils.clean_fid import CleanFIDInception
from utils.clip import CLIPEmbedder
import torch.distributed as dist
from datetime import timedelta

# ---------------------- arg parsing ----------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampling (DDP)")
    parser.add_argument('--tag',             type=str,   default='tag')
    parser.add_argument('--model',           type=str,   default='SANA')
    parser.add_argument('--model_id',        type=str,   default=None)
    parser.add_argument('--solver',          type=str,   default='DPM-Solver')
    parser.add_argument('--algorithm_type',  type=str,   default='data_prediction')
    parser.add_argument('--skip_type',       type=str,   default='time_uniform')
    parser.add_argument('--flow_shift',      type=float, default=3.0)
    parser.add_argument('--NFE',             type=int,   default=10)
    parser.add_argument('--CFG',             type=float, default=4.5)
    parser.add_argument('--k',               type=float, default=0.5)
    parser.add_argument('--order',           type=int,   default=2)
    parser.add_argument('--data',            type=str,   default='MSCOCO2017')
    parser.add_argument('--save_root',       type=str,   default='/data/scpark/samplings/')
    parser.add_argument('--pt_dir',          type=str,   default=None)
    parser.add_argument('--pt_criterion',          type=str,   default='train_loss')
    parser.add_argument('--n_samples',       type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=5)
    parser.add_argument('--output_noise',    action='store_true',  default=False)
    parser.add_argument('--output_traj',     action='store_true',  default=False)
    parser.add_argument('--output_inception',       action='store_true',  default=False)
    parser.add_argument('--output_clean_inception', action='store_true',  default=False)
    parser.add_argument('--output_sample',          action='store_true',  default=False)
    parser.add_argument('--output_clip',            action='store_true',  default=False)
    parser.add_argument('--seed_offset',     type=int,   default=0)
    return parser

def parse_args() -> EasyDict:
    return EasyDict(vars(build_parser().parse_args()))

# ---------------------- user funcs (원본 유지) ----------------------
def get_model(config: EasyDict):
    if config.model == 'SANA':
        from backbones.sana import SANA
        if config.model_id is not None:
            return SANA(model_id=config.model_id)
        else:
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
        from solvers.others.euler_solver import Euler_Solver
        return Euler_Solver
    if config.solver == 'DPM-Solver':
        from solvers.others.dpm_solver import DPM_Solver
        return DPM_Solver
    if config.solver == 'UniPC':
        from solvers.others.unipc_solver import UniPC_Solver    
        return UniPC_Solver
    if config.solver == 'Dual-Solver':
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        return GDual_Solver
    if config.solver == 'BNS-Solver':
        from solvers.competing.bns.bns_solver import BNS_Solver
        return BNS_Solver
    if config.solver == 'BNS-Solver_Vec':
        from solvers.competing.bns.bns_solver_vec import BNS_Solver
        return BNS_Solver
    if config.solver == 'DS-Solver':
        from solvers.competing.ds.ds_solver import DS_Solver
        return DS_Solver
    if config.solver == 'DS-Solver_DDPM':
        from solvers.competing.ds.ds_solver_ddpm import DS_Solver
        return DS_Solver
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

# 공통 헬퍼
def compact(t, dtype=torch.float32):
    return t.detach().to(dtype).clone().cpu()

# ---------------------- main ----------------------
def main():
    config = parse_args()

    rank, world, local = init_dist()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    # rank0이 save_dir 결정/생성 후 모두에게 공유 (경합 방지)
    save_dir = get_sampling_dir(config) if rank == 0 else None
    save_dir = bcast_obj(save_dir, src=0)
    config.save_dir = save_dir
    if rank == 0:
        save_config(config)
    barrier(local)

    # 모델/솔버/데이터
    model  = get_model(config)
    Solver = get_solver(config)
    data   = get_data(config)
    if config.output_inception:
        inception = FIDInception().to(device)
    if config.output_clean_inception:
        clean_inception = CleanFIDInception().to(device)
    if config.output_clip:
        clip = CLIPEmbedder(device=getattr(model, "device", device))

    # 전역 인덱스 샤딩 (seed = seed_offset + global_idx 유지)
    all_idx = list(range(config.n_samples))
    my_idx  = all_idx[rank::world]

    n_iters = math.ceil(len(my_idx) / config.batch_size)
    pbar = tqdm(total=n_iters, desc=f"Sampling(rank{rank})", disable=(rank != 0))

    torch.backends.cudnn.benchmark = True
    #with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
    with torch.no_grad():
        ptr = 0
        while ptr < len(my_idx):
            batch_indices = my_idx[ptr: ptr + config.batch_size]
            ptr += config.batch_size

            conds = [data[i] for i in batch_indices]
            seeds = config.seed_offset + np.asarray(batch_indices, dtype=int)

            noise_schedule = model.get_noise_schedule()
            model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
            noises = model.get_noise(seeds=seeds)
            solver = Solver(noise_schedule, config.NFE, order=config.order,
                            skip_type=config.skip_type, flow_shift=config.flow_shift,
                            algorithm_type=config.algorithm_type, k=config.k).to(device)
            if config.pt_dir is not None:
                from utils.util import get_pt
                best_pt = get_pt(config.pt_dir, config.pt_criterion)
                state_dict = torch.load(best_pt, map_location='cpu', weights_only=False)['solver_state_dict']
                solver.load_state_dict(state_dict, strict=True)

            outputs = solver.sample(noises, model_fn, output_traj=config.output_traj)
            if config.output_inception or config.output_clip or config.output_clean_inception:
                decoded = model.decode_vae(outputs['samples'], raw_output=True, pil_output=True)
            if config.output_inception:
                inception_features = inception(decoded['pil_output']).detach().cpu()
            if config.output_clean_inception:
                clean_inception_features = clean_inception(decoded['pil_output']).detach().cpu()
            if config.output_clip:
                clip_features = clip.encode_image(decoded['raw_output']).detach().cpu()

            samples = outputs['samples'].detach().cpu()
            if config.output_noise:
                noises = noises.detach().cpu()
            if config.output_traj and 'trajs' in outputs:
                trajs = outputs['trajs'].detach().cpu()
                timesteps = outputs['timesteps'].detach().cpu()
                alphas = outputs['alphas'].detach().cpu()
                sigmas = outputs['sigmas'].detach().cpu()
            
            # 글로벌 인덱스로 저장 (충돌 없음)
            for j, gidx in enumerate(batch_indices):
                output = {'cond': conds[j]}
                if config.output_sample:
                    output['sample'] = compact(samples[j])
                if config.output_inception:
                    output['inception_feature'] = compact(inception_features[j])
                if config.output_clean_inception:
                    output['clean_inception_feature'] = compact(clean_inception_features[j])
                if config.output_clip:
                    output['clip_feature'] = compact(clip_features[j])
                if config.output_noise:
                    output['noise'] = compact(noises[j])
                if config.output_traj and trajs is not None:
                    output['traj'] = compact(trajs[j])
                    output['timesteps'] = compact(timesteps)
                    output['alphas'] = compact(alphas)
                    output['sigmas'] = compact(sigmas)
                torch.save(output, os.path.join(config.save_dir, f"{gidx}.pt"))

            pbar.update(1)

    pbar.close()
    barrier(local)
    if rank == 0:
        print(f"Done. Saved to: {config.save_dir}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
