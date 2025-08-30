#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, re, sys, json, math
import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm

# ----- 평가기 / 임베더 -----
from utils.fid import FIDInception
from utils.clean_fid import CleanFIDInception
from utils.clip import CLIPEmbedder

# ---------------------- arg parsing ----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run sampling (Single GPU)")
    p.add_argument('--tag',             type=str,   default='tag')
    p.add_argument('--model',           type=str,   default='SANA')
    p.add_argument('--model_id',        type=str,   default=None)
    p.add_argument('--solver',          type=str,   default='DPM-Solver')
    p.add_argument('--algorithm_type',  type=str,   default='data_prediction')
    p.add_argument('--skip_type',       type=str,   default='time_uniform')
    p.add_argument('--flow_shift',      type=float, default=3.0)
    p.add_argument('--NFE',             type=int,   default=10)
    p.add_argument('--CFG',             type=float, default=4.5)
    p.add_argument('--k',               type=float, default=0.5)
    p.add_argument('--order',           type=int,   default=2)
    p.add_argument('--data',            type=str,   default='MSCOCO2017')
    p.add_argument('--save_root',       type=str,   default='/data/scpark/samplings/')
    p.add_argument('--pt_dir',          type=str,   default=None)
    p.add_argument('--pt_criterion',    type=str,   default='train_loss')
    p.add_argument('--solver_ckpt',     type=str,   default=None, help='(선택) solver checkpoint 경로 직접 지정')
    p.add_argument('--n_samples',       type=int,   default=100)
    p.add_argument('--batch_size',      type=int,   default=5)
    p.add_argument('--output_noise',    action='store_true',  default=False)
    p.add_argument('--output_traj',     action='store_true',  default=False)
    p.add_argument('--output_inception',       action='store_true',  default=False)
    p.add_argument('--output_clean_inception', action='store_true',  default=False)
    p.add_argument('--output_sample',          action='store_true',  default=False)
    p.add_argument('--output_clip',            action='store_true',  default=False)
    p.add_argument('--seed_offset',     type=int,   default=0)
    return p

def parse_args() -> EasyDict:
    return EasyDict(vars(build_parser().parse_args()))

# ---------------------- user funcs (원본 유지) ----------------------
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
        return [i % 1000 for i in range(config.n_samples)]
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

# ---------------------- utils ----------------------
def compact(t: torch.Tensor, dtype=torch.float32):
    return t.detach().to(dtype).clone().cpu()

def find_ckpt_path(cfg) -> str | None:
    """우선순위: --solver_ckpt > pt_dir/best.pt > pt_dir/ckpt_*.pt 최신"""
    if getattr(cfg, "solver_ckpt", None) and os.path.isfile(cfg.solver_ckpt):
        return cfg.solver_ckpt
    if getattr(cfg, "pt_dir", None):
        best = os.path.join(cfg.pt_dir, "best.pt")
        if os.path.isfile(best):
            return best
        import glob
        cands = sorted(glob.glob(os.path.join(cfg.pt_dir, "ckpt_*.pt")))
        if cands:
            return cands[-1]
    return None

# ---------------------- main (single GPU) ----------------------
def main():
    cfg = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 저장 디렉토리 준비
    cfg.save_dir = get_sampling_dir(cfg)
    save_config(cfg)

    # 모델/솔버/데이터
    model  = get_model(cfg)
    Solver = get_solver(cfg)
    data   = get_data(cfg)

    # 평가기 준비(필요 시)
    inception = FIDInception().to(device) if cfg.output_inception else None
    clean_inception = CleanFIDInception().to(device) if cfg.output_clean_inception else None
    clip = CLIPEmbedder(device=getattr(model, "device", device)) if cfg.output_clip else None

    # 전역 인덱스(싱글 프로세스이므로 그대로)
    all_idx = list(range(cfg.n_samples))
    n_iters = math.ceil(len(all_idx) / cfg.batch_size)
    pbar = tqdm(total=n_iters, desc="Sampling(single)")

    # 체크포인트 (옵션)
    solver_ckpt_path = find_ckpt_path(cfg)
    if solver_ckpt_path is None and cfg.pt_dir is not None:
        print(f"[WARN] No checkpoint found under {cfg.pt_dir} (expect best.pt or ckpt_*.pt). Proceeding w/o loading.")

    with torch.no_grad():
        ptr = 0
        while ptr < len(all_idx):
            batch_indices = all_idx[ptr: ptr + cfg.batch_size]
            ptr += cfg.batch_size

            conds = [data[i] for i in batch_indices]
            seeds = cfg.seed_offset + np.asarray(batch_indices, dtype=int)

            noise_schedule = model.get_noise_schedule()
            model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=cfg.CFG)
            noises = model.get_noise(seeds=seeds)

            solver = Solver(noise_schedule, cfg.NFE, order=cfg.order,
                            skip_type=cfg.skip_type, flow_shift=cfg.flow_shift,
                            algorithm_type=cfg.algorithm_type, k=cfg.k).to(device)

            # 안전 로드
            if solver_ckpt_path is not None:
                state_obj = torch.load(solver_ckpt_path, map_location='cpu', weights_only=False)
                if "solver_state_dict" in state_obj:
                    solver.load_state_dict(state_obj["solver_state_dict"], strict=True)
                else:
                    print(f"[WARN] solver_state_dict not in {solver_ckpt_path}; skip loading.")

            outputs = solver.sample(noises, model_fn, output_traj=cfg.output_traj)

            decoded = None
            if cfg.output_inception or cfg.output_clip or cfg.output_clean_inception:
                decoded = model.decode_vae(outputs['samples'], raw_output=True, pil_output=True)

            # 특성 추출
            if cfg.output_inception:
                inception_features = inception(decoded['pil_output']).detach().cpu()
            if cfg.output_clean_inception:
                clean_inception_features = clean_inception(decoded['pil_output']).detach().cpu()
            if cfg.output_clip:
                clip_features = clip.encode_image(decoded['raw_output']).detach().cpu()

            samples = outputs['samples'].detach().cpu()
            if cfg.output_noise:
                noises = noises.detach().cpu()

            # 트래젝토리(안전 초기화)
            trajs = timesteps = alphas = sigmas = None
            if cfg.output_traj and ('trajs' in outputs):
                trajs = outputs['trajs'].detach().cpu()
                timesteps = outputs['timesteps'].detach().cpu()
                alphas = outputs['alphas'].detach().cpu()
                sigmas = outputs['sigmas'].detach().cpu()

            # 저장
            for j, gidx in enumerate(batch_indices):
                out = {'cond': conds[j]}
                if cfg.output_sample:
                    out['sample'] = compact(samples[j])
                if cfg.output_inception:
                    out['inception_feature'] = compact(inception_features[j])
                if cfg.output_clean_inception:
                    out['clean_inception_feature'] = compact(clean_inception_features[j])
                if cfg.output_clip:
                    out['clip_feature'] = compact(clip_features[j])
                if cfg.output_noise:
                    out['noise'] = compact(noises[j])
                if cfg.output_traj and (trajs is not None):
                    out['traj'] = compact(trajs[j])
                    out['timesteps'] = compact(timesteps)
                    out['alphas'] = compact(alphas)
                    out['sigmas'] = compact(sigmas)
                torch.save(out, os.path.join(cfg.save_dir, f"{gidx}.pt"))

            pbar.update(1)

    pbar.close()
    print(f"Done. Saved to: {cfg.save_dir}", flush=True)

if __name__ == '__main__':
    main()
