#!/usr/bin/env python
import argparse, os, re, sys, json, math, torch, numpy as np
from easydict import EasyDict
from pathlib import Path
from PIL import Image
from tqdm import tqdm, trange   # ← trange 추가
from utils.fid import FIDInception
from utils.clean_fid import CleanFIDInception
from utils.clip import CLIPEmbedder
from utils.tf_fid_cpu import tf_inception_encode
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
    parser.add_argument('--gamma_init',      type=float, default=0.0)
    parser.add_argument('--tau_init',        type=float, default=1.0)
    parser.add_argument('--tau_x_init',        type=float, default=1.0)
    parser.add_argument('--tau_e_init',        type=float, default=1.0)
    parser.add_argument('--NFE',             type=int,   default=10)
    parser.add_argument('--CFG',             type=float, default=4.5)
    #parser.add_argument('--cfg_channels',    type=str,   default='full')
    parser.add_argument('--k',               type=float, default=0.5)
    parser.add_argument('--order',           type=int,   default=2)
    parser.add_argument('--afs', type=lambda s: s.lower() == 'true', default=False, help='Use AFS (pass "True" or "False")')
    parser.add_argument('--data',            type=str,   default='MSCOCO2014_valid')
    parser.add_argument('--save_root',       type=str,   default='/data/scpark/samplings/')
    parser.add_argument('--pt_dir',          type=str,   default=None)
    parser.add_argument('--pt_criterion',    type=str,   default='train_loss')
    parser.add_argument('--pt_step',          type=str,   default=None)
    parser.add_argument('--n_samples',       type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=5)
    parser.add_argument('--bottleneck_dim',  type=int,   default=1024)
    parser.add_argument('--output_noise',    action='store_true',  default=False)
    parser.add_argument('--output_traj',     action='store_true',  default=False)
    parser.add_argument('--output_preds',     action='store_true',  default=False)
    parser.add_argument('--output_inception',       action='store_true',  default=False)
    parser.add_argument('--output_clean_inception', action='store_true',  default=False)
    parser.add_argument('--output_tf_inception',    action='store_true', default=False)
    parser.add_argument('--output_sample',          action='store_true',  default=False)
    parser.add_argument('--output_clip',            action='store_true',  default=False)
    parser.add_argument('--output_clip_score',      action='store_true',  default=False)
    parser.add_argument('--output_raw',             action='store_true',  default=False)
    # (기존) parser.add_argument('--clip_model',            type=str,   default='ViT-B/16')
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                        help='CLIP model name or comma-separated list (e.g., "ViT-L/14, ViT-L/14@336px, RN101")')
    parser.add_argument('--output_png', action='store_true', default=False,
                        help='Decode VAE output and save PNGs to save_dir as {gidx:06d}.png')
    parser.add_argument('--build_npz', action='store_true', default=False,
                        help='After sampling, build samples.npz (arr_0, NHWC uint8) from saved PNGs (rank0 only)')


    parser.add_argument('--seed_offset',     type=int,   default=0)
    parser.add_argument('--dtype',       type=str, default='bf16',
                        help="Torch dtype: one of {bf16, fp32, fp16} (DiT honors this).")
    return parser

def parse_args() -> EasyDict:
    return EasyDict(vars(build_parser().parse_args()))

# ---- add below (helper) ----
def resolve_dtype(name: str) -> torch.dtype:
    n = (name or "").lower()
    if n in ("bf16", "bfloat16"): return torch.bfloat16
    if n in ("fp32", "float32", "float"): return torch.float32
    if n in ("fp16", "float16", "half"):  return torch.float16
    raise ValueError(f"Unknown dtype: {name}. Use one of bf16|fp32|fp16")

# ---- PNG → NPZ 빌더 ----
def build_npz_from_pngs(png_dir: Path, out_npz: Path, n: int):
    """
    png_dir/{000000.png, 000001.png, ...} → out_npz (arr_0: NHWC uint8)
    NOTE: n=50_000, 256x256x3이면 메모리 ~9.8GB 필요.
    """
    arr = np.empty((n, 256, 256, 3), dtype=np.uint8)  # NHWC
    for i in trange(n, desc="Building NPZ"):
        im = Image.open(png_dir / f"{i:06d}.png").convert("RGB")
        arr[i] = np.asarray(im, dtype=np.uint8)
    np.savez_compressed(out_npz, arr_0=arr)


# ---------------------- user funcs (원본 유지) ----------------------
def get_model(config: EasyDict):
    dt = resolve_dtype(config.dtype)

    if config.model == 'SD':
        from backbones.stable_diffusion import StableDiffusion
        try:
            return StableDiffusion(model_id=config.model_id, dtype=dt) if config.model_id is not None else StableDiffusion(dtype=dt)
        except TypeError:
            return StableDiffusion(model_id=config.model_id) if config.model_id is not None else StableDiffusion()

    if config.model == 'SANA':
        from backbones.sana import SANA
        try:
            return SANA(model_id=config.model_id, dtype=dt) if config.model_id is not None else SANA(dtype=dt)
        except TypeError:
            return SANA(model_id=config.model_id) if config.model_id is not None else SANA()

    if config.model == 'PixArt-Sigma':
        from backbones.pixart_sigma import PixArtSigma
        try:
            return PixArtSigma(dtype=dt)
        except TypeError:
            return PixArtSigma()

    if config.model == 'PixArt-Alpha':
        from backbones.pixart_alpha import PixArtAlpha
        try:
            return PixArtAlpha(dtype=dt)
        except TypeError:
            return PixArtAlpha()

    if config.model == 'DiT':
        from backbones.dit import DiT
        # DiT 백본은 dtype 인자를 지원 (이전 메시지의 클래스와 호환)
        return DiT(dtype=dt, model_id=config.model_id) if config.model_id is not None else DiT(dtype=dt)

    if config.model == 'GMDiT':
        GMFLOW = os.path.join("submodules", "GMFlow")
        sys.path.insert(0, GMFLOW)
        from backbones.gmdit import GMDiT
        try:
            return GMDiT(dtype=dt)
        except TypeError:
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
        
    if config.solver == 'Dual-Solver_LogLinear':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_max=2, kappa_max=2, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL2':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform2 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_max=5, kappa_max=5, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL3':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform3 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_GEO':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_geo_transform import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL4':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform4 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL5':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform5 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL6':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform6 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_x_init=config.tau_x_init, tau_e_init=config.tau_e_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL7':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform7 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL9':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform9 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_LL10':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform10 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_init=config.gamma_init, tau_init=config.tau_init, eps=1e-2)
        return partial(GDual_Solver, transform=transform)

    if config.solver == 'Dual-Solver_CLIP':
        from functools import partial
        from solvers.taylor.solver.gdual_solver import GDual_Solver
        from solvers.taylor.transform.loglinear_transform2 import LogLinearTransform
        transform = LogLinearTransform(gamma_push=True, gamma_max=5, kappa_max=5, eps=1e-2)
        return partial(GDual_Solver, transform=transform, param_extractor='tau_table_extractor')

    if config.solver == 'Dual-Solver_Quad' or config.solver == 'Dual-Solver_Legendre':
        from solvers.taylor.solver.gdual_solver_steps_list import GDual_Solver
        return GDual_Solver    
    if config.solver == 'BNS-Solver':
        from solvers.competing.bns.bns_solver import BNS_Solver
        return BNS_Solver
    if config.solver == 'BNS-Solver_Vec':
        from solvers.competing.bns.bns_solver_vec import BNS_Solver
        return BNS_Solver
    if config.solver == 'BNS-Solver_Sep':
        from solvers.competing.bns.bns_solver_sep import BNS_Solver
        return BNS_Solver
    if config.solver == 'DS-Solver_DDPM':
        from solvers.competing.ds.ds_solver_diffusion import DS_Solver
        return DS_Solver
    if config.solver == 'DS-Solver_Flow':
        from solvers.competing.ds.ds_solver_flow import DS_Solver
        return DS_Solver
    if config.solver == 'EPD-Solver_ARI':
        from solvers.competing.epd.epd_solver_arithmetic import EPD_Solver
        return EPD_Solver
    if config.solver == 'EPD-Solver_GEO':
        from solvers.competing.epd.epd_solver_geometric import EPD_Solver
        return EPD_Solver
    if config.solver == 'AMED-Solver_GEO':
        from solvers.competing.amed.amed_solver_geo import AMED_Solver
        return AMED_Solver
    if config.solver == 'AMED-Solver_ARI':
        from solvers.competing.amed.amed_solver_ari import AMED_Solver
        return AMED_Solver
    if config.solver == 'DDPM-Solver':
        from solvers.others.ddpm_solver import DDPM_Solver
        return DDPM_Solver
    raise ValueError(f"Unknown solver: {config.solver}")

def get_data(config: EasyDict):
    if config.data == 'MSCOCO2014_train':
        data = np.load('prompts/mscoco2014_train.npz')['arr_0'].tolist()
        data = [d[1] for d in data]
        return data
    if config.data == 'MSCOCO2014_valid':
        data = np.load('prompts/mscoco2014_valid.npz')['arr_0'].tolist()
        data = [d[1] for d in data]
        return data
    if config.data == 'MSCOCO2014_valid_30k':
        data = np.load('prompts/mscoco2014_valid_30k.npz')['arr_0'].tolist()
        data = [d[1] for d in data]
        return data
    if config.data == 'MJHQ-30k':
        import json
        with open('mjhq_fid/prompts.json', 'r') as f:
            json_data = json.load(f)
        data = [json_data[key]['prompt'] for key in json_data]
        return data
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
    # 무시하고 덮어 쓰고 싶으면 -1 할당            
    i = -1
    sampling_dir = os.path.join(r, f"{p}_{i+1}")
    os.makedirs(sampling_dir, exist_ok=True)        
    return sampling_dir

def save_config(config):
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(dict(config), f, indent=2)

# ---------------------- DDP helpers (최소 추가) ----------------------
from datetime import timedelta
import torch.distributed as dist
import torch

def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    backend = os.environ.get("DIST_BACKEND", "gloo")  # ← 기본 gloo
    if world > 1:
        rank  = int(os.environ["RANK"])
        local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        torch.cuda.set_device(local)
        dev = torch.device(f"cuda:{local}")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=300),
        )
        return rank, world, local
    return 0, 1, 0

def barrier(_local_rank: int):
    if dist.is_available() and dist.is_initialized():
        dist.barrier()            
            
def bcast_obj(obj, src=0):
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]

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

    # ---- CHANGES: CLIP 여러 개 로딩 (콤마 분리) ----
    clip_models = [m.strip() for m in str(config.clip_model).split(',') if m.strip()]
    clip_nets = None
    if config.output_clip or config.output_clip_score:
        clip_nets = {m: CLIPEmbedder(model_name=m, device=device) for m in clip_models}

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

            all_exists = True
            for gidx in batch_indices:
                file = os.path.join(config.save_dir, f"{gidx}.pt")
                if not os.path.exists(file):
                    all_exists = False
                    break        
            if all_exists:
                print('All Exists, ptr :', ptr)
                pbar.update(1)
                continue
            
            conds = [data[i] for i in batch_indices]
            seeds = config.seed_offset + np.asarray(batch_indices, dtype=int)

            noise_schedule = model.get_noise_schedule()
            model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)#, cfg_channels=config.cfg_channels)
            noises = model.get_noise(seeds=seeds)
            solver = Solver(noise_schedule, steps=config.NFE, order=config.order,
                            skip_type=config.skip_type, flow_shift=config.flow_shift, bottleneck_dim=config.bottleneck_dim,
                            algorithm_type=config.algorithm_type, k=config.k, use_afs=config.afs, solver=config.solver).to(device)
            if config.pt_dir is not None:
                from utils.util import get_pt
                best_pt = get_pt(config.pt_dir, config.pt_criterion, config.pt_step)
                state_dict = torch.load(best_pt, map_location='cpu', weights_only=False)['solver_state_dict']
                solver.load_state_dict(state_dict, strict=False)
                #print('Loaded checkpoint :', best_pt)

            outputs = solver.sample(noises, model_fn, output_traj=config.output_traj, output_preds=config.output_preds, backbone=model)

            # ---- 디코딩 필요 여부 (PNG/특징 추출 모두 포함) ----
            needs_decode = (
                config.output_png or
                config.output_inception or config.output_clean_inception or
                config.output_tf_inception or config.output_clip or config.output_clip_score or config.output_raw
            )
            if needs_decode:
                decoded = model.decode_vae(outputs['samples'], raw_output=True, pil_output=True)

            # ---- 특징 추출 ----
            if config.output_inception:
                inception_features = inception(decoded['pil_output']).detach().cpu()
            if config.output_clean_inception:
                clean_inception_features = clean_inception(decoded['pil_output']).detach().cpu()
            if config.output_tf_inception:
                tf_feats = tf_inception_encode(decoded['pil_output'])  # np.float32 [B,2048]
                tf_inception_features = torch.from_numpy(tf_feats).to("cpu", dtype=torch.float32)

            # ---- CHANGES: 여러 CLIP 모델 각각 처리 ----
            clip_features_dict = {}
            clip_scores_dict   = {}
            if config.output_clip and clip_nets is not None:
                for m, net in clip_nets.items():
                    clip_features_dict[m] = net.encode_image(decoded['raw_output']).detach().cpu()
            if config.output_clip_score and clip_nets is not None:
                # autocast는 그대로 유지
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    for m, net in clip_nets.items():
                        clip_scores_dict[m] = (
                            1 - net.get_cossim_loss(decoded['raw_output'], conds, clamp_mode='hard', reduction='none')
                        ).detach().cpu()

            samples = outputs['samples'].detach().cpu()
            if config.output_noise:
                noises = noises.detach().cpu()

            # 안전 초기화 (solver가 trajs 키를 안 줄 수도 있음)
            preds = trajs = timesteps = alphas = sigmas = None
            if config.output_traj and ('trajs' in outputs):
                trajs = outputs['trajs'].detach().cpu()
                timesteps = outputs['timesteps'].detach().cpu()
                alphas = outputs['alphas'].detach().cpu()
                sigmas = outputs['sigmas'].detach().cpu()
            if config.output_preds and ('preds' in outputs):
                preds = outputs['preds'].detach().cpu()
                timesteps = outputs['timesteps'].detach().cpu()
                alphas = outputs['alphas'].detach().cpu()
                sigmas = outputs['sigmas'].detach().cpu()

            # ---- 개별 저장 (전역 인덱스) ----
            for j, gidx in enumerate(batch_indices):
                output = {'cond': conds[j]}

                # PNG 저장
                if config.output_png:
                    img = decoded['pil_output'][j]   # PIL.Image.Image
                    (Path(config.save_dir) / f"{gidx:06d}.png").parent.mkdir(parents=True, exist_ok=True)
                    img.save(Path(config.save_dir) / f"{gidx:06d}.png")
                if config.output_sample:
                    output['sample'] = compact(samples[j])
                if config.output_raw:
                    output['raw'] = compact(decoded['raw_output'][j])
                if config.output_inception:
                    output['inception_feature'] = compact(inception_features[j])
                if config.output_clean_inception:
                    output['clean_inception_feature'] = compact(clean_inception_features[j])
                if config.output_tf_inception:
                    output['tf_inception_feature'] = compact(tf_inception_features[j])

                # ---- CHANGES: CLIP per-model로 저장 ----
                if config.output_clip and clip_features_dict:
                    for m in clip_models:
                        output[f'clip_feature_{m}'] = compact(clip_features_dict[m][j])
                if config.output_clip_score and clip_scores_dict:
                    for m in clip_models:
                        output[f'clip_score_{m}'] = compact(clip_scores_dict[m][j])
                    
                if config.output_noise:
                    output['noise'] = compact(noises[j])
                if (config.output_traj and (trajs is not None)):
                    output['traj'] = compact(trajs[j])
                    output['timesteps'] = compact(timesteps)
                    output['alphas'] = compact(alphas)
                    output['sigmas'] = compact(sigmas)
                if (config.output_preds and (preds is not None)):
                    output['preds'] = compact(preds[j])
                    output['timesteps'] = compact(timesteps)
                    output['alphas'] = compact(alphas)
                    output['sigmas'] = compact(sigmas)

                torch.save(output, os.path.join(config.save_dir, f"{gidx}.pt"))

            pbar.update(1)

    pbar.close()
    barrier(local)

    # rank0에서만 NPZ 생성
    if rank == 0 and config.build_npz:
        out_npz_path = Path(config.save_dir) / "samples.npz"
        build_npz_from_pngs(Path(config.save_dir), out_npz_path, config.n_samples)
        print(f"NPZ built: {out_npz_path}", flush=True)

    if rank == 0:
        print(f"Done. Saved to: {config.save_dir}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
