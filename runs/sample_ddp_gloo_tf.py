#!/usr/bin/env python
import argparse, os, re, sys, json, math, torch, numpy as np
from easydict import EasyDict
from tqdm import tqdm
import tensorflow.compat.v1 as tf
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.disable_eager_execution()

from utils.tf_fid import InceptionPool3
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
    parser.add_argument('--order',           type=int,   default=2)
    parser.add_argument('--data',            type=str,   default='MSCOCO2017')
    parser.add_argument('--save_root',       type=str,   default='/data/scpark/samplings/')
    parser.add_argument('--n_samples',       type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=5)
    parser.add_argument('--output_noise',    action='store_true',  default=False)
    parser.add_argument('--output_traj',     action='store_true',  default=False)
    parser.add_argument('--inception',       action='store_true',  default=False)
    parser.add_argument('--sample',          action='store_true',  default=False)
    parser.add_argument('--clip',            action='store_true',  default=False)
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
    if config.inception:
        tf_cfg = tf.ConfigProto(allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        if torch.cuda.is_available():
            tf_cfg.gpu_options.visible_device_list = str(local)  # 각 rank GPU 고정
        # rank0이 먼저 다운로드 트리거
        if rank == 0:
            _sess0 = tf.Session(config=tf_cfg)
            _ = InceptionPool3(session=_sess0, batch_size=1)  # ensure model file exists
            _sess0.close()
        barrier(local)
        # 이제 각 rank에서 실제 extractor 생성
        tf_sess = tf.Session(config=tf_cfg)
        inception = InceptionPool3(session=tf_sess, batch_size=config.batch_size)
    else:
        tf_sess = None
        inception = None

    if config.clip:
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
                            algorithm_type=config.algorithm_type)

            outputs = solver.sample(noises, model_fn, output_traj=config.output_traj)
            if config.inception or config.clip:
                decoded = model.decode_vae(outputs['samples'], raw_output=True, pil_output=True)
            if config.inception:
                # PIL.Image 리스트 -> [N,H,W,3] uint8
                np_imgs = np.stack([np.array(im, dtype=np.uint8) for im in decoded['pil_output']], axis=0)
                inc_np = inception.extract(np_imgs)                 # [N,2048] float32 (numpy)
                inception_features = torch.from_numpy(inc_np)       # torch.Tensor(CPU)
            if config.clip:
                clip_features = clip.encode_image(decoded['raw_output']).detach().cpu()

            samples = outputs['samples'].detach().cpu()
            if config.output_noise:
                noises = noises.detach().cpu()
            if config.output_traj and 'trajs' in outputs:
                trajs = outputs['trajs'].detach().cpu()
            else:
                trajs = None

            # 글로벌 인덱스로 저장 (충돌 없음)
            for j, gidx in enumerate(batch_indices):
                output = {'cond': conds[j]}
                if config.sample:
                    output['sample'] = compact(samples[j])
                if config.inception:
                    output['inception_feature'] = compact(inception_features[j])
                if config.clip:
                    output['clip_feature'] = compact(clip_features[j])
                if config.output_noise:
                    output['noise'] = compact(noises[j])
                if config.output_traj and trajs is not None:
                    output['traj'] = compact(trajs[j])
                torch.save(output, os.path.join(config.save_dir, f"{gidx}.pt"))

            pbar.update(1)

    pbar.close()
    barrier(local)
    if rank == 0:
        print(f"Done. Saved to: {config.save_dir}", flush=True)
    if tf_sess is not None:
        tf_sess.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
