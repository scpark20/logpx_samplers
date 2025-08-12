# MS-COCO FID 측정 코드 
# LD3와 AMED-Solver참고하여 작성 
# FID 측정 통계 값 --> ref_stats 폴더에 다음 링크에 포함된 데이터 다운 https://github.com/zju-pi/diff-sampler/blob/main/diff-solvers-main/README.md
# FID 측정 방식 관련 참고 - https://github.com/boomb0om/text2image-benchmark 

import argparse
import os
import sys
import math
import torch
import numpy as np
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import pickle
from PIL import Image
from runs.dnnlib.util import open_url

from runs.sample_distributed import  get_model
import scipy.linalg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampling")
    parser.add_argument('--ref_path',        type=str,   default='ref_stats/ms_coco-512x512.npz')
    parser.add_argument('--n_samples',       type=int,   default=30000)
    parser.add_argument('--sample_path',     type=str,   \
        default='samplings/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG3.0)(ORDER2)'
        )
    return parser

def parse_args() -> EasyDict:
    parser = build_parser()
    args = parser.parse_args()
    return EasyDict(vars(args))

def calc_inception_stats(sample_path, num_expected=None):
    
    BATCH_SIZE=250
    FEATURE_DIM=2048

    mu = torch.zeros([FEATURE_DIM], dtype=torch.float64)
    sigma = torch.zeros([FEATURE_DIM, FEATURE_DIM], dtype=torch.float64)

    if os.path.exists(sample_path):
        pt_files = [f for f in os.listdir(sample_path) if f.endswith('.pt')]
        pt_files.sort()
        n_samples = len(pt_files)
        print(f"Found {n_samples} .pt files in {sample_path}")
    else:
        raise FileNotFoundError(f"Sample path not found: {sample_path}")

    if num_expected is not None and len(pt_files) < num_expected:
        raise Exception(f'Found {len(pt_files)} images, but expected at least {num_expected}')

    n_samples = num_expected
    
    for start in tqdm(range(0, n_samples, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, n_samples)
        
        # Load feaures processed by StyleGAN3 Inception
        batch = []
        for i in range(start, end):
            pt_path = os.path.join(sample_path, pt_files[i])
            feature = torch.load(pt_path)['features'].cpu()
            batch.append(feature)
            
        features = torch.stack(batch)
        mu += features.sum(0)
        sigma += features.T @ features

    mu /= n_samples
    sigma -= mu.ger(mu) * n_samples
    sigma /= n_samples - 1

    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def main():
    config = parse_args()
    ref = None
    with open_url(config.ref_path) as f:
        ref = dict(np.load(f))
    
    mu, sigma = calc_inception_stats(config.sample_path, num_expected=config.n_samples)

    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'FID : {fid:g}')


if __name__ == '__main__':
    main()