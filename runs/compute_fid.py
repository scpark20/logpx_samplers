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
    # New options for computing stats directly from an image directory using StyleGAN3 Inception
    parser.add_argument('--images_path',     type=str,   default=None,
                        help='Directory containing images to extract StyleGAN3 Inception features from')
    parser.add_argument('--out_stats_path',  type=str,   default=None,
                        help='Output npz path to save stats (mu, sigma). Defaults to prompts/<images_dir_name>_stylegan3.npz')
    parser.add_argument('--device',          type=str,   default=None,
                        help='Torch device for detector (e.g., cuda:0 or cpu). Defaults to cuda:0 if available else cpu')
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

def _load_stylegan3_inception(device: str):
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    with open_url(detector_url) as f:
        detector_net = pickle.load(f).to(device)
    return detector_net, detector_kwargs

def _list_image_files(images_path: str):
    exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    files = [f for f in os.listdir(images_path) if os.path.splitext(f)[1] in exts]
    files.sort()
    return files

@torch.no_grad()
def calc_inception_stats_from_images(images_path: str, num_expected: int = None, device: str = None):
    """Compute mu, sigma of StyleGAN3 Inception features from an image directory.

    Mirrors the feature extraction path used in runs/sample_distributed.py:
      - Load StyleGAN3 Inception (inception-2015-12-05.pkl)
      - Convert PIL image -> numpy -> torch tensor [C,H,W] with dtype uint8
      - No explicit resizing/normalization; detector handles it internally
      - Return 2048-dim features in float64 and aggregate stats
    """
    if not os.path.isdir(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    detector_net, detector_kwargs = _load_stylegan3_inception(device)

    image_files = _list_image_files(images_path)
    if len(image_files) == 0:
        raise Exception(f"No images found under: {images_path}")

    if num_expected is not None and len(image_files) < num_expected:
        raise Exception(f'Found {len(image_files)} images, but expected at least {num_expected}')

    n = num_expected if num_expected is not None else len(image_files)

    feature_dim = 2048
    sum_feat = torch.zeros([feature_dim], dtype=torch.float64)
    sum_outer = torch.zeros([feature_dim, feature_dim], dtype=torch.float64)

    print(f"Extracting StyleGAN3 Inception features from {n} images (device: {device})")
    for idx, fname in enumerate(image_files[:n]):
        path = os.path.join(images_path, fname)
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")
            continue

        arr = np.array(img)  # HWC, uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)  # CHW, uint8
        tensor = tensor.unsqueeze(0)  # BCHW

        feats = detector_net(tensor, **detector_kwargs).to(torch.float64)  # [1, 2048]
        f = feats[0].detach().cpu()  # float64 on CPU

        sum_feat += f
        sum_outer += torch.outer(f, f)

        if (idx + 1) % 500 == 0 or (idx + 1) == n:
            print(f"  Progress: {idx + 1}/{n} ({(idx + 1) / n * 100:.1f}%)")

    # Unbiased covariance estimate
    mu = sum_feat / n
    sigma = (sum_outer - torch.outer(mu, mu) * n) / (n - 1)

    return mu.numpy(), sigma.numpy()

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def main():
    config = parse_args()
    # If images_path is provided, compute StyleGAN3 Inception stats from real images and optionally exit
    if config.images_path is not None:
        out_path = config.out_stats_path
        if out_path is None:
            base = os.path.basename(os.path.normpath(config.images_path))
            out_path = os.path.join('prompts', f'{base}_stylegan3.npz')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        mu, sigma = calc_inception_stats_from_images(
            config.images_path,
            num_expected=config.n_samples,
            device=config.device
        )
        np.savez(out_path, mu=mu, sigma=sigma)
        print(f"Saved stats to: {out_path}")
        return

    # Otherwise, compute FID between precomputed sample features and a reference npz
    ref = None
    with open_url(config.ref_path) as f:
        ref = dict(np.load(f))
    mu, sigma = calc_inception_stats(config.sample_path, num_expected=config.n_samples)
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'FID : {fid:g}')


if __name__ == '__main__':
    main()