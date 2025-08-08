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
import torchvision.transforms as transforms

from runs.sample_distributed import  get_model
import scipy.linalg

# Import torch_utils.persistence to resolve pickle dependencies
import runs.torch_utils as torch_utils_module
import runs.torch_utils.persistence as persistence
import runs.dnnlib as dnnlib_module
import sys
sys.modules['torch_utils'] = torch_utils_module
sys.modules['torch_utils.persistence'] = persistence
sys.modules['dnnlib'] = dnnlib_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampling")
    parser.add_argument('--model',           type=str,   default='SANA')
    parser.add_argument('--data',            type=str,   default='MSCOCO2017')
    parser.add_argument('--batch_size',      type=int,   default=25)
    parser.add_argument('--device',          type=str,   default='cuda')
    parser.add_argument('--ref_path',        type=str,   default='ref_stats/ms_coco-512x512.npz')
    parser.add_argument('--n_samples',       type=int,   default=3000)
    parser.add_argument('--sample_path',     type=str,   \
        default='/AiWorkflowStg/jyshin_backup/logpx_sampler/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE10)(CFG3.5)(ORDER2)'
        )
    
    return parser

def parse_args() -> EasyDict:
    parser = build_parser()
    args = parser.parse_args()
    return EasyDict(vars(args))

def calc_inception_stats(sample_path, model, num_expected=None, batch_size=10, device='cuda'):
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    with open_url(detector_url) as f:
        detector_net = pickle.load(f).to(device)

    # Define preprocessing transform for StyleGAN3 Inception model
    # StyleGAN3 expects [0,255] range uint8 images, NOT ImageNet normalized
    transform = transforms.Compose([
        transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        # Do NOT use transforms.ToTensor() as it normalizes to [0,1]
        # Do NOT use ImageNet normalization
    ])

    if os.path.exists(sample_path):
        png_files = [f for f in os.listdir(sample_path) if f.endswith('.png')]
        png_files.sort()
        n_samples = len(png_files)
        print(f"Found {n_samples} .png files in {sample_path}")
    else:
        raise FileNotFoundError(f"Sample path not found: {sample_path}")

    if num_expected is not None and len(png_files) < num_expected:
        raise Exception(f'Found {len(png_files)} images, but expected at least {num_expected}')

    n_samples = num_expected
    
    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        
        # Load and preprocess images for StyleGAN3 Inception
        batch = []
        for i in range(start, end):
            img_path = os.path.join(sample_path, png_files[i])
            image = Image.open(img_path).convert('RGB')  # Ensure RGB format
            
            # Apply only resize and crop (no normalization)
            image = transform(image)
            
            # Convert PIL to numpy array [H, W, C] with values [0, 255]
            img_array = np.array(image, dtype=np.uint8)
            
            # Convert to torch tensor and change to [C, H, W] format
            # Keep values in [0, 255] range as StyleGAN3 expects
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            
            batch.append(tensor)

        # Stack into batch tensor: [batch_size, 3, 299, 299] with values [0, 255]
        batch = torch.stack(batch).to(device)
        
        features = detector_net(batch, **detector_kwargs).to(torch.float64)

        mu += features.sum(0)
        sigma += features.T @ features

    mu /= len(png_files)
    sigma -= mu.ger(mu) * len(png_files)
    sigma /= len(png_files) - 1

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

    print('ref : ', ref)
    
    # def calc_inception_stats(sample_path, model, num_expected=None, batch_size=10, device='cuda'):
    mu, sigma = calc_inception_stats(config.sample_path, config.model, num_expected=config.n_samples, batch_size=config.batch_size, device=config.device)
    print('mu.shape : ', mu.shape)
    print('sigma.shape : ', sigma.shape)
    print('mu : ', mu)
    print('sigma : ', sigma)
    
    # TODO : reference data의 stat과 비교하여 FID 계산 

    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'FID : {fid:g}')



if __name__ == '__main__':
    main()