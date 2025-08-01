

# MS-COCO FID 측정 코드 
#   DPM-Solver Repo에서는 평가 코드 못찾음 
#   LD3 혹은 AMED-Solver는 있음. 이것 참고~?

# data와 측정 방식에 따라 다르게 reference 파일을 로드하고, 
# path 혹은 통계값을 전달 받아 fid 를 계산하도록 처리 
# --> sampling 된 경로를 입력받아 계산하도록 처리 (생성된 .pt파일을 디코딩한 후 inception model로 처리)

# SANA는 다른 MJHQ-30K 으로 FID 측정함 (이건 구현 안함)


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

from runs.dnnlib.util import open_url, set_cache_dir

from runs.sample import get_data
from runs.sample_distributed import  get_model

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
    parser.add_argument('--batch_size',      type=int,   default=5)
    parser.add_argument('--device',          type=str,   default='cuda')
    parser.add_argument('--sample_path',     type=str,   \
        default='samplings/SANA(MSCOCO2017)(DPM-Solver)(data_prediction)(time_uniform_flow)(FS3.0)(NFE3)(CFG3.5)(ORDER2)'
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

    decoder = get_model(EasyDict({'model': model}), 0).decode_vae

    with open_url(detector_url) as f:
        detector_net = pickle.load(f).to(device)

    if os.path.exists(sample_path):
        pt_files = [f for f in os.listdir(sample_path) if f.endswith('.pt')]
        pt_files.sort()
        n_samples = len(pt_files)
        print(f"Found {n_samples} .pt files in {sample_path}")
    else:
        raise FileNotFoundError(f"Sample path not found: {sample_path}")

    if num_expected is not None and len(pt_files) < num_expected:
        raise Exception(f'Found {len(pt_files)} images, but expected at least {num_expected}')

    
    for start in tqdm(range(0, n_samples, batch_size)
    ):
        end = min(start + batch_size, n_samples)
        
        # Load .pt files and stack into batch
        batch = []
        for i in range(start, end):
            pt_path = pt_files[i]
            sample = torch.load(os.path.join(sample_path, pt_path), map_location=device)
            batch.append(sample)

        batch = torch.stack(batch)
        imgs = decoder(batch, output_type='pt')
        features = detector_net(imgs, **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    mu /= len(pt_files)
    sigma -= mu.ger(mu) * len(pt_files)
    sigma /= len(pt_files) - 1

    return mu.cpu().numpy(), sigma.cpu().numpy()



def main():
    config = parse_args()
    # def calc_inception_stats(sample_path, model, num_expected=None, batch_size=10, device='cuda'):
    mu, sigma = calc_inception_stats(config.sample_path, config.model, num_expected=1000, batch_size=config.batch_size, device=config.device)
    print('mu.shape : ', mu.shape)
    print('sigma.shape : ', sigma.shape)
    print('mu : ', mu)
    print('sigma : ', sigma)
    
    # TODO : reference data의 stat과 비교하여 FID 계산 





if __name__ == '__main__':
    main()