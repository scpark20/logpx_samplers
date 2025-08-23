import argparse
import os
import re
import sys
import json
import math
import torch
import numpy as np
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
from utils.inception import FIDInception
from utils.clip import CLIPEmbedder

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Train")
    parser.add_argument('--model_json',           type=str,   default='SANA')
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
    parser.add_argument('--clip',            action='store_true',  default=False)
    parser.add_argument('--seed_offset',     type=int,   default=0)
    return parser

def parse_args() -> EasyDict:
    parser = build_parser()
    args = parser.parse_args()
    return EasyDict(vars(args))

def get_model(config: EasyDict):

def get_solver(config: EasyDict):

def get_data(config: EasyDict):

def save_config(config):
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(dict(config), f, indent=2)

def main():

if __name__ == '__main__':
    main()