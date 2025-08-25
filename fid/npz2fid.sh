#!/usr/bin/env bash
# 사용: (1) 아래 변수값을 편집하거나 (2) npz_file=path.npz ./run_npz2fid.sh
npz_file="${npz_file:-/data/scpark/fid/VIRTUAL_imagenet256_labeled.npz}"

python npz2fid.py "$npz_file" \
  --save_stats "${npz_file%.npz}_stats.pt"