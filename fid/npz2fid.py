#!/usr/bin/env python
import argparse, numpy as np, torch, torch.nn.functional as F
from pytorch_fid.inception import InceptionV3

@torch.no_grad()
def extract_pool3_from_npz(npz_path, batch_size=128, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    net = InceptionV3([block]).to(device, dtype=torch.bfloat16).eval()

    arr = np.load(npz_path)["arr_0"]          # NHWC, [0..255]
    assert arr.ndim == 4 and arr.shape[-1] == 3, "arr_0 must be NHWC with 3 channels"

    feats = []
    for i in range(0, len(arr), batch_size):
        x = torch.from_numpy(arr[i:i+batch_size]).to(device, dtype=torch.float32) / 255.0
        x = x.permute(0,3,1,2)                # NHWC -> NCHW
        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, (299, 299), mode="bilinear",
                               align_corners=False, antialias=True)
        x = x.to(torch.bfloat16)
        f = net(x)[0].squeeze(-1).squeeze(-1) # (N, 2048)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)            # (N, 2048)

def compute_stats(feats: torch.Tensor):
    x = feats.to(torch.float64)
    mu = x.mean(0)
    xc = x - mu
    sigma = (xc.t() @ xc) / (x.shape[0] - 1)  # sample covariance (ddof=1)
    return mu, sigma, feats.shape[0]          # torch tensors (float64), int

def main():
    ap = argparse.ArgumentParser(description="NPZ(arr_0)->Inception(pool3) features & stats (.pt save)")
    ap.add_argument("npz", type=str)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--save_feats", type=str, help="path to save features .pt")
    ap.add_argument("--save_stats", type=str, help="path to save stats .pt (mu,sigma,n)")
    args = ap.parse_args()

    feats = extract_pool3_from_npz(args.npz, args.batch)
    print("features:", tuple(feats.shape))     # (N, 2048)

    if args.save_feats:
        torch.save({"feats": feats}, args.save_feats)
    if args.save_stats:
        mu, sigma, n = compute_stats(feats)
        torch.save({"mu": mu, "sigma": sigma, "n": n}, args.save_stats)

if __name__ == "__main__":
    main()
