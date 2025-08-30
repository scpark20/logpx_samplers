import os, csv, json
import torch, torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageFile
from utils.fid import FIDInception
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False  # 깨진/잘린 이미지에서 예외 발생하도록

# ── Dataset: 깨진 파일 발견 시 즉시 예외 → 전체 중단 ───────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")  # 변환
                im.load()               # 전체 디코드 → 손상 시 여기서 예외
            return im, str(p)
        except Exception as e:
            raise RuntimeError(f"Corrupted image detected: {p}") from e

def extract_feats_ddp(in_root, out_dir, batch_size=1024, shard_size=20000, num_workers=8):
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    model = FIDInception(device=device).eval()

    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    all_paths = sorted(p for p in Path(in_root).rglob("*") if p.is_file() and p.suffix.lower() in exts)
    paths_rank = all_paths[rank::world]

    ds = ImageDataset(paths_rank)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, drop_last=False, collate_fn=lambda b: b)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    writer = csv.writer((out/f"index_rank{rank}.csv").open("w", newline=""))
    writer.writerow(["image_path","shard_id","offset"])

    buf, paths, shard_id, n_ok = [], [], 0, 0
    def flush():
        nonlocal shard_id, buf, paths
        if not buf: return
        feats = torch.cat(buf).cpu()
        torch.save({"feats":feats}, out/f"feats_rank{rank}_{shard_id:06d}.pt")
        for i, p in enumerate(paths): writer.writerow([p, shard_id, i])
        shard_id += 1; buf.clear(); paths.clear()

    iterator = loader if rank!=0 else tqdm(loader, desc="Rank0")
    for batch in iterator:
        imgs, pths = zip(*batch)  # 여기서 깨진 파일이 있으면 __getitem__에서 이미 예외 발생해 전체 중단
        with torch.no_grad():
            feats = model.forward(list(imgs))
        buf.append(feats); paths.extend(pths); n_ok += len(pths)
        if len(paths) >= shard_size: flush()
    flush()

    json.dump({"rank":rank,"num_images":n_ok,"num_shards":shard_id},
              (out/f"meta_rank{rank}.json").open("w"))
    dist.barrier()
    if rank==0: print("✅ done")

def main():
    dist.init_process_group("nccl")
    extract_feats_ddp(
        in_root="/home/scpark/data/imagenet/train",
        out_dir="/home/scpark/data/imagenet_feats/train_pytorch",
        batch_size=512, shard_size=50000, num_workers=8
    )
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
