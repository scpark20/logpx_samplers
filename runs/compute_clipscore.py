"""Script for calculating the CLIP Score."""

import os
import csv
import json
import click
import tqdm
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp
import runs.dataset as dataset
import open_clip
from torchvision import transforms
import numpy as np

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate CLIP score.
    python clip_score.py calc --images=path/to/images
    Multi-GPU runs are handled automatically by spawning one process per visible GPU.
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=10000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=250, show_default=True)
@click.option('--prompts-npz', 'prompts_npz', help='Path to prompts npz used during sampling',      type=str, default='prompts/mscoco2014_val.npz', show_default=True)

@torch.no_grad()
def calc(image_path, batch, num_expected=10000, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=None, prompts_npz='prompts/mscoco2014_val.npz'):
    mp.set_start_method('spawn', force=True)

    world_size = torch.cuda.device_count()
    if world_size < 1:
        world_size = 1

    if world_size > 1:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        mp.spawn(
            _worker,
            nprocs=world_size,
            args=(world_size, image_path, batch, num_expected, seed, num_workers, prefetch_factor, prompts_npz),
        )
    else:
        _worker(0, world_size, image_path, batch, num_expected, seed, num_workers, prefetch_factor, prompts_npz)
@torch.inference_mode()
def _worker(rank, world_size, image_path, batch, num_expected, seed, num_workers, prefetch_factor, prompts_npz):
    is_multi = world_size > 1
    # Resolve device first (before process group init)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    if is_multi:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch_dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    def print0(*args, **kwargs):
        if (not torch_dist.is_initialized()) or rank == 0:
            print(*args, **kwargs)

    # List images.
    if rank == 0:
        print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=None, random_seed=seed)

    # Load prompts from NPZ used during sampling
    sample_prompts = np.load(prompts_npz)['arr_0'].tolist()
    if num_expected is not None:
        sample_prompts = sample_prompts[:num_expected]

    if rank == 0:
        print0(f"prompts_npz: {prompts_npz}")
        print0("len(sample_prompts): ", len(sample_prompts))
        print0("num_expected: ", num_expected)

    # Load CLIP model
    if rank == 0:
        print0('Loading CLIP-ViT-g-14 model...')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-g-14')
    model.to(device)
    model.eval()

    # Determine evaluation ordering and indices to align with prompts
    effective_len = min(len(dataset_obj), len(sample_prompts))
    try:
        import os as _os
        ordered_pairs = []
        for idx, fname in enumerate(getattr(dataset_obj, '_image_fnames', [])):
            stem = _os.path.splitext(_os.path.basename(fname))[0]
            ordered_pairs.append((int(stem), idx))
        ordered_pairs.sort(key=lambda x: x[0])
        ordered_indices = torch.tensor([idx for _, idx in ordered_pairs[:effective_len]], dtype=torch.long)
        idx_to_key = {idx: key for key, idx in ordered_pairs[:effective_len]}
        if ordered_indices.numel() < effective_len:
            # fallback if not enough numeric names parsed
            ordered_indices = torch.arange(effective_len, dtype=torch.long)
            idx_to_key = {i: i for i in range(effective_len)}
    except Exception:
        ordered_indices = torch.arange(effective_len, dtype=torch.long)
        idx_to_key = {i: i for i in range(effective_len)}

    # Divide images into per-rank batches using the ordered indices
    max_batch_size = batch
    num_batches = ((effective_len - 1) // (max_batch_size * world_size) + 1) * world_size
    all_batches = ordered_indices.tensor_split(num_batches)
    rank_batches = all_batches[rank :: world_size]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_sampler=rank_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Accumulate partial sums
    if rank == 0:
        print0(f'Calculating statistics for {len(dataset_obj)} images...')
    partial_sum = 0.0
    partial_count = 0
    to_pil = transforms.ToPILImage()
    for batch_idx, (images, _) in enumerate(tqdm.tqdm(data_loader, unit='batch', disable=(rank != 0))):
        batch_indices = rank_batches[batch_idx].tolist()
        prompts = [sample_prompts[idx_to_key[i]] for i in batch_indices]
        prompts = [str(p) for p in prompts]

        images = torch.stack([preprocess(to_pil(img)) for img in images], dim=0).to(device)
        text = tokenizer(prompts).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Direct pairwise cosine similarity for aligned pairs
        sd_clip_score = 100 * (image_features * text_features).sum(dim=-1)
        partial_sum += sd_clip_score.sum().item()
        partial_count += sd_clip_score.shape[0]

    stats = torch.tensor([partial_sum, float(partial_count)], dtype=torch.float64, device=device)
    if torch_dist.is_initialized():
        torch_dist.all_reduce(stats, op=torch_dist.ReduceOp.SUM)

    if (not torch_dist.is_initialized()) or rank == 0:
        total_sum, total_count = stats.tolist()
        avg_clip_score = total_sum / max(total_count, 1.0)
        print(f"CLIP score: {avg_clip_score}")

    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------