import os, torch, torch.distributed as dist
from datetime import timedelta

def main():
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)  # ✅ 명시

    # ✅ device_id를 명시하면 NCCL가 rank→GPU 매핑을 확실히 압니다.
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=120),
        device_id=local_rank,          # ← 여기!
    )

    print(f"[rank {rank}/{world_size}] hello on cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)})", flush=True)

    dist.barrier(device_ids=[local_rank])     # ← barrier도 내 GPU를 명시
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
