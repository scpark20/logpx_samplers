import os, torch, torch.distributed as dist
from datetime import timedelta
rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", init_method="env://", timeout=timedelta(seconds=60))
print(f"hello from rank {rank} on cuda:{local_rank} ({torch.cuda.get_device_name()})", flush=True)
dist.barrier(); dist.destroy_process_group()