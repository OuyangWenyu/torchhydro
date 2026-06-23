import os
import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl"):
    """
    Initialize torch.distributed from torchrun env variables.
    Return (is_dist, rank, world_size, local_rank).
    """
    if "RANK" not in os.environ:
        # 没有用 torchrun，按单机单卡走
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
