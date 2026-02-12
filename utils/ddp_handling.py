import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def ddp_setup():
    """
        Function to setup the Distributed Data Parallel (DDP) Environment
    """
    dist.init_process_group(backend="nccl")



def ddp_cleanup():
    """
        Function to cleanup the Distributed Data Parallel (DDP) Environment
    """
    dist.destroy_process_group()


def ddp_is_running():
    if "RANK" in os.environ: 
        return True
    if dist.is_available() and dist.is_initialized(): 
        return True
    return False
