import os
import torch
import torch.distributed as dist

def init():
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, dist.get_rank() if world > 1 else 0, world

def sum_(x):
    if dist.is_initialized():
        dist.all_reduce(x)
    return x

def close():
    if dist.is_initialized():
        dist.destroy_process_group()
