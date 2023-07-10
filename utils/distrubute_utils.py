import os

import torch
import torch.distributions as dist


def init_distrubuted_mode(opt):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt.rank = int(os.environ['RANK'])
        opt.world_size = int(os.environ['WORLD_SIZE'])
        opt.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        opt.rank = int(os.environ['SLURM_PROCID'])
        opt.gpu = opt.rank % torch.cuda.device_count()
    else:
        print('device cannot setup distributed mode')
        opt.distributed = False
        return

    opt.distrubuted = True

    torch.cuda.set_device(opt.gpu)
    opt.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(opt.rank, opt.dist_url), flush=True)

    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url, world_size=opt.world_size,
                            rank=opt.rank)
    dist.barrier()


def clean_up():
    dist.destroy_process_group()


def is_dist_availble_or_initial():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_availble_or_initial():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_availble_or_initial():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
