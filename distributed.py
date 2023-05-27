import os
import torch

def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def init_distributed_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    if is_using_distributed():
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
        )
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.distributed = True

    else:
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=1,
            rank=0,
        )

    if torch.cuda.is_available():
        if args.distributed:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    return device
