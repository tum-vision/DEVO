import cv2
import os
import numpy as np
from collections import OrderedDict
import contextlib

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from devo.data_readers.factory import dataset_factory

from devo.lietorch import SE3
from devo.logger import Logger
import torch.nn.functional as F

from devo.net import VONet
from devo.enet import eVONet

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.viz_utils import plot_flow_tartan_train


def setup_ddp(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
        
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.gpu_num,                              
    	rank=rank)

    torch.manual_seed(0)
    torch.cuda.set_device(rank)


def plot_flow_train_loader(rank, args):
    """ main training loop """
    
    # coordinate multiple GPUs
    if args.ddp:
        setup_ddp(rank, args)

    # fetch dataset
    if args.evs:
        db = dataset_factory(['tartan_evs'], datapath=args.datapath, n_frames=args.n_frames,
                             fgraph_pickle=args.fgraph_pickle, train_split=args.train_split, 
                             val_split=args.val_split, strict_split=False, return_fname=True)
    else:
        db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames,
                             fgraph_pickle=args.fgraph_pickle, train_split=args.train_split, 
                             val_split=args.val_split, strict_split=False, return_fname=True, aug=False)

    # setup dataloader
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            db, shuffle=True, num_replicas=args.gpu_num, rank=rank)
        train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=4)
    else:
        train_loader = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=4)

    # Initial VOnet
    net = VONet() if not args.evs else eVONet(patch_selector=args.patch_selector.lower())
    # net.train()
    net.cuda()
    
    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)

    total_steps = 0
    if args.checkpoint is not None and args.checkpoint != '':
        print(f"Loading from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        if 'model_state_dict' in checkpoint:
            if args.ddp:
                net.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                net.load_state_dict(checkpoint['model_state_dict'])
        else:
            # legacy
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
            net.load_state_dict(new_state_dict, strict=False)
        if 'steps' in checkpoint:
            total_steps = checkpoint['steps']

    
    for data_blob in train_loader: # plotting 1 epoch
        images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob[:-1]] # (B,n_frames,C,H,W)#
        images_ = torch.clone(images)

        # fix poses to gt for first 1k steps
        so = False

        poses = SE3(poses).inv()
        traj = net(images, poses, disps, intrinsics, M=1024, STEPS=args.iters, structure_only=so, plot_patches=True, patches_per_image=args.patches_per_image) # list [valid, p_ij, p_ij_gt, poses, poses_gt, kl] of iters (update operator)
        patch_data = traj.pop()

        seq = data_blob[-1][0]
        seq = seq.split("/")[-3:]
        seq = os.path.join(*seq).replace("/", "_")
        plot_flow_tartan_train(images_, patch_data, evs=args.evs, outdir=f"../viz/flow_tartan_train/name_{args.name}/{seq}/step_{total_steps}/")
        total_steps += 1

    if args.ddp:
        dist.destroy_process_group()


def assert_config(args):
    assert os.path.isfile(args.config)
    assert os.path.isdir(args.datapath)
    
    assert args.gpu_num > 0 and args.gpu_num <= 10
    if args.gpu_num > 1:
        assert args.ddp
    assert args.batch > 0 and args.batch <= 1024
    assert args.steps > 0 and args.steps <= 2500000
    assert args.iters >= 2 and args.iters <= 50
    assert args.lr > 0 and args.lr < 1
    assert args.n_frames > 7 and args.n_frames < 100 #  The first 8 frames are used for initialization while the next n_frames-8 frames are added one at a time
    assert args.pose_weight >= 0 and args.pose_weight <= 100 and args.flow_weight >= 0 and args.flow_weight <= 100

    if args.checkpoint is not None and args.checkpoint != '':
        assert os.path.isfile(args.checkpoint)
        assert ".pth" in args.checkpoint or ".pt" in args.checkpoint 
    if args.fgraph_pickle is not None and args.fgraph_pickle != '':
        assert os.path.isfile(args.fgraph_pickle)
        assert ".pickle" in args.fgraph_pickle
    
    assert os.path.isfile(args.train_split)
    assert os.path.isfile(args.val_split)

if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        default='config/DPVO_base.conf',
        is_config_file=True,
        help='config file path',
    )
    parser.add_argument('--name', '--expname', default='bla', help='name your experiment')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint to restore')
    parser.add_argument('--fgraph_pickle', type=str, default="", help='precomputed frame graph (copied to expdir)')
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=240000, help='total steps')
    parser.add_argument('--iters', type=int, default=2, help='iterations of update operator per edge in patch graph') # default: 18
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--evs', action='store_true', help='event-based DPVO')
    parser.add_argument('--train_split', type=str, default="splits/tartan/tartan_train.txt", help='train seqs (line separated).')
    parser.add_argument('--val_split', type=str, default="splits/tartan/tartan_default_val.txt", help='val seqs (line separated)')
    parser.add_argument('--ddp', action='store_true', help='use multi-gpu')
    parser.add_argument('--gpu_num', type=int, default=1, help='distributed over more gpus')
    parser.add_argument('--port', default="12345", help='free port for master node')
    parser.add_argument('--profiler', action='store_true', help='enable autograd profiler')
    parser.add_argument('--patches_per_image', type=int, default=80, help='number of patches per image')
    parser.add_argument('--patch_selector', type=str, default="random", help='name of patch selector (gradient,...)')
    
    args = parser.parse_known_args()[0]
    assert_config(args)
    print("----------")
    print("Print config")
    print(args)
    print("----------")
    print(parser.format_values())
    print("----------")
    
    # TODO Debugging
    os.system("nvidia-smi")
    print(torch.version.cuda)

    cmd = "echo $HOSTNAME"
    os.system(cmd)

    cmd = "ulimit -n 2000000"
    os.system(cmd)

    if not os.path.isdir(f'../checkpoints/{args.name}'):
        os.makedirs(f'../checkpoints/{args.name}')
    
    if args.ddp:
        mp.spawn(plot_flow_train_loader, nprocs=args.gpu_num, args=(args,))
    else:
        plot_flow_train_loader(0, args)
