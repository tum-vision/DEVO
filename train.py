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

# from devo.net import VONet # TODO add net.py
from devo.enet import eVONet
from devo.selector import SelectionMethod

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.viz_utils import plot_patch_following_all, plot_patch_following, plot_patch_depths_all

DEBUG_PLOT_PATCHES = False

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

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    """ compute optimal scaling (SIM3) that minimizing RMSD between two sets of points """
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def train(rank, args):
    """ main training loop """
    
    # coordinate multiple GPUs
    if args.ddp:
        setup_ddp(rank, args)

    # fetch dataset
    if args.evs:
        db = dataset_factory(['tartan_evs'], datapath=args.datapath, n_frames=args.n_frames,
                             fgraph_pickle=args.fgraph_pickle, train_split=args.train_split,
                             val_split=args.val_split, strict_split=False, sample=True, return_fname=True, scale=args.scale)
    elif args.e2vid:
        db = dataset_factory(['tartan_e2vid'], datapath=args.datapath, n_frames=args.n_frames,
                             fgraph_pickle=args.fgraph_pickle, train_split=args.train_split,
                             val_split=args.val_split, strict_split=False, sample=True, return_fname=True, scale=args.scale)  
    else:
        db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames,
                             fgraph_pickle=args.fgraph_pickle, train_split=args.train_split, 
                             val_split=args.val_split, strict_split=False, sample=True, return_fname=True, scale=args.scale)

    # setup dataloader
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            db, shuffle=True, num_replicas=args.gpu_num, rank=rank)
        train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=4)
    else:
        train_loader = DataLoader(db, batch_size=args.batch, shuffle=True, num_workers=4)

    # Initial VOnet
    kwargs_net = {"dim_inet": args.dim_inet, "dim_fnet": args.dim_fnet, "dim": args.dim}
    net = VONet(**kwargs_net, patch_selector=args.patch_selector.lower()) if not args.evs else \
    eVONet(**kwargs_net, patch_selector=args.patch_selector.lower(), norm=args.norm, randaug=args.randaug)

    net.train()
    net.cuda()
    P = net.P # patch size (squared)
        
    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    total_steps = 0
    if args.checkpoint is not None and args.checkpoint != '':
        print(f"Loading from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        model = net.module if args.ddp else net
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # legacy
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                new_state_dict[k.replace('module.', '')] = v
            # with RGB pretraining
            update_dict = { k: v for k, v in new_state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape }
            print(update_dict.keys())
            state = model.state_dict()
            state.update(update_dict)
            # keys with different shape: ['patchify.fnet.conv1.weight', 'patchify.inet.conv1.weight']
            # corresponding values: [torch.Size([32, 3, 7, 7]), torch.Size([32, 3, 7, 7])]
            model.load_state_dict(state, strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'steps' in checkpoint:
            total_steps = checkpoint['steps']

    if rank == 0:
        logger = Logger(args.name, scheduler, args.gpu_num * total_steps, args.gpu_num)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=1997, wait=1, warmup=1, active=2, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'../runs/{args.name}', rank),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) if args.profiler else contextlib.nullcontext() as prof:
        while True:
            for data_blob in train_loader:
                scene_id = data_blob.pop()
                images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob] # images: (B,n_frames,C,H,W), poses: (B,n_frames,7), disps: (B,n_frames,H,W) (all float32)
                optimizer.zero_grad() # TODO set_to_none=True

                # fix poses to gt for first 1k steps
                so = total_steps < (1000 // args.gpu_num) and (args.checkpoint is None or args.checkpoint == "")

                poses = SE3(poses).inv() # [simon]: this does c2w -> w2c (which dpvo predicts&stores internally)
                traj = net(images, poses, disps, intrinsics, M=1024, STEPS=args.iters, structure_only=so, plot_patches=DEBUG_PLOT_PATCHES, patches_per_image=args.patches_per_image)
                # list [valid, p_ij, p_ij_gt, poses, poses_gt, kl] of iters (update operator)
                if DEBUG_PLOT_PATCHES:
                    patch_data = traj.pop()
                # valid (B,edges)
                # p_ij, p_ij_gt (B,edges,P,P,2), float32
                # poses, poses_gt (SE3.data of dim (B,n_frames,7))

                # Compute loss and metrics
                loss = 0.0
                pose_loss = 0.0
                flow_loss = 0.0
                scores_loss = torch.as_tensor(0.0)
                for i, data in enumerate(traj):
                    if args.patch_selector == SelectionMethod.SCORER:
                        (v, x, y, P1, P2, kl, scores, v_full, x_full, y_full, ba_weights, kk, dij) = data
                    else:
                        (v, x, y, P1, P2, kl) = data
                    valid = (v > 0.5).reshape(-1) 
                    e = (x - y).norm(dim=-1) # residual (p_ij - p_ij_gt)
                    ef = e.reshape(-1, P**2)[valid].min(dim=-1).values # e.shape: (B*edges,P^2) -> (B*edges)
                    flow_loss = ef.mean()
                    
                    start_scorer = (i == (len(traj)-1)) and (total_steps // args.gpu_num) >= 1e+4
                    start_scorer = (i == (len(traj)-1))

                    if args.patch_selector == SelectionMethod.SCORER and start_scorer:
                        import math
                        valid_full = (v_full >= 0.5).reshape(-1)

                        kk = kk[valid_full]
                        e_full = (x_full - y_full).norm(dim=-1) # residual (p_ij - p_ij_gt)
                        e_full = e_full.reshape(-1, P**2)[valid_full].min(dim=-1).values # e.shape: (B*edges,P^2) -> (B*edges)
                        # scorer (flow only)
                        # scores_loss = (scores.view(-1)[kk] * e_full).mean()
                        # scorer (flow + ba)
                        scores_loss = ((-0.5*(ba_weights.view(-1,2)[valid_full].mean(dim=-1)).log() + 1) * scores.view(-1)[kk] * e_full).mean()
                        
                        scores = torch.max(scores, torch.as_tensor(1e-6))  
                        scores = -scores.log()
                        scores_loss += scores.mean()
                    else:
                        scores_loss = torch.as_tensor(0.0)
                    
                    N = P1.shape[1] # number frames (n_frames)
                    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                    ii = ii.reshape(-1).cuda()
                    jj = jj.reshape(-1).cuda()

                    k = ii != jj # not same frame
                    ii = ii[k]
                    jj = jj[k]

                    P1 = P1.inv() # because of (SE3(poses).inv())
                    P2 = P2.inv() # w2c => c2w

                    t1 = P1.matrix()[...,:3,3] # predicted translation # TODO with detach()?
                    t2 = P2.matrix()[...,:3,3] # gt translation # TODO with detach()?

                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0) # how to handle batch greater than 1?
                    P1 = P1.scale(s.view(1, 1))

                    dP = P1[:,ii].inv() * P1[:,jj] # predicted poses from frame i to j (G_ij)
                    dG = P2[:,ii].inv() * P2[:,jj] # gt poses from frame i to j (T_ij)

                    e1 = (dP * dG.inv()).log() # poses loss for each pair of frames
                    tr = e1[...,0:3].norm(dim=-1) # tx ty tz
                    ro = e1[...,3:6].norm(dim=-1) # qx qy qz

                    loss += args.flow_weight * flow_loss
                    loss += args.scores_weight * scores_loss
                    pose_loss = tr.mean() + ro.mean()
                    if not so and i >= 2:
                        loss += args.pose_weight * pose_loss

                if rank == 0 and DEBUG_PLOT_PATCHES:
                    plot_patch_following_all(images, patch_data, evs=args.evs, outdir=f"../viz/patches_all/name_{args.name}/step_{total_steps}/")
                    plot_patch_following(images, patch_data, evs=args.evs, outdir=f"../viz/patches/name_{args.name}/step_{total_steps}/")
                    plot_patch_depths_all(images, patch_data, disps, evs=args.evs, outdir=f"../viz/patches_depths_all/name_{args.name}/step_{total_steps}/")
                
                if torch.isnan(loss):
                    print(f"nan at {total_steps}: {scene_id}")
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()   
                scheduler.step()

                total_steps += 1
                
                metrics = {
                    "loss/train": loss.item(),                          # total loss value (pose_weight * pose_loss + flow_weight * flow_floss)
                    "loss/pose_train": pose_loss.item(),                # pose loss (rotation + translation)
                    "loss/rotation_train": ro.float().mean().item(),    # rotation loss
                    "loss/translation_train": tr.float().mean().item(), # translation loss
                    "loss/flow_train": flow_loss.item(),                # flow loss
                    "loss/scores_train": scores_loss.item(),            # scores loss
                    "px1": (e < .25).float().mean().item(),             # AUC
                    "r1": (ro < .001).float().mean().item(),            #
                    "r2": (ro < .01).float().mean().item(),             #
                    "t1": (tr < .001).float().mean().item(),            #
                    "t2": (tr < .01).float().mean().item(),             #
                }

                if rank == 0:
                    logger.push(metrics)

                if total_steps % 10000 == 0:
                    torch.cuda.empty_cache()

                    if rank == 0:
                        PATH = '../checkpoints/%s/%06d.pth' % (args.name, args.gpu_num * total_steps)
                        torch.save({
                            'steps': total_steps,
                            'model_state_dict': net.module.state_dict() if args.ddp else net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()}, PATH)
                        
                        if args.eval:
                            if args.evs:
                                from evals.eval_evs.eval_tartan_evs import evaluate as eval_tartan_evs
                                val_results, val_figures = eval_tartan_evs(None, None, net.module if args.ddp else net, total_steps,
                                                                        args.datapath, args.val_split, return_figure=True, plot=True, rpg_eval=False,
                                                                        scale=args.scale, expname=args.name, **kwargs_net)
                            else:
                                from evals.eval_rgb.eval_tartan import evaluate as eval_tartan
                                val_results, val_figures = eval_tartan(None, None, net.module if args.ddp else net, total_steps,
                                                                   args.datapath, args.val_split, return_figure=True, plot=True, rpg_eval=False,
                                                                   scale=args.scale, expname=args.name)
                            logger.write_dict(val_results)
                            logger.write_figures(val_figures)
                    torch.cuda.empty_cache()
                    net.train()
                
                if args.profiler:
                    prof.step()
            
                if total_steps >= args.steps:
                    break
            else:
                continue
            break
    
    if rank == 0:
        logger.close()
    if args.ddp:
        dist.destroy_process_group()


def assert_config(args):
    assert os.path.isfile(args.config)
    assert os.path.isdir(args.datapath)
    
    assert args.gpu_num > 0 and args.gpu_num <= 10
    if args.gpu_num > 1:
        assert args.ddp
    assert args.batch > 0 and args.batch <= 1024
    assert args.steps > 0 and args.steps <= 4800000
    assert args.steps % args.gpu_num == 0
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

    if args.ddp:
        assert DEBUG_PLOT_PATCHES == False

    if args.e2vid:
        assert args.evs == False

if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        default='config/DEVO_base.conf',
        is_config_file=True,
        help='config file path',
    )
    parser.add_argument('--name', '--expname', default='bla', help='name your experiment')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint to restore')
    parser.add_argument('--fgraph_pickle', type=str, default="TartanAirEVS.pickle", help='precomputed frame graph (copied to expdir)')
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=240000, help='total steps')
    parser.add_argument('--iters', type=int, default=18, help='iterations of update operator per edge in patch graph') # default: 18
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    parser.add_argument('--scores_weight', type=float, default=0.05)
    parser.add_argument('--evs', action='store_true', help='event-based DPVO')
    parser.add_argument('--e2vid', action='store_true', help='baseline on e2v reconstruction')
    parser.add_argument('--eval', action='store_true', help='enable eval on TartanAir')
    parser.add_argument('--train_split', type=str, default="splits/tartan/tartan_train.txt", help='train seqs (line separated).')
    parser.add_argument('--val_split', type=str, default="splits/tartan/tartan_default_val.txt", help='val seqs (line separated)')
    parser.add_argument('--ddp', action='store_true', help='use multi-gpu')
    parser.add_argument('--gpu_num', type=int, default=1, help='distributed over more gpus')
    parser.add_argument('--port', default="12348", help='free port for master node')
    parser.add_argument('--profiler', action='store_true', help='enable autograd profiler')
    parser.add_argument('--scale', type=float, default=1.0, help='reduce computation')
    parser.add_argument('--dim_inet', type=int, default=384, help='channel dimension of hidden state')
    parser.add_argument('--dim_fnet', type=int, default=128, help='channel dimension of last layer fnet')
    parser.add_argument('--dim', type=int, default=32, help='channel dimension of first layer in extractor')
    parser.add_argument('--patches_per_image', type=int, default=80, help='number of patches per image')
    parser.add_argument('--patch_selector', type=str, default="random", help='name of patch selector (random, gradient, scorer)')
    parser.add_argument('--norm', type=str, default="rescale", help='name of norm (evs only) (none, rescale, standard)')
    parser.add_argument('--randaug', action='store_true', help='enable randAug (evs only)')
    
    args = parser.parse_known_args()[0]
    assert_config(args)
    print("----------")
    print("Print config")
    print(args)
    print("----------")
    print(parser.format_values())
    print("----------")
    args.steps = args.steps // args.gpu_num 
    
    os.system("nvidia-smi")
    print(torch.version.cuda)

    cmd = "echo $HOSTNAME"
    os.system(cmd)

    cmd = "ulimit -n 2000000"
    os.system(cmd)

    if not os.path.isdir(f'../checkpoints/{args.name}'):
        os.makedirs(f'../checkpoints/{args.name}')
    
    if args.ddp:
        mp.spawn(train, nprocs=args.gpu_num, args=(args,))
    else:
        train(0, args)
