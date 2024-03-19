import os
import math
import numpy as np
import os.path as osp

import torch
from devo.config import cfg

from utils.eval_utils import run_rgb, assert_eval_config
from utils.load_utils import video_iterator
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference
import glob

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, viz_flow=False, scale=1.0, rpg_eval=True,
             **kwargs):
    dataset_name = "tartanair_e2v"

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    scenes = open(split_file).read().split()

    results_dict_scene, loss_dict_scene, figures = {}, {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            # estimated trajectory
            difficulty = scene.split("/")[2]
            seqNum = scene.split("/")[-1]
            seqName = scene.split("/")[0]
            datapath_val = os.path.join(datapath, seqName, difficulty)
            scene_path = os.path.join(datapath_val, "e2v", seqName, seqName, difficulty, seqNum, "e2calib/")
            scene_tartan = os.path.join("", seqName, difficulty, "image_left", seqName, seqName, difficulty, seqNum)
            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            traj_ref = np.loadtxt(osp.join(scene_tartan, 'pose_left.txt'), delimiter=' ')[1::stride, PERM]
            assert traj_ref.shape[0] == len(glob.glob(os.path.join(scene_path,"*png"))) or stride != 1


            # run the slam system
            traj_est, tstamps, flowdata = run_rgb(scene_path, config, net, viz=viz, iterator=
                                        video_iterator(scene_path, timing=timing, stride=stride), timing=timing, viz_flow=viz_flow)
            

            FREQ = 50  
            # do evaluation 
            data = (traj_ref, tstamps/FREQ*1e6, traj_est, tstamps/FREQ*1e6)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, rpg_eval=rpg_eval, stride=stride)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)
            
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    # results_dict = compute_results(results_dict_scene, all_results, loss_dict_scene)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/default.yaml") # for DPVO settings
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="")
    parser.add_argument('--val_split', type=str, default="splits/tartan/tartan_val.txt")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--scale', type=float, default=1.0, help='reduce computation')
    parser.add_argument('--dim_inet', type=int, default=384, help='channel dimension of hidden state') 
    parser.add_argument('--dim_fnet', type=int, default=128, help='channel dimension of last layer fnet') 
    parser.add_argument('--dim', type=int, default=32, help='channel dimension of first layer in extractor')
    parser.add_argument('--rpg_eval', action="store_true", help='advanced eval') # TODO adapt rest evals

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_tartan_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)
    
    kwargs = {"scale": args.scale, "dim_inet": args.dim_inet, "dim_fnet": args.dim_fnet, "dim": args.dim}
    args.plot = True
    args.save_trajectory = True
    args.return_figs = True
    args.rpg_eval = True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        stride=args.stride,
                       **kwargs, viz_flow=args.viz_flow, rpg_eval=args.rpg_eval)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
