import os
import numpy as np
import os.path as osp

import torch
from devo.config import cfg

from utils.eval_utils import run_voxel, assert_eval_config
from utils.load_utils import voxel_iterator_parallel
from utils.eval_utils import log_results, write_raw_results, compute_median_results

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False):
    dataset_name = "tartanair_evs"

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):

            # estimated trajectory
            datapath_val = os.path.join(datapath, scene.split("/")[0], scene.split("/")[2])
            scene_path = os.path.join(datapath_val, "evs_left", scene, "h5")
            traj_ref = osp.join(datapath_val, "image_left", scene, "pose_left.txt")

            # run the slam system
            traj_est, tstamps, flows = run_voxel(scene_path, config, net, viz=viz, iterator=voxel_iterator_parallel(scene_path, timing=timing, stride=stride), timing=timing, viz_flow=True)

            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            # events between two adjacent frames t-1 and t are accumulated in event voxel t -> ignore first pose (t=0)
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[1::stride, PERM]

            # do evaluation 
            data = (traj_ref, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                   expname=args.expname)
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/default.yaml") # for DPVO settings
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="edpvo.pth")
    parser.add_argument('--val_split', type=str, default="splits/tartan/tartan_val.txt")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_tartan_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)
   
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, stride=args.stride)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
