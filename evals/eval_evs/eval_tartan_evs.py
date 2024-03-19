import os
import math
import numpy as np
import os.path as osp

import torch
from devo.config import cfg

from utils.eval_utils import run_voxel, assert_eval_config
from utils.load_utils import voxel_iterator_parallel, voxel_iterator
from utils.eval_utils import log_results, log_results_loss, write_raw_results, compute_results, compute_median_results
from utils.transform_utils import transform_rescale_poses
from utils.viz_utils import viz_flow_inference

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, viz_flow=False, scale=1.0,
             rpg_eval=True, expname="", **kwargs):
    dataset_name = "tartanair_evs"

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")

    scenes = open(split_file).read().split()

    results_dict_scene, loss_dict_scene, figures = {}, {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        # scene_name = '_'.join(scene.split('/')[1:]).title() if "/P0" in scene else scene.title()
        # results_dict_scene[f"{dataset_name}/{scene_name}"] = [] # TODO use dataset_name/scene_name?
        #loss_dict_scene[f"{dataset_name}_{scene_name}"] = []
        results_dict_scene[scene] = []

        for trial in range(trials):

            # estimated trajectory
            datapath_val = os.path.join(datapath, scene.split("/")[0], scene.split("/")[2])
            scene_path = os.path.join(datapath_val, "evs_left", scene, "h5")
            traj_ref = osp.join(datapath_val, "image_left", scene, "pose_left.txt")

            # run the slam system
            if scale != 1.0:
                nH, nW = math.floor(scale * 480), math.floor(scale * 640) # TODO in tartan_rgb and tartan_frame
                kwargs.update({"scale": scale, "H": nH, "W": nW})
            traj_est, tstamps, flowdata = run_voxel(scene_path, config, net, viz=viz,
                                                 iterator=voxel_iterator(scene_path, timing=timing, stride=stride, scale=scale),
                                                 timing=timing, **kwargs, viz_flow=viz_flow)

            PERM = [1, 2, 0, 4, 5, 3, 6] # ned -> xyz
            # events between two adjacent frames t-1 and t are accumulated in event voxel t -> ignore first pose (t=0)
            traj_ref = np.loadtxt(traj_ref, delimiter=" ")[1::stride, PERM] # dtype="float32"
            if scale != 1.0:
                traj_ref = transform_rescale_poses(scale, torch.from_numpy(traj_ref)).data.numpy()

            FREQ = 50
            # do evaluation 
            data = (traj_ref, tstamps*1e6/FREQ, traj_est, tstamps*1e6/FREQ)
            data = (traj_ref, tstamps, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, rpg_eval=rpg_eval, stride=stride,
                                                                   expname=expname)
            
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
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/tartan/tartan_val.txt")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="viz_scorer")
    parser.add_argument('--scale', type=float, default=1.0, help='reduce computation')
    parser.add_argument('--dim_inet', type=int, default=384, help='channel dimension of hidden state')
    parser.add_argument('--dim_fnet', type=int, default=128, help='channel dimension of last layer fnet')
    parser.add_argument('--dim', type=int, default=32, help='channel dimension of first layer in extractor')
    parser.add_argument('--rpg_eval', action="store_true", help='advanced eval')

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_tartan_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)
    
    kwargs = {"scale": args.scale, "dim_inet": args.dim_inet, "dim_fnet": args.dim_fnet, "dim": args.dim}
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                        plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        stride=args.stride, **kwargs, viz_flow=args.viz_flow, rpg_eval=args.rpg_eval, expname=args.expname)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
