import os
import numpy as np
import os.path as osp

import torch
from devo.config import cfg

from utils.load_utils import load_tumvie_traj, video_iterator, load_intrinsics_tumvie
from utils.eval_utils import assert_eval_config, run_rgb
from utils.eval_utils import log_results, write_raw_results, compute_median_results

H, W = 720, 1280

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=True, viz=False, camID=0, timing=False):
    dataset_name = "tumvie_evs_viz"
    assert camID == 2 or camID == 3
    side = "left" if camID == 2 else "right"
    assert H == 720 and W == 1280, "Resizing option not implemented yet (might be needed only later to train&eval quickly on TUMVIE due to large resolution)"

    if config is None:
        config = cfg
        config.merge_from_file("config/default_rgb.yaml")
        config.__setattr__('camID', camID)

    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []
        
        for trial in range(trials):
            datapath_val = os.path.join(datapath, scene)
            scene_path = os.path.join(datapath_val, f"all_evs_{side}_undist_viz")
            traj_hf_path = osp.join(datapath_val, "mocap_data.txt")
            intrinsics = load_intrinsics_tumvie(datapath_val, camID=camID)
            tss_file = os.path.join(datapath_val, f"{side}_images", f"image_timestamps_{side}.txt")

            # run the slam system
            traj_est, tstamps, flows = run_rgb(scene_path, config, net, viz=viz, 
                                        iterator=video_iterator(scene_path, tss_file, timing=timing, ext="_undist.png", stride=stride, intrinsics=intrinsics), 
                                        timing=timing, H=H, W=W)

            tss_traj_us, traj_hf = load_tumvie_traj(traj_hf_path)

            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride, camID_tumvie=camID,
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
    parser.add_argument('--config', default="config/default_rgb.yaml") # for DPVO settings
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="dpvo.pth")
    parser.add_argument('--val_split', type=str, default="splits/tumvie/tumvie_val.txt")
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--camID', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_tumvie.py with config...")
    print(cfg)

    torch.manual_seed(1234)

    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, camID=args.camID, timing=args.timing, stride=args.stride)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])

