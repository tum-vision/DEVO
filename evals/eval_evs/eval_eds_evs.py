import os
import torch
from devo.config import cfg

from utils.load_utils import load_eds_traj, eds_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference

H, W = 480, 640

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, calib1=False, timing=False, viz_flow=False):
    dataset_name = "eds_evs"

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")
        config.__setattr__('calib1', calib1)
        
    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            # estimated trajectory
            datapath_val = os.path.join(datapath, scene)
            traj_hf_path = os.path.join(datapath_val, "stamped_groundtruth_us.txt")

            # run the slam system
            traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=eds_evs_iterator(datapath_val, calib1=calib1, stride=stride, timing=timing, H=H, W=W),
                                          timing=timing, H=H, W=W, viz_flow=viz_flow)

            # load traj
            tss_traj_us, traj_hf = load_eds_traj(traj_hf_path)

            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride, calib1_eds=calib1,
                                                                   expname=args.expname)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)
            
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
    parser.add_argument('--config', default="config/eval_eds.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/eds/eds_val.txt")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--calib1', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print(f"Running eval_eds_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)

    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, calib1=args.calib1, \
                        timing=args.timing, stride=args.stride, viz_flow=args.viz_flow)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
