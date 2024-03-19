import os
import torch
from devo.config import cfg

from utils.load_utils import load_eds_traj

from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.eval_utils import run_rgb, assert_eval_config
from utils.load_utils import video_iterator


@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, calib1=False, timing=False):
    dataset_name = "eds_evs_viz"

    if not calib1:
        intrinsics = [713.6517944335938, 737.5368041992188, 288.29385382423607, 226.97368855930836] 
    elif calib1:
        intrinsics = [704.6842041015625, 729.3246459960938, 286.39991203466707, 227.79612335843012]

    if config is None:
        config = cfg
        config.merge_from_file("config/default_rgb.yaml")
        config.__setattr__('calib1', calib1)
        
    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            datapath_val = os.path.join(datapath, scene)
            scene_path = os.path.join(datapath_val, "viz_h5")
            traj_hf_path = os.path.join(datapath_val, "stamped_groundtruth_us.txt")
            tss_file = os.path.join(datapath_val, "images_timestamps_us.txt")

            # run the slam system
            traj_est, tstamps, flows = run_rgb(scene_path, config, net, viz=viz, iterator=video_iterator(scene_path, tss_file, intrinsics=intrinsics, timing=timing, stride=stride), timing=timing)

            # load traj
            tss_traj_us, traj_hf = load_eds_traj(traj_hf_path)
 
            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride, calib1_eds=calib1,
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
    parser.add_argument('--val_split', type=str, default="splits/eds/eds_val.txt")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--calib1', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_eds_evs_frame.py with config...")
    print(cfg)

    torch.manual_seed(1234)

    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, calib1=args.calib1, timing=args.timing, stride=args.stride)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
