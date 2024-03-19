import os
import torch
from devo.config import cfg

from utils.load_utils import load_tumvie_traj, tumvie_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference

H, W = 720, 1280

@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None,
             stride=1, trials=1, plot=False, save=False, return_figure=False, viz=False, camID=2, timing=False, outdir=None, viz_flow=False):
    dataset_name = "tumvie_evs"
    assert camID == 2 or camID == 3
    assert H == 720 and W == 1280, "Resizing option not implemented yet (might be needed only later to train&eval quickly on TUMVIE due to large resolution)"

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")
        config.__setattr__('camID', camID)
        
    scenes = open(split_file).read().split()
    scenes = [s for s in scenes if '#' not in s]

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes): # TODO: parallel over scenes!
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        for trial in range(trials):
            datapath_val = os.path.join(datapath, scene)
            traj_hf_path = os.path.join(datapath_val, "mocap_data.txt")

            # run the slam system
            traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=tumvie_evs_iterator(datapath_val, camID=camID, stride=stride, timing=timing, dT_ms=25, H=H, W=W), 
                                          timing=timing, H=H, W=W, viz_flow=viz_flow)

            # load traj
            tss_traj_us, traj_hf = load_tumvie_traj(traj_hf_path)
 
            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride, camID_tumvie=camID, outdir=outdir,
                                                                   expname=args.expname)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)
            
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name, outfolder)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_tumvie.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/tumvie/tumvie_val.txt")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--camID', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--outdir', type=str, default="")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_tumvie_evs.py with config...")
    print(cfg)

    torch.manual_seed(1234)

    args.plot = True
    args.save_trajectory = True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                       plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, camID=args.camID, \
                        timing=args.timing, stride=args.stride, viz_flow=args.viz_flow)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
