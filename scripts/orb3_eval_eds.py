import numpy as np
import os
import argparse
import shutil, glob
import ruamel.yaml
from utils.eval_utils import log_results


def orb3_seq_eds(datadir, evaloutdir, orb3home=""):
    for seqname in sorted(os.listdir(datadir)):
        if not os.path.isdir(os.path.join(datadir, seqname)):
            continue
        orb3dir = os.path.join(datadir, seqname, "ORB3_calib0")
        assert os.path.isdir(orb3dir)  

        os.makedirs(evaloutdir, exist_ok=True)

        traj_gt = np.loadtxt(os.path.join(orb3dir, "groundtruth_ns.txt"), delimiter=",")
        tss_gt_us, traj_gt = traj_gt[:, 0] / 1000 , traj_gt[:, 1:]
        assert traj_gt.shape[1] == 7
        assert traj_gt.shape[0] > 50

        estfiles = sorted(glob.glob(os.path.join(orb3home, f"f_{seqname}*.txt")))
        if len(estfiles) == 0:
            print(f"Did not run ORB3 yet for {seqname}")
            return
        
        for trial, estfn in enumerate(estfiles):
            shutil.copy(estfn, orb3dir)
            traj_est = np.loadtxt(estfn, delimiter=" ")
            tss_est_us, traj_est = traj_est[:, 0] / 1000 , traj_est[:, 1:]

            cfg = None 
            args = None 
            all_results = []
            results_dict_scene, figures = {}, {}
            results_dict_scene[seqname] = []
            data = (traj_gt, tss_gt_us, traj_est, tss_est_us)
            hyperparam = (None, None, f"{seqname}_Trial{trial}", seqname, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                plot=True, save=True, return_figure=True, stride=1, calib1_eds=False, outdir=evaloutdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval orb3")
    parser.add_argument(
        "--datadir", help="Orbdi.", default=""
    )
    parser.add_argument(
        "--evaldir", help="Eavl Dir.", default=""
    )
    args = parser.parse_args()

    orb3_seq_eds(args.datadir, args.evaldir)