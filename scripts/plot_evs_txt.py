import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import glob
from utils.event_utils import *
from utils.viz_utils import render
from utils.load_utils import read_ecd_tss


def main():
    parser = argparse.ArgumentParser(description="Plotting event chunks from h5 file")
    parser.add_argument(         
        "--infile", help="Infile", default=""
    )
    parser.add_argument("--H",  default=260, type=int)
    parser.add_argument("--W", default=346, type=int)
    args = parser.parse_args()

    assert "events.txt" in args.infile
    outvizfolder = os.path.join(os.path.dirname(args.infile), "viz_evs_txt")
    os.makedirs(outvizfolder, exist_ok=True)

    indir = os.path.dirname(args.infile)
    
    W = args.W
    H = args.H

    evs = np.asarray(np.loadtxt(args.infile, delimiter=" ", skiprows=1)) # (N, 4) with [ts_sec, x, y, p]
    evs[:, 0] = evs[:, 0] * 1e6

    tss_imgs_us = read_ecd_tss(os.path.join(indir, "images.txt"), idx=1)
    
    pbar = tqdm.tqdm(total=len(tss_imgs_us)-1)
    for i in range(len(tss_imgs_us)-1):
        # visualize events

        mask = (evs[:, 0] >= tss_imgs_us[i]) & (evs[:, 0] < tss_imgs_us[i+1])
        if mask.sum() == 0:
            print(f"no events in chunk {tss_imgs_us[i]/1e6}")
            pbar.update(1)
            continue
        xs = evs[mask, 1]
        ys = evs[mask, 2]
        ps = evs[mask, 3]
        
        evimg = render(xs, ys, ps, H, W)
        cv2.imwrite(os.path.join(outvizfolder,  "%06d_evs" % i + ".png"), evimg)
        pbar.update(1)




if __name__ == "__main__":
    main()
