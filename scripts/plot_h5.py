import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import hdf5plugin
import glob
import math
from utils.event_utils import EventSlicer, compute_ms_to_idx
from utils.viz_utils import render


def main():
    parser = argparse.ArgumentParser(description="Plotting event chunks from h5 file")
    parser.add_argument(         
        "--infile", help="Infile", default=""
    )
    parser.add_argument(         
        "--rmap_fname", help="Infile", default=None,
    )
    parser.add_argument(         
        "--tss_fname", help="Infile", default=None,  
    )
    parser.add_argument("--H",  default=480, type=int)
    parser.add_argument("--W", default=640, type=int)
    args = parser.parse_args()
    assert ".h5" in args.infile or ".hdf5" in args.infile, "infile must be a h5 file"
    outvizfolder = os.path.join(os.path.dirname(args.infile), "viz_h5")
    os.makedirs(outvizfolder, exist_ok=True)
    
    W = args.W
    H = args.H

    if args.rmap_fname is None:
        args.rmap_fname = sorted(glob.glob(os.path.join(os.path.dirname(args.infile), "rectify_map*.h5")))[0]

    ef_in = h5py.File(args.infile, "r")
    rmap = h5py.File(args.rmap_fname, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    ev_mapx, ev_mapy = rectify_map[...,0], rectify_map[...,1]
    rmap.close()

    event_slicer = EventSlicer(ef_in)
    print(f"Visualizing undistorted events in {event_slicer.get_start_time_us()/1e6} to {event_slicer.get_final_time_us()/1e6} secs around images (with aligned optical axis)")
    
    if args.tss_fname is None:
        fname = glob.glob(os.path.join(os.path.dirname(args.infile), "images_timestamps_us.txt"))
        if len(fname) != 1:
            fname = glob.glob(os.path.join(os.path.dirname(args.infile), "tss_img*.txt"))
        args.tss_fname = fname[0]
    FREQ = 1e6/np.diff(np.loadtxt(args.tss_fname)).mean()
    
    Nbatches = math.ceil((event_slicer.get_final_time_us()-event_slicer.get_start_time_us())/1e6 * FREQ)
    print(f"Nb = {Nbatches}")
    tss_imgs_us = np.linspace(event_slicer.get_start_time_us(), event_slicer.get_final_time_us(), Nbatches+2)
    pbar = tqdm.tqdm(total=len(tss_imgs_us)-1)
    for i in range(len(tss_imgs_us)-2):
        # visualize events
        start_time_us = tss_imgs_us[i]
        end_time_us = tss_imgs_us[i+1]
        ev_batch = event_slicer.get_events(start_time_us, end_time_us)
        if ev_batch is None:
            print(f"Got no events in {start_time_us/1e3} ms to {end_time_us/1e3} ms")
            continue
        p = ev_batch['p']
        x = ev_batch['x']
        y = ev_batch['y']
        x_rect = ev_mapx[y, x]
        y_rect = ev_mapy[y, x]
        evimg = render(x_rect, y_rect, p, H, W)
        cv2.imwrite(os.path.join(outvizfolder,  "%06d_undist" % i + ".png"), evimg)
        pbar.update(1)

    ef_in.close()



if __name__ == "__main__":
    main()
