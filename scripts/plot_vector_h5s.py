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
from utils.event_utils import EventSlicer
from utils.viz_utils import render


def plot_seq(infile, rmap_fname, H=480, W=640):
    outvizfolder = os.path.join(os.path.dirname(infile), "viz_h5")
    os.makedirs(outvizfolder, exist_ok=True)

    ef_in = h5py.File(infile, "r")
    rmap = h5py.File(rmap_fname, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    ev_mapx, ev_mapy = rectify_map[...,0], rectify_map[...,1]
    rmap.close()

    event_slicer = EventSlicer(ef_in)
    print(f"Visualizing undistorted events in {event_slicer.get_start_time_us()/1e6} to {event_slicer.get_final_time_us()/1e6} secs around images (with aligned optical axis)")
    
    tssfn = glob.glob(os.path.join(os.path.dirname(infile), "tss_imgs_us_left.txt"))[0]
    FREQ = 1e6/np.diff(np.loadtxt(tssfn)).mean()
    
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


def main():
    parser = argparse.ArgumentParser(description="Plotting event chunks from h5 file")
    parser.add_argument(         
        "--indir", help="Infile", default=""
    )
    parser.add_argument("--H",  default=480, type=int)
    parser.add_argument("--W", default=640, type=int)
    args = parser.parse_args()

    assert "/VECTOR/" in args.indir

    for seq in os.listdir(args.indir):
        if "0_calib" in seq:
            continue

        evsfname = sorted(glob.glob(os.path.join(args.indir, seq, "*left_event.hdf5")))[0]
        rmap_fname = sorted(glob.glob(os.path.join(args.indir, seq, "*rectify_map_left.h5")))[0]
        
        plot_seq(evsfname, rmap_fname, H=args.H, W=args.W)
        #rmap_fname = os.path.join(args.indir, seq, "rectify_map.hdf5")
        #

if __name__ == "__main__":
    main()
