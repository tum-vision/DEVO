from genericpath import exists
from math import dist
import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import glob

from utils.load_utils import load_intrinsics_rpg


def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(        
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    print(f"Undistorting {args.indir}")

    assert "e2vid" in args.indir or "e2calib" in args.indir

    
    intrinsics = [196.63936292910697, 196.7329768429481, 105.06412666477927, 72.47170071387173, 
                    -0.3367326394292646, 0.11178850939644308, -0.0014005281258491276, -0.00045959441440687044]

    fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
    K_evs = np.zeros((3,3))  
    K_evs[0,0] = fx
    K_evs[0,2] = cx
    K_evs[1,1] = fy
    K_evs[1,2] = cy
    K_evs[2, 2] = 1
    dist_coeffs_evs = np.asarray([k1, k2, p1, p2])

    H, W = 180, 240
    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))

    intrinsics = load_intrinsics_rpg(os.path.join(args.indir, f"../calib_undist_left.txt"))
    intrinsics = np.array(intrinsics)
    assert np.allclose(np.array([K_new_evs[0,0], K_new_evs[1,1], K_new_evs[0,2], K_new_evs[1,2]]), intrinsics)
    
    img_list = sorted(glob.glob(args.indir + "/*.png"))
    hin, win, _ = cv2.imread(img_list[0]).shape
    assert H == hin and W == win

    imgoutdir = os.path.join(os.path.dirname(args.indir), "e2calib_undistorted")
    os.makedirs(imgoutdir, exist_ok=True)
    assert imgoutdir != args.indir
    
    pbar = tqdm.tqdm(total=len(img_list))
    for i in range(len(img_list)):
        # undistort img
        image =  cv2.imread(img_list[i])
        img_undist = cv2.undistort(image, K_evs, dist_coeffs_evs, newCameraMatrix=K_new_evs) 
        cv2.imwrite(os.path.join(imgoutdir, f"{i:021d}.png"), img_undist)
        print(f"writing eds to {imgoutdir}")
        pbar.update(1)
        
    print(f"Done Undistorting {args.indir}")

if __name__ == "__main__":
    main()
