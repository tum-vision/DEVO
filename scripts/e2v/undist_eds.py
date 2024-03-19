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

def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(        
        "--indir", help="Input image directory.", default=""
    )
    parser.add_argument(                               
        "--calibstr", help="Start idx", default="calib0", type=str
    )
    args = parser.parse_args()

    print(f"Undistorting {args.indir}")

    assert "e2vid" in args.indir or "e2calib" in args.indir
    calibstr = args.calibstr
    assert calibstr == "calib1" or calibstr == "calib0"

    K_evs = np.zeros((3,3))
    if calibstr == "calib1":
        K_evs[0,0] = 548.8989250692618
        K_evs[0,2] = 313.5293514832678
        K_evs[1,1] = 550.0282089284915
        K_evs[1,2] = 219.6325753720951
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray([-0.08095806072593555, 0.15743578875760092, -0.0035154416164982195, -0.003950567808338846])
    elif calibstr == "calib0":
        K_evs[0,0] = 560.8520948927032
        K_evs[0,2] = 313.00733235019237
        K_evs[1,1] = 560.6295819972383
        K_evs[1,2] = 217.32858679842997
        K_evs[2, 2] = 1
        dist_coeffs_evs = np.asarray([-0.09776467241921379, 0.2143738428636279, -0.004710710105172864, -0.004215916089401789])

    W, H = 640, 480
    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
    
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
        img_undist = cv2.undistort(image, K_evs, dist_coeffs_evs, newCameraMatrix=K_new_evs) # k1,k2,p1,p2
        # img_undist2 = cv2.remap(image, ev_mapx, ev_mapy, cv2.INTER_LINEAR)  # 
        #assert compute_psnr(img_undist2, img_undist) > 50
        cv2.imwrite(os.path.join(imgoutdir, f"{i:021d}.png"), img_undist)
        print(f"writing eds to {imgoutdir}")
        pbar.update(1)
        
    print(f"Done Undistorting {args.indir}")

if __name__ == "__main__":
    main()
