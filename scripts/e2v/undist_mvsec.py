import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import glob
import yaml

def main():
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(        
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    print(f"Undistorting {args.indir}")

    assert "e2vid" in args.indir or "e2calib" in args.indir
  
    H, W = 260, 346
    with open(os.path.join(args.indir, "../../indoor_flying_calib/camchain-imucam-indoor_flying.yaml"), 'r') as file:
        all_intr = yaml.safe_load(file)

    camID = "cam0"
    fx, fy, cx, cy = all_intr[camID]["intrinsics"]
    k1, k2, k3, k4 = all_intr[camID]["distortion_coeffs"]

    K_evs =  np.zeros((3,3))        
    K_evs[0,0] = fx
    K_evs[0,2] = cx 
    K_evs[1,1] = fy
    K_evs[1,2] = cy
    K_evs[2, 2] = 1
    dist_coeffs_evs = np.asarray([k1, k2, k3, k4])
    K_new_evs = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_evs, dist_coeffs_evs, (W, H), np.eye(3), balance=0)
    K_new_evs_load = np.loadtxt(os.path.join(args.indir, "../calib_undist_left.txt"))
    assert np.allclose(np.array([K_new_evs[0,0], K_new_evs[1,1], K_new_evs[0,2], K_new_evs[1,2]]), K_new_evs_load)

    img_list = sorted(glob.glob(args.indir + "/*.png"))
    hin, win, _ = cv2.imread(img_list[0]).shape
    assert H == hin and W == win

    imgoutdir = os.path.join(os.path.dirname(args.indir), "e2calib_undistorted")
    os.makedirs(imgoutdir, exist_ok=True)
    assert imgoutdir != args.indir
    
    pbar = tqdm.tqdm(total=len(img_list))
    for i in range(len(img_list)):
        # undistort img
        image = cv2.imread(img_list[i])
        img_undist = cv2.fisheye.undistortImage(image, K_evs, dist_coeffs_evs, Knew=K_new_evs)
        cv2.imwrite(os.path.join(imgoutdir, f"{i:021d}.png"), img_undist)
        pbar.update(1)
    print(f"Done Undistorting {args.indir}")

if __name__ == "__main__":
    main()
