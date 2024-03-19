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
  
    H, W = 480, 640
    fname_evcalib = os.path.join(args.indir, f"../../0_calib/left_event_camera_intrinsic_results.yaml")
    with open(fname_evcalib, 'r') as file:
        intr_evs = yaml.safe_load(file)
    K_evs = np.array(intr_evs["camera_matrix"]["data"]).reshape((3, 3))
    dist_coeffs_evs = np.array(intr_evs["distortion_coefficients"]["data"])
    K_new_evs, _ = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))

    K_new_evs_load = np.loadtxt(os.path.join(args.indir, "../calib_undist_evs_left.txt"))
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
        img_undist = cv2.undistort(image, K_evs, dist_coeffs_evs, newCameraMatrix=K_new_evs) 
        cv2.imwrite(os.path.join(imgoutdir, f"{i:021d}.png"), img_undist)
        pbar.update(1)
    print(f"Done Undistorting {args.indir}")

if __name__ == "__main__":
    main()
