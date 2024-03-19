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
  
    H, W = 720, 1280
    with open(os.path.join(args.indir, "../calibration.json"), 'r') as f:
        calibdata = json.load(f)
        # loop all 4 cameras (0=left, 1=right, 2=left events, 3=right events)
        i = 2
        K_evs = np.zeros((3,3)) 
        K_evs[0,0] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["fx"]
        K_evs[0,2] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["cx"]
        K_evs[1,1] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["fy"]
        K_evs[1,2] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["cy"]
        K_evs[2, 2] = 1

        k1 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k1"]
        k2 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k2"]
        k3 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k3"]
        k4 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k4"]
        dist_coeffs_evs = np.asarray([k1, k2, k3, k4])

        W = calibdata["value0"]["resolution"][i][0]
        H = calibdata["value0"]["resolution"][i][1]
    K_new_evs = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_evs, dist_coeffs_evs, (W, H), np.eye(3), balance=0.0)

    with open(os.path.join(args.indir, "../calib_undist.json"), 'r') as f:
        calibdata = json.load(f)
        K_new_evs_load = np.array(list(calibdata["value0"]["intrinsics_undistorted"][i].values()))
        # assert np.allclose(np.array([K_new_evs[0,0], K_new_evs[1,1], K_new_evs[0,2], K_new_evs[1,2]]), K_new_evs_load)

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
