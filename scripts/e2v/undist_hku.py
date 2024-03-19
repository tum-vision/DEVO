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
    args = parser.parse_args()

    print(f"Undistorting {args.indir}")

    assert "e2vid" in args.indir or "e2calib" in args.indir
  
    H, W = 260, 346
    intr = [249.69341447817564, 248.41625664694038, 176.74240257052816, 129.47631010746218]
    dist_coeffs_evs = np.array([-0.3794794654640921, 0.15393049046270296, 0.0011400586965363895, -0.0019042695753031854])
    K_evs = np.eye(3)
    K_evs[0,0] = intr[0]
    K_evs[1,1] = intr[1]
    K_evs[0,2] = intr[2]
    K_evs[1,2] = intr[3]
    
    K_new_evs, _ = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))

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
