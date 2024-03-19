import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import multiprocessing

from utils.load_utils import compute_rmap_ecd, read_ecd_tss, compute_rmap_ecd
from utils.viz_utils import render


def process_dirs(indirs):
    for indir in indirs:
        print(f"\n\n ECD: Undistorting {indir} intrinsics, rgbs and evs")

        imgdir = os.path.join(indir, "images")
        imgdirout = os.path.join(indir, "images_undistorted")
        
        img_list = [os.path.join(indir, imgdir, im) for im in sorted(os.listdir(imgdir)) if im.endswith(".png")]
        H, W, _ = cv2.imread(img_list[0]).shape
        assert H == 180 and W == 240

        if not os.path.exists(imgdirout):
            os.makedirs(imgdirout)
        else:
            img_list_undist = [os.path.join(indir, imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".png")]
            if len(img_list) == len(img_list_undist):
                print(f"\n\nWARNING **** Images already undistorted. Skipping {indir} ***** \n\n")
                assert os.path.isfile(os.path.join(indir, "rectify_map.h5"))
                continue

        # saving tss
        tss_imgs_us = read_ecd_tss(os.path.join(indir, "images.txt"))
        f = open(os.path.join(indir, "tss_us.txt"), 'w')
        for t in tss_imgs_us:
            f.write(f"{t}\n")
        f.close()

        # creating rectify map
        intrinsics = np.loadtxt(os.path.join(indir, "calib.txt"))
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics
        Kdist =  np.zeros((3,3))        
        Kdist[0,0] = fx
        Kdist[0,2] = cx
        Kdist[1,1] = fy
        Kdist[1,2] = cy
        Kdist[2, 2] = 1
        dist_coeffs = np.asarray([k1, k2, p1, p2, k3])
        R = np.eye(3)

        rectify_map, Knew = compute_rmap_ecd(indir, H=H, W=W)
        f = open(os.path.join(indir, "calib_undist.txt"), 'w')
        f.write(f"{Knew[0,0]} {Knew[1,1]} {Knew[0,2]} {Knew[1,2]}")
        f.close()

        img_mapx, img_mapy = cv2.initUndistortRectifyMap(Kdist, dist_coeffs, R, Knew, (W, H), cv2.CV_32FC1)
        # undistorting images
        pbar = tqdm.tqdm(total=len(img_list)-1)
        for f in img_list:
                image = cv2.imread(f)
                img = cv2.remap(image, img_mapx, img_mapy, cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(imgdirout, os.path.split(f)[1]), img)
                pbar.update(1)


        # [DEBUG]
        evs_file = glob.glob(os.path.join(indir, "events.txt"))
        assert len(evs_file) == 1
        evs = np.asarray(np.loadtxt(evs_file[0], delimiter=" ")) # (N, 4) with [ts_sec, x, y, p]
        evs[:, 0] = evs[:, 0] * 1e6
        outvizfolder = os.path.join(indir, "evs_undist_viz") 
        os.makedirs(outvizfolder, exist_ok=True)
        DELTA_MS = 200
        
        pbar = tqdm.tqdm(total=len(img_list)-1)
        for (ts_idx, ts_us) in enumerate(tss_imgs_us):
            if ts_idx == len(tss_imgs_us) - 1:
                break
            
            if DELTA_MS is None:
                evs_idx = np.where((evs[:, 0] >= ts_us) & (evs[:, 0] < tss_imgs_us[ts_idx+1]))[0]
            else:
                evs_idx = np.where((evs[:, 0] >= ts_us) & (evs[:, 0] < ts_us + DELTA_MS*1e3))[0]
                
            if len(evs_idx) == 0:
                print(f"no events in range {ts_us*1e-3} - {tss_imgs_us[ts_idx+1]*1e-3} milisecs")
                continue
            evs_batch = np.array(evs[evs_idx, :]).copy()


            img = render(evs_batch[:, 1], evs_batch[:, 2], evs_batch[:, 3], H, W) 
            imgname = os.path.split(img_list[ts_idx])[1]
            cv2.imwrite(os.path.join(outvizfolder, imgname), img)

            rect = rectify_map[evs_batch[:, 2].astype(np.int32), evs_batch[:, 1].astype(np.int32)]
            img = render(rect[:, 0], rect[:, 1], evs_batch[:, 3], H, W)
            imgname = imgname.split(".")[0] + "_undist.png"
            cv2.imwrite(os.path.join(outvizfolder, imgname), img)

            pbar.update(1)

        print(f"Finshied processing {indir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP ECD data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for d in dirs:
            if "images" in d and "images.txt" in files:
                if root not in roots:
                    roots.append(root)

    
    cors = 16
    roots_split = np.array_split(roots, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_dirs, args=(roots_split[i].tolist(),))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
