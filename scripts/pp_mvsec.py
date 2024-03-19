import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import multiprocessing
import h5py
import yaml, shutil

from utils.load_utils import compute_rmap_ecd, read_ecd_tss, compute_rmap_ecd
from utils.viz_utils import render


def compute_rmap_mvsec(scenedir, fx, fy, cx, cy, k1, k2, k3, k4, side, H=260, W=346):

    K_evs =  np.zeros((3,3))        
    K_evs[0,0] = fx
    K_evs[0,2] = cx 
    K_evs[1,1] = fy
    K_evs[1,2] = cy
    K_evs[2, 2] = 1
    dist_coeffs_evs = np.asarray([k1, k2, k3, k4])

    # K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
    K_new_evs = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_evs, dist_coeffs_evs, (W, H), np.eye(3), balance=0)
    
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float64")
    term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
    points = cv2.fisheye.undistortPoints(coords.T[:, None, :], K_evs, dist_coeffs_evs, P=K_new_evs)
    # cv2.undistortPointsIter(coords, K_evs, dist_coeffs_evs, np.eye(3), K_new_evs, criteria=term_criteria)
    rectify_map = points.reshape((H, W, 2))

    # 4) Create rectify map for events
    h5outfile = os.path.join(scenedir, f"rectify_map_{side}.h5")
    ef_out = h5py.File(h5outfile, 'w')
    ef_out.clear()
    ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
    ef_out["rectify_map"][:] = rectify_map
    ef_out.close()

    return rectify_map, K_new_evs

def process_seq_mvsec(infilesh5, side="left"):
    for fnameh5 in infilesh5:
        print(f"\n\n MVSEC: Computing rect-map for {fnameh5}, and undistorting images.")

        outdir = fnameh5.split(".")[0]
        fnameh5_gt = fnameh5.split(".")[0][:-5] + "_gt.hdf5"
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            
            shutil.copy(fnameh5, outdir)
            shutil.copy(fnameh5_gt, outdir)

            fnameh5 = os.path.join(outdir, os.path.basename(fnameh5))
            fnameh5_gt = fnameh5.split(".")[0][:-5] + "_gt.hdf5"
        
        datain = h5py.File(fnameh5, 'r')
        H, W = datain["davis"][side]["image_raw"].shape[1:]
        num_imgs = datain["davis"][side]["image_raw"].shape[0]

        tss_imgs = datain["davis"][side]["image_raw_ts"][:] * 1e6
        f = open(os.path.join(outdir, f"tss_imgs_us_{side}.txt"), 'w')
        for tss in tss_imgs:
            f.write(f"{tss}\n")
        f.close()
        # continue

        imgdirout = os.path.join(outdir, f"images_undistorted_{side}")
        if not os.path.exists(imgdirout):
            os.makedirs(imgdirout)
        else:
            img_list_undist = [os.path.join(imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".png")]
            if num_imgs == len(img_list_undist):
                print(f"\n\nWARNING **** Images already undistorted. Skipping {outdir} ***** \n\n")
                # assert os.path.isfile(os.path.join(outdir, f"rectify_map_{side}.h5"))
                # continue

        # creating rectify map
        with open(os.path.join(outdir, "../indoor_flying_calib/camchain-imucam-indoor_flying.yaml"), 'r') as file:
            all_intr = yaml.safe_load(file)

        camID = "cam0" if side == "left" else "cam1"
        fx, fy, cx, cy = all_intr[camID]["intrinsics"]
        k1, k2, k3, k4 = all_intr[camID]["distortion_coeffs"]
        dist_coeffs = np.asarray([k1, k2, k3, k4])

        rectify_map, Knew = compute_rmap_mvsec(outdir, fx, fy, cx, cy, k1, k2, k3, k4, side, H=H, W=W)
        f = open(os.path.join(outdir, f"calib_undist_{side}.txt"), 'w')
        f.write(f"{Knew[0,0]} {Knew[1,1]} {Knew[0,2]} {Knew[1,2]}")
        f.close()

        Kdist =  np.zeros((3,3))        
        Kdist[0,0] = fx
        Kdist[0,2] = cx
        Kdist[1,1] = fy
        Kdist[1,2] = cy
        Kdist[2, 2] = 1
        R = np.eye(3)

        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(Kdist, dist_coeffs, (W, H), np.eye(3), balance=0)
        img_mapx, img_mapy = cv2.fisheye.initUndistortRectifyMap(Kdist, dist_coeffs, R, Knew, (W, H), cv2.CV_32FC1)
        # undistorting images
        pbar = tqdm.tqdm(total=num_imgs-1)
        for i in range(num_imgs):
            image = datain["davis"][side]["image_raw"][i]
            # DEBUG
            # img = cv2.fisheye.undistortImage(image, Kdist, dist_coeffs, Knew=K_new, new_size=(W, H)) # seems more blurry than remap
            # cv2.imwrite(os.path.join(imgdirout, f"{i:06d}_undistFisheye.png"), img)
            # cv2.imwrite(os.path.join(imgdirout, f"{i:06d}_dist.png"), image) # DEBUG
            img = cv2.remap(image, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imgdirout, f"{i:06d}.png"), img)
            pbar.update(1)



        ###############################
        # [DEBUG]
        # outvizfolder = os.path.join(outdir, f"evs_undist_viz_{side}") 
        # os.makedirs(outvizfolder, exist_ok=True)
        
        # pbar = tqdm.tqdm(total=num_imgs-1)
        # event_idxs = datain["davis"][side]["image_raw_event_inds"]
        # all_evs = datain["davis"][side]["events"][:]
        # evidx_left = 0
        # for img_i in range(num_imgs):             
        #     evid_nextimg = event_idxs[img_i]
        #     evs_batch = all_evs[evidx_left:evid_nextimg][:]
        #     evidx_left = evid_nextimg

        #     img = render(evs_batch[:, 0], evs_batch[:, 1], evs_batch[:, 3], H, W) 
        #     cv2.imwrite(os.path.join(outvizfolder, f"{img_i:06d}_dist.png"), img)

        #     rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]
        #     img = render(rect[:, 0], rect[:, 1], evs_batch[:, 3], H, W)
        #     cv2.imwrite(os.path.join(outvizfolder, f"{img_i:06d}.png"), img)

        #     pbar.update(1)
        # end [DEBUG]
        ###############################

        datain.close()
        print(f"Finshied processing MVSEC {outdir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP MVSEC data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    infilesh5 = []
    for file in os.listdir(args.indir):
        if "_data.hdf5" in file:
            infilesh5.append(os.path.join(args.indir, file))

    cors = 2
    infilesh5_split = np.array_split(infilesh5, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_seq_mvsec, args=(infilesh5_split[i].tolist(),))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
