import numpy as np
import os
import argparse
import cv2
import tqdm
import multiprocessing
import h5py
import hdf5plugin
import json
import glob
from scipy.spatial.transform import Rotation as R

from utils.viz_utils import render
from utils.event_utils import EventSlicer, compute_ms_to_idx
from utils.load_utils import read_ecd_tss, get_calib_fpv
from utils.pose_utils import check_rot


def write_poses(indir, T_cam_imu):
    fposesin = os.path.join(indir, "stamped_groundtruth_us.txt")
    
    poses_in = np.loadtxt(fposesin, skiprows=1)
    tss_us, poses_in = poses_in[:, 0], poses_in[:, 1:]

    T_body_cam = np.linalg.inv(T_cam_imu)

    poses_out = []
    for i, quat_in in enumerate(poses_in):
        T_world_body = np.eye(4)
        T_world_body[:3, 3] = quat_in[:3]
        T_world_body[:3, :3] = R.from_quat(quat_in[3:]).as_matrix()
        check_rot(T_world_body[:3, :3])
        T_world_cam = T_world_body @ T_body_cam

        quat_out = R.from_matrix(T_world_cam[:3, :3]).as_quat()
        pos_out = T_world_cam[:3, 3]
        poses_out.append(np.array([tss_us[i], pos_out[0], pos_out[1], pos_out[2], quat_out[0], quat_out[1], quat_out[2], quat_out[3]]))

    poses_out = np.asarray(poses_out)
    fposesout = os.path.join(indir, "stamped_groundtruth_us_cam.txt")
    np.savetxt(fposesout, poses_out, fmt="%.6f")
      
    
def process_seq_fpv(indirs):
    for indir in indirs:
        print(f"Processing {indir}")

        has_gt = "_with_gt" in indir
        
        evs_file = glob.glob(os.path.join(indir, "events.txt"))
        assert len(evs_file) == 1
        evs = np.asarray(np.loadtxt(evs_file[0], delimiter=" ")) # (N, 4) with [ts_sec, x, y, p]
        evs[:, 0] = evs[:, 0] * 1e6 

        imgdir = os.path.join(indir, "img")
        imgdirout = os.path.join(indir, f"images_undistorted")
        os.makedirs(imgdirout, exist_ok=True)

        img_list = sorted(os.listdir(os.path.join(indir, imgdir)))
        img_list = [os.path.join(indir, imgdir, im) for im in img_list if im.endswith(".png")]
        H, W, _ = cv2.imread(img_list[0]).shape
        assert W == 346
        assert H == 260

        # 1) Getting offset which is substracted from evs, mocap and images.
        tss_evs_us = evs[:, 0].copy()
        tss_imgs_us = read_ecd_tss(os.path.join(indir, "images.txt"), idx=1)
        if has_gt:
            gt_us = np.loadtxt(os.path.join(indir, "groundtruth.txt"), skiprows=1) 
            tss_gt_us = gt_us[:, 0] * 1e6
        else:
            tss_gt_us = tss_imgs_us.copy()
        
        if not os.path.isfile(os.path.join(indir, "t_offset_us.txt")):
            offset_us = np.minimum(tss_evs_us.min(), np.minimum(tss_gt_us.min(), tss_imgs_us.min())).astype(np.int64)
            print(f"Minimum/offset_us is {offset_us}. tss_evs_us.min() = {tss_evs_us.min()-offset_us},  tss_gt_us.min() = {tss_gt_us.min()-offset_us}, tss_imgs_us.min() = {tss_imgs_us.min()-offset_us}")
            assert offset_us != 0 
            assert offset_us > 0

            if has_gt:
                tss_gt_us -= offset_us
                gt_us[:, 0] = tss_gt_us
                np.savetxt(os.path.join(indir, "stamped_groundtruth_us.txt"), gt_us, header="#timestamp[us] px py pz qx qy qz qw")

            tss_imgs_us -= offset_us
            np.savetxt(os.path.join(indir, "images_timestamps_us.txt"), tss_imgs_us, fmt="%d")

            evs[:, 0] -= offset_us
            tss_evs_us -= offset_us
            np.savetxt(os.path.join(indir, "t_offset_us.txt"), np.array([offset_us]))
        else:
            offset_us = np.loadtxt(os.path.join(indir, "t_offset_us.txt")).astype(np.int64)
            print(f"Using offset_us = {offset_us}")
            evs[:, 0] -= offset_us
            assert evs[0, 0] < 1e6

        # calib data
        Kdist, dist_coeffs, T_cam_imu = get_calib_fpv(indir)

        if has_gt:
            write_poses(indir, T_cam_imu)
        
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(Kdist, dist_coeffs, (W, H), np.eye(3), balance=0)
        img_mapx, img_mapy = cv2.fisheye.initUndistortRectifyMap(Kdist, dist_coeffs, np.eye(3), Knew, (W, H), cv2.CV_32FC1)

        f = open(os.path.join(indir, "calib_undist.txt"), 'w')
        f.write(f"{Knew[0,0]} {Knew[1,1]} {Knew[0,2]} {Knew[1,2]}")
        f.close()        

        # 1) undistorting images
        img_list_undist = sorted(os.listdir(os.path.join(indir, imgdirout)))
        if len(img_list_undist) == len(img_list):
            print("Images already undistorted. Skipping")
        else:
            print("Undistorting images")
            pbar = tqdm.tqdm(total=len(img_list))
            for f in img_list:
                image = cv2.imread(f)
                img = cv2.remap(image, img_mapx, img_mapy, cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(imgdirout, os.path.split(f)[1]), img)
                pbar.update(1)
                # for debugging: 
                # cv2.imwrite(os.path.join(imgdirout,  os.path.split(f)[1][:-4] + "dist.jpg"),  image)

        # 2) undistorting events => visualize
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        xys = np.stack((xs, ys), axis=-1) # (H, W, 2)
        xys_remap = cv2.fisheye.undistortPoints(xys.astype(np.float32), Kdist, dist_coeffs, R=np.eye(3), P=Knew)

        # 4) Create rectify map for events
        h5outfile = os.path.join(indir, f"rectify_map.h5")
        ef_out = h5py.File(h5outfile, 'w')
        ef_out.clear()
        ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
        ef_out["rectify_map"][:] = xys_remap
        ef_out.close()

        tss_imgs_us = np.loadtxt(os.path.join(indir, "images_timestamps_us.txt"))
        assert len(tss_imgs_us) == len(img_list)

        #########
        # [DEBUG]
        # outvizfolder = os.path.join(indir, "evs_undist_viz")
        # os.makedirs(outvizfolder, exist_ok=True)
        # pbar = tqdm.tqdm(total=len(tss_imgs_us)-1)
        # for i in range(len(tss_imgs_us)-1):
        #     if i < 200:
        #         continue
        #     mask = (evs[:, 0] >= tss_imgs_us[i]) & (evs[:, 0] < tss_imgs_us[i+1])
        #     if mask.sum() == 0:
        #         print(f"no events in {i}th chunk {tss_imgs_us[i]/1e6}")
        #         pbar.update(1)
        #         continue

        #     xs = evs[mask, 1].astype(np.int32)
        #     ys = evs[mask, 2].astype(np.int32)
        #     ps = evs[mask, 3].astype(np.int32)
        #     img = render(xs, ys, ps, H, W)
        #     cv2.imwrite(os.path.join(outvizfolder,  "%06d" % i + "_dist.png"), img)

        #     rect = xys_remap[ys, xs]
        #     img = render(rect[..., 0], rect[...,1], ps, H, W)
        #     cv2.imwrite(os.path.join(outvizfolder,  "%06d" % i + ".png"), img)         
        #     pbar.update(1)
        # end [DEBUG]
        #########

        print(f"Finshied processing FPV {indir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP FPV data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for d in dirs:
            if "img" in d and "images.txt" in files and "events.txt" in files:
                if root not in roots :
                    roots.append(root)
    
    cors = 4
    roots_split = np.array_split(roots, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_seq_fpv, args=(roots_split[i].tolist(),))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
