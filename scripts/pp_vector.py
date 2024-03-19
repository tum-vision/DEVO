import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import multiprocessing
import h5py
import yaml, shutil

from utils.viz_utils import render
from utils.pose_utils import quatList_to_poses_hom_and_tss, poses_hom_to_quatlist, check_rot

from utils.event_utils import EventSlicer, to_voxel_grid
from utils.load_utils import compute_rmap_vector

H, W = 480, 640
Hgray, Wgray = 1024, 1224


def process_seq_vector(indirs, side="left"):
    for indir in indirs:
        print(f"\n\n VECTOR: Computing rect-map for {indir}, and undistorting images.")
        assert side == "left" or side == "right"

        seq = indir.split("/")[-1]
        fnameh5 = os.path.join(indir, f"{seq}1.synced.{side}_event.hdf5")
        datain = h5py.File(fnameh5, 'r') # (events, ms_to_idx, t_offset)
        evs_slicer = EventSlicer(datain)

        imgdirname = os.path.join(indir, f"{seq}1.synced.{side}_camera")
        imgin_list = sorted(glob.glob(os.path.join(imgdirname, "*.png")))
        num_imgs = len(imgin_list)
        tss_imgs = np.loadtxt(os.path.join(imgdirname, "timestamp.txt"), skiprows=2)
        tss_imgs = 1e6*(tss_imgs[:, 0] + tss_imgs[:, 1])/2
        assert len(tss_imgs) == num_imgs
        assert abs(np.diff(tss_imgs).mean()/1e3 - 33.3) < 3

        # 1) load poses & extrinsics
        poses_in_secs = np.loadtxt(os.path.join(indir, f"{seq}1.synced.gt.txt"), skiprows=2)
        tss_poses_in, poses_hom_in = quatList_to_poses_hom_and_tss(poses_in_secs)
        fname_excalib = os.path.join(indir, f"../0_calib/camera_mocap_extrinsic_results1.yaml")
        with open(fname_excalib, 'r') as file:
            extrinsics = yaml.safe_load(file)

        T_camgrayLeft_body = np.array(extrinsics["cam0"]["T_cam_body"]).reshape(4,4)
        
        fname_jointexcalib = os.path.join(indir, f"../0_calib/small_scale_joint_camera_extrinsic_results.yaml")
        with open(fname_jointexcalib, 'r') as file:
            jointextrinsics = yaml.safe_load(file)
        
        # 1a) compute event poses
        camId = 2 if side == "left" else 3
        T_camgrayLeft_camEvs = np.array(jointextrinsics[f"cam{camId}"][f"T_cam0_cam{camId}"]).reshape(4,4)

        poses_out_hom = []
        tss_poses_out_us = []
        for i, pin_world_body in enumerate(poses_hom_in):
            pout_world_body = pin_world_body @ np.linalg.inv(T_camgrayLeft_body) @  T_camgrayLeft_camEvs
            check_rot(pout_world_body[:3,:3])
            poses_out_hom.append(pout_world_body)
            ts_us = tss_poses_in[i] * 1e6
            tss_poses_out_us.append(ts_us)
        quatlist_us = poses_hom_to_quatlist(np.array(poses_out_hom), tss_poses_out_us)

        # 1a) write event poses
        f = open(os.path.join(indir, f"poses_evs_{side}.txt"), 'w')
        for q in quatlist_us:
            f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} {q[4]} {q[5]} {q[6]} {q[7]}\n")
        f.close()

        # 1b) compute gray poses
        camId = 0 if side == "left" else 1
        T_camgrayLeft_camGray = np.eye(4) if camId == 0 else np.array(jointextrinsics[f"cam{camId}"][f"T_cam0_cam{camId}"])
        poses_out_hom = []
        tss_poses_out_us = []
        for i, pin_world_body in enumerate(poses_hom_in):
            pout_world_body = pin_world_body @ np.linalg.inv(T_camgrayLeft_body) @  T_camgrayLeft_camGray
            check_rot(pout_world_body[:3,:3])
            poses_out_hom.append(pout_world_body)
            ts_us = tss_poses_in[i] * 1e6
            tss_poses_out_us.append(ts_us)
        quatlist_us = poses_hom_to_quatlist(np.array(poses_out_hom), tss_poses_out_us)

        # 1b) write gray poses
        f = open(os.path.join(indir, f"poses_gray_{side}.txt"), 'w')
        for q in quatlist_us:
            f.write(f"{q[0]} {q[1]} {q[2]} {q[3]} {q[4]} {q[5]} {q[6]} {q[7]}\n")
        f.close()

        continue

        f = open(os.path.join(indir, f"tss_imgs_us_{side}.txt"), 'w')
        for tss in tss_imgs:
            f.write(f"{tss}\n")
        f.close()
        # continue

        imgdirout = os.path.join(indir, f"images_undistorted_{side}")
        if not os.path.exists(imgdirout):
            os.makedirs(imgdirout)
        else:
            img_list_undist = [os.path.join(imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".png")]
            if num_imgs == len(img_list_undist):
                print(f"\n\nWARNING **** Images already undistorted. Skipping {indir} ***** \n\n")
                # assert os.path.isfile(os.path.join(outdir, f"rectify_map_{side}.h5"))
                # continue

        # creating rectify map
        fname_evcalib = os.path.join(indir, f"../0_calib/{side}_event_camera_intrinsic_results.yaml")
        fname_graycalib = os.path.join(indir, f"../0_calib/{side}_regular_camera_intrinsic_results.yaml")
        with open(fname_evcalib, 'r') as file:
            intr_evs = yaml.safe_load(file)
        with open(fname_graycalib, 'r') as file:
            intr_gray = yaml.safe_load(file)

        # undist images
        Kgraydist = np.array(intr_gray["camera_matrix"]["data"]).reshape((3, 3))
        distcoeffs_gray = np.array(intr_gray["distortion_coefficients"]["data"]) # plumb_blob

        K_new, roi = cv2.getOptimalNewCameraMatrix(Kgraydist, distcoeffs_gray, (Wgray, Hgray), alpha=0, newImgSize=(Wgray, Hgray))
        f = open(os.path.join(indir, f"calib_undist_regular_{side}.txt"), 'w')
        f.write(f"{K_new[0,0]} {K_new[1,1]} {K_new[0,2]} {K_new[1,2]}")
        f.close()

        img_mapx, img_mapy = cv2.initUndistortRectifyMap(Kgraydist, distcoeffs_gray, np.eye(3), K_new, (Wgray, Hgray), cv2.CV_32FC1)  
        # undistorting images
        pbar = tqdm.tqdm(total=num_imgs-1)
        for i in range(num_imgs):
            image = cv2.imread(imgin_list[i])
            # DEBUG
            # cv2.imwrite(os.path.join(imgdirout, f"{i:06d}_dist.png"), image) 
            img = cv2.remap(image, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imgdirout, f"{i:06d}.png"), img)
            pbar.update(1)

        # undist events
        Kevsdist = np.array(intr_evs["camera_matrix"]["data"]).reshape((3, 3))
        distcoeffs_evs = np.array(intr_evs["distortion_coefficients"]["data"])

        rectify_map, Knewevs = compute_rmap_vector(Kevsdist, distcoeffs_evs, indir, side)
        f = open(os.path.join(indir, f"calib_undist_evs_{side}.txt"), 'w')
        f.write(f"{Knewevs[0,0]} {Knewevs[1,1]} {Knewevs[0,2]} {Knewevs[1,2]}")
        f.close()

        ###############################
        # [DEBUG]
        # outvizfolder = os.path.join(indir, f"evs_undist_viz_{side}") 
        # os.makedirs(outvizfolder, exist_ok=True)
        
        # pbar = tqdm.tqdm(total=num_imgs-1)
        # for img_i, ts_us in enumerate(tss_imgs):
        #     if img_i == num_imgs-1:
        #         continue

        #     evs_batch = evs_slicer.get_events(ts_us, tss_imgs[img_i+1])
        #     if evs_batch is None:
        #         print(f"WARNING: No events between {ts_us} and {tss_imgs[img_i+1]}")
        #         continue

        #     img = render(evs_batch["x"], evs_batch["y"], evs_batch["p"], H, W) 
        #     cv2.imwrite(os.path.join(outvizfolder, f"{img_i:06d}_dist.png"), img)

        #     rect = rectify_map[evs_batch["y"].astype(np.int32), evs_batch["x"].astype(np.int32)]
        #     img = render(rect[..., 0], rect[..., 1], evs_batch["p"], H, W)
        #     cv2.imwrite(os.path.join(outvizfolder, f"{img_i:06d}.png"), img)

        #     pbar.update(1)
        # end [DEBUG]
        ###############################

        datain.close()
        print(f"Finshied processing VECTOR {indir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP ECD data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    indirs = []
    for seq in os.listdir(args.indir):
        if "0_calib" not in seq:
            indirs.append(os.path.join(args.indir, seq))

    cors = 1
    indirs_split = np.array_split(indirs, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_seq_vector, args=(indirs_split[i].tolist(),))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
