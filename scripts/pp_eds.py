import numpy as np
import os
import argparse
import cv2
import tqdm
import multiprocessing
import h5py
import hdf5plugin
import json

from utils.viz_utils import render
from utils.event_utils import EventSlicer, compute_ms_to_idx


def process_seq_eds(indirs, calibstr="calib0"):
    for indir in indirs:
        assert calibstr == "calib0" or calibstr == "calib1"
        print(f"Processing {indir}")

        evinfile = os.path.join(indir, "events.h5")
        assert os.path.isfile(evinfile)
        
        imgdir = os.path.join(indir, "images")
        imgdirout = os.path.join(indir, f"images_undistorted_{calibstr}")
        os.makedirs(imgdirout, exist_ok=True)

        img_list = sorted(os.listdir(os.path.join(indir, imgdir)))
        img_list = [os.path.join(indir, imgdir, im) for im in img_list if im.endswith(".png")]
        H, W, _ = cv2.imread(img_list[0]).shape
        assert W == 640
        assert H == 480

        # 1) Getting offset which is substracted from evs, mocap and images.
        ef_in = h5py.File(os.path.join(indir, evinfile), "r+")
        tss_evs_us = ef_in["t"][:]
        gt_us = np.loadtxt(os.path.join(indir, "stamped_groundtruth.txt"))
        tss_gt_us = gt_us[:, 0] * 1e6
        tss_imgs_us = np.loadtxt(os.path.join(indir, "images_timestamps.txt"), skiprows=0)
        
        if not os.path.isfile(os.path.join(indir, "t_offset_us.txt")):
            offset_us = np.minimum(tss_evs_us.min(), np.minimum(tss_gt_us.min(), tss_imgs_us.min())).astype(np.int64)
            print(f"Minimum/offset_us is {offset_us}. tss_evs_us.min() = {tss_evs_us.min()-offset_us},  tss_gt_us.min() = {tss_gt_us.min()-offset_us}, tss_imgs_us.min() = {tss_imgs_us.min()-offset_us}")
            assert offset_us != 0 
            assert offset_us > 0

            tss_gt_us -= offset_us
            gt_us[:, 0] = tss_gt_us
            np.savetxt(os.path.join(indir, "stamped_groundtruth_us.txt"), gt_us, header="#timestamp[us] px py pz qx qy qz qw")

            tss_imgs_us -= offset_us
            np.savetxt(os.path.join(indir, "images_timestamps_us.txt"), tss_imgs_us, fmt="%d")

            ef_in["t"][:] -= offset_us
            tss_evs_us -= offset_us
            np.savetxt(os.path.join(indir, "t_offset_us.txt"), np.array([offset_us]))
        else:
            assert ef_in["t"][0] < 5000

        # calib data
        K_rgb = np.zeros((3,3))
        if calibstr == "calib1":
            K_rgb[0,0] = 758.1291471478728 ##### calib1 
            K_rgb[0,2] = 289.0985666049996
            K_rgb[1,1] = 759.5125594392973
            K_rgb[1,2] = 228.23374237672056
            K_rgb[2, 2] = 1
            dist_coeffs_rgb = np.asarray([-0.36599825863847607, 0.15566628749131536, 0.003684464282510181, 0.004564651739351755])
        elif calibstr == "calib0":
            K_rgb[0,0] = 766.536025127154 ##### calib0
            K_rgb[0,2] = 291.0503512057777
            K_rgb[1,1] = 767.5749459126396
            K_rgb[1,2] = 227.4060484950132
            K_rgb[2, 2] = 1
            dist_coeffs_rgb = np.asarray([-0.36965913545735024, 0.17414034009883844, 0.003915245015812422, 0.003666687416655559])
        
        K_new_rgb, roi = cv2.getOptimalNewCameraMatrix(K_rgb, dist_coeffs_rgb, (W, H), alpha=0, newImgSize=(W, H)) # alpha = 0 => all pixels in undistorted image are valid
        x,y,w,h = roi
        assert x == 0 and y == 0 and w+1 == W and h+1 == H
        intr_undist = []
        intr_undist.append({"fx": K_new_rgb[0,0], "fy": K_new_rgb[1,1], "cx": K_new_rgb[0,2], "cy": K_new_rgb[1,2]})

        K_evs = np.zeros((3,3))
        if calibstr == "calib1":
            K_evs[0,0] = 548.8989250692618 ##### calib1 
            K_evs[0,2] = 313.5293514832678
            K_evs[1,1] = 550.0282089284915
            K_evs[1,2] = 219.6325753720951
            K_evs[2, 2] = 1
            dist_coeffs_evs = np.asarray([-0.08095806072593555, 0.15743578875760092, -0.0035154416164982195, -0.003950567808338846])
        elif calibstr == "calib0":
            K_evs[0,0] = 560.8520948927032 ##### calib0 
            K_evs[0,2] = 313.00733235019237
            K_evs[1,1] = 560.6295819972383
            K_evs[1,2] = 217.32858679842997
            K_evs[2, 2] = 1
            dist_coeffs_evs = np.asarray([-0.09776467241921379, 0.2143738428636279, -0.004710710105172864, -0.004215916089401789])
        
        K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
        x,y,w,h = roi
        assert x == 0 and y == 0 and w+1 == W and h+1 == H
        intr_undist.append({"fx": K_new_evs[0,0], "fy": K_new_evs[1,1], "cx": K_new_evs[0,2], "cy": K_new_evs[1,2]})

        # 1) Saving undistorted intrinsics
        with open(os.path.join(indir, f"calib_undist_{calibstr}.json"), 'w') as f:
            calibdata = {}
            calibdata["intrinsics_undistorted"] = intr_undist
            json.dump(calibdata, f)

        # 2) undistorting images
        print("Undistorting images")
        pbar = tqdm.tqdm(total=len(img_list))
        for f in img_list:
            image = cv2.imread(f)
            img = cv2.undistort(image, K_rgb, dist_coeffs_rgb, newCameraMatrix=K_new_rgb)
            cv2.imwrite(os.path.join(imgdirout, os.path.split(f)[1]), img)
            pbar.update(1)
            # for debugging: 
            # cv2.imwrite(os.path.join(imgdirout,  os.path.split(f)[1][:-4] + "_undist.jpg"),  image)
        # shutil.copy(os.path.join(imgdir, "timestamps.txt"), os.path.join(imgdirout, "timestamps.txt"))
        # sys.exit()

        # 3) undistorting events => visualize
        coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float32")
        term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
        points = cv2.undistortPointsIter(coords, K_evs, dist_coeffs_evs, np.eye(3), K_new_evs, criteria=term_criteria)
        rectify_map = points.reshape((H, W, 2))

        # 4) Create rectify map for events
        h5outfile = os.path.join(indir, f"rectify_map_{calibstr}.h5")
        ef_out = h5py.File(h5outfile, 'w')
        ef_out.clear()
        ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
        ef_out["rectify_map"][:] = rectify_map
        ef_out.close()

        # 5) Computing ms_to_idx
        tss_evs_ns = tss_evs_us * 1000
        if "ms_to_idx" not in ef_in.keys():
            print(f"Start computing ms_to_idx, with {len(tss_evs_ns)} tss_evs_ns, {tss_evs_ns}, ms_start={ef_in['t'][0]}")
            ms_to_idx = compute_ms_to_idx(tss_evs_ns)
            print(f"Done computing ms_to_idx")
            ef_in.create_dataset('ms_to_idx', shape=len(ms_to_idx), dtype="<u8")
            ef_in["ms_to_idx"][:] = ms_to_idx

        # 6) Visualization from here on (undistorted evs)
        outvizfolder = os.path.join(indir, f"evs_rgb_joint_{calibstr}")
        os.makedirs(outvizfolder, exist_ok=True)
        event_slicer = EventSlicer(ef_in)

        if calibstr == "calib1":
            T_rgb_imu = np.asarray([ # cam0 calib1
                        [-0.9990674261177589, 0.003631371785536113, 0.04302430951674526, 0.029775744033325717], 
                        [-0.0028069257561474303, -0.9998115832434417, 0.019207268937644115, -3.945506719509616e-05],
                        [0.043085951750390296, 0.019068590697760665, 0.9988893780647411, -0.05058322878791149], 
                        [0.0, 0.0, 0.0, 1.0]])

            T_ev_imu = np.asarray([ # cam1 calib1
                        [-0.9995991738524179, 0.005137352230635228, 0.027840604261081696, 0.025511174632699758], 
                        [-0.005338788412348343, -0.9999600732256928, -0.007165842082780113, 0.00040093104150374986],
                        [0.027802679220750436, -0.007311604921325729, 0.9995866903183648, -0.06778644321079669], 
                        [0.0, 0.0, 0.0, 1.0]])
            T_ev_rgb = T_ev_imu @ np.linalg.inv(T_rgb_imu) # == T_cn_cnm1 == T_cam1_cam0  (calib1)
        elif calibstr == "calib0":
            T_ev_rgb = np.asarray([  # calib0
                [0.9998964430808897, -0.0020335804041023736, -0.014246672065022661, -0.00011238613157578769],
                [0.001703024953250547, 0.9997299470300024, -0.023176123864880376, -0.0005981481496958399],
                [0.014289955220253567, 0.02314946137886846, 0.9996298813149167, -0.004416681577516066],
                [0.0, 0.0, 0.0, 1.0]
            ])

        # Computing joint recitificaiton
        K_joint = K_evs # (K_evs + K_rgb) / 2.
        newR = T_ev_rgb[:3, :3]
        img_mapx, img_mapy = cv2.initUndistortRectifyMap(K_rgb, dist_coeffs_rgb, newR, K_joint, (W, H), cv2.CV_32FC1)
        ev_mapx, ev_mapy = cv2.initUndistortRectifyMap(K_evs, dist_coeffs_evs, np.eye(3), K_joint, (W, H), cv2.CV_32FC1)

        tss_img_us = np.loadtxt(os.path.join(indir, "images_timestamps_us.txt"))
        assert len(tss_imgs_us) == len(img_list)
        assert len(tss_img_us) == len(tss_imgs_us) # bug checker
        dT_ms_trigger_period = np.diff(tss_imgs_us).mean()/1e3
        print(f"Visualizing undistorted {len(tss_img_us)} event slices around images (with aligned optical axis)")
        pbar = tqdm.tqdm(total=len(tss_img_us)-1)
        for i in range(len(tss_img_us)-1):
            # undistort img
            # img = cv2.undistort(image, K_rgb, dist_coeffs_rgb, newCameraMatrix=K_joint)
            image =  cv2.imread(img_list[i])
            img = cv2.remap(image, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(outvizfolder, "%06d_a" % i + ".png"), img)

            start_time_us = tss_img_us[i] - 1e3 * dT_ms_trigger_period / 2.
            end_time_us = start_time_us + 1e3 * dT_ms_trigger_period / 2.
            ev_batch = event_slicer.get_events(start_time_us, end_time_us)
            if ev_batch is None:
                print(f"Got no events in {start_time_us/1e3} ms to {end_time_us/1e3} ms")
                continue
            p = ev_batch['p']
            x = ev_batch['x']
            y = ev_batch['y']
            # img = render_ev_accumulation(x, y, p, H, W)
            # cv2.imwrite(os.path.join(outvizfolder,  "%06d_b" % i + ".png"), img)
            x_rect = ev_mapx[y, x]
            y_rect = ev_mapy[y, x]
            img = render(x_rect, y_rect, p, H, W)
            cv2.imwrite(os.path.join(outvizfolder,  "%06d_undist" % i + ".png"), img)
            pbar.update(1)

        ef_in.close()

        print(f"Finshied processing {indir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP EDS data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for d in dirs:
            if "images" in d and "images_timestamps.txt" in files:
                if root not in roots:
                    roots.append(root)

    
    cors = 4
    roots_split = np.array_split(roots, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_seq_eds, args=(roots_split[i].tolist(),))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
