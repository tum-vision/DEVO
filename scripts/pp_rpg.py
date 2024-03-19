import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import multiprocessing

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tqdm as tqdm
import h5py
from utils.bag_utils import read_H_W_from_bag, read_tss_us_from_rosbag, read_images_from_rosbag, read_evs_from_rosbag, read_calib_from_bag, read_t0us_evs_from_rosbag, read_poses_from_rosbag

def write_gt_stamped(poses, tss_us_gt, outfile):
    with open(outfile, 'w') as f:
        for pose, ts in zip(poses, tss_us_gt):
            f.write(f"{ts} ")
            for i, p in enumerate(pose):
                if i < len(pose) - 1:
                    f.write(f"{p} ")
                else:
                    f.write(f"{p}")
            f.write("\n")


def get_calib_rpg(H, W, side, bag, imtopic):
    if H == 180 and W == 240:
        if side == "left":
            intrinsics = [196.63936292910697, 196.7329768429481, 105.06412666477927, 72.47170071387173, 
                          -0.3367326394292646, 0.11178850939644308, -0.0014005281258491276, -0.00045959441440687044]
        else:
            intrinsics = [196.42564072599785, 196.56440793223533, 110.74517642512458, 88.11310058123058,
                          -0.3462937629552321, 0.12772002965572962, -0.00027205054024332645, -0.00019580078540073353]
        fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
        Kdist =  np.zeros((3,3))   
        Kdist[0,0] = fx
        Kdist[0,2] = cx
        Kdist[1,1] = fy
        Kdist[1,2] = cy
        Kdist[2, 2] = 1
        dist_coeffs = np.asarray([k1, k2, p1, p2])

        return Kdist, dist_coeffs
    elif H == 260 and W == 346:
        K = read_calib_from_bag(bag, imtopic.replace("image_raw", "camera_info"))
        Kdist =  np.zeros((3,3))   
        Kdist[0,0] = K[0]
        Kdist[0,2] = K[2]
        Kdist[1,1] = K[4]
        Kdist[1,2] = K[5]
        Kdist[2, 2] = 1
        return Kdist, np.array([0., 0., 0., 0.])
    else:
        raise NotImplementedError


def process_dirs(indirs, side="left", DELTA_MS=None):
    for indir in indirs: 
        seq = indir.split("/")[-1]
        print(f"\n\n RPG: Undistorting {seq} evs & rgb")

        inbag = os.path.join(indir, f"../{seq}.bag")
        bag = rosbag.Bag(inbag, "r")
        topics = list(bag.get_type_and_topic_info()[1].keys())
        topics = sorted([t for t in topics if "events" in t or "image" in t])
        assert topics == sorted(['/davis_left/events', '/davis_left/image_raw', '/davis_right/events', '/davis_right/image_raw']) or topics == sorted(['/davis/left/events', '/davis/left/image_raw', '/davis/right/events', '/davis/right/image_raw'])
        if side == "left":
            imgtopic_idx = 1
            evtopic_idx = 0
        elif side == "right":
            imgtopic_idx = 3
            evtopic_idx = 2
        else:
            raise NotImplementedError

        imgdirout = os.path.join(indir, f"images_undistorted_{side}")
        H, W = read_H_W_from_bag(bag, topics[imgtopic_idx])
        assert (H == 180 and W == 240) or (H == 260 and W == 346)

        if not os.path.exists(imgdirout):
            os.makedirs(imgdirout)
        else:
            img_list_undist = [os.path.join(indir, imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".png")]
            if bag.get_message_count(topics[1]) == len(img_list_undist):
                print(f"\n\nWARNING **** Images already undistorted. Skipping {indir} ***** \n\n")
                assert os.path.isfile(os.path.join(indir, f"rectify_map_{side}.h5")) or seq == "simulation_3planes"
                # continue

        imgs = read_images_from_rosbag(bag, topics[imgtopic_idx], H=H, W=W)
    
        # creating rectify map
        if seq != "simulation_3planes":
            if side == "left":
                intrinsics = [196.63936292910697, 196.7329768429481, 105.06412666477927, 72.47170071387173, 
                            -0.3367326394292646, 0.11178850939644308, -0.0014005281258491276, -0.00045959441440687044]
            else:
                intrinsics = [196.42564072599785, 196.56440793223533, 110.74517642512458, 88.11310058123058,
                            -0.3462937629552321, 0.12772002965572962, -0.00027205054024332645, -0.00019580078540073353]
            fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics
            Kdist =  np.zeros((3,3))   
            Kdist[0,0] = fx
            Kdist[0,2] = cx
            Kdist[1,1] = fy
            Kdist[1,2] = cy
            Kdist[2, 2] = 1
            dist_coeffs = np.asarray([k1, k2, p1, p2])

            K_new, roi = cv2.getOptimalNewCameraMatrix(Kdist, dist_coeffs, (W, H), alpha=0, newImgSize=(W, H))
            
            coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float32") # TODO: +-1 missing??
            term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
            points = cv2.undistortPointsIter(coords, Kdist, dist_coeffs, np.eye(3), K_new, criteria=term_criteria)
            rectify_map = points.reshape((H, W, 2))       

            h5outfile = os.path.join(indir, f"rectify_map_{side}.h5")
            ef_out = h5py.File(h5outfile, 'w')
            ef_out.clear()
            ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
            ef_out["rectify_map"][:] = rectify_map
            ef_out.close() 

            img_mapx, img_mapy = cv2.initUndistortRectifyMap(Kdist, dist_coeffs, np.eye(3), K_new, (W, H), cv2.CV_32FC1)  
        else:
            for topic, msg, t in bag.read_messages(topics[imgtopic_idx].replace("image_raw", "camera_info")):
                K = msg.K
                break
            Kdist =  np.zeros((3,3))   
            Kdist[0,0] = K[0]
            Kdist[0,2] = K[2]
            Kdist[1,1] = K[4]
            Kdist[1,2] = K[5]
            Kdist[2, 2] = 1
            
            K_new = Kdist.copy()

        f = open(os.path.join(indir, f"calib_undist_{side}.txt"), 'w')
        f.write(f"{K_new[0,0]} {K_new[1,1]} {K_new[0,2]} {K_new[1,2]}")
        f.close()

        # undistorting images
        pbar = tqdm.tqdm(total=len(imgs)-1)
        for i, img in enumerate(imgs):
            # cv2.imwrite(os.path.join(imgdirout, f"{i:012d}_DIST.png"), img)
            if seq != "simulation_3planes":
                img = cv2.remap(img, img_mapx, img_mapy, cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(imgdirout, f"{i:012d}.png"), img)
            pbar.update(1)

        # writing pose to file
        if not seq == "simulation_3planes":
            posetopic = "/optitrack/davis_stereo"
            T_marker_cam0 = np.array([[5.36262328777285e-01, -1.748374625145743e-02, -8.438296573030597e-01, -7.009849865398374e-02],
                                      [8.433577587813513e-01, -2.821937531845164e-02, 5.366109927684415e-01, 1.881333563905305e-02],
                                      [-3.31943162375816e-02, -9.994488408486204e-01, -3.897382049768972e-04, -6.966829200678797e-02],
                                      [0.0, 0.0, 0.0, 1.0]])
            if side == "left":
                T_cam0_cam1 = np.eye(4)
            else:
                T_cam0_cam1 =  np.array([[0.9991089760393723, -0.04098010198963204, 0.010093821797214667, -0.1479883582369969],
                                        [0.04098846609277917, 0.9991594254283246, -0.000623077121092687, -0.003289908601915284], 
                                        [-0.010059803423311134, 0.0010362522169301642, 0.9999488619606629, 0.0026798262366239016], 
                                        [0.0, 0.0, 0.0, 1.0]])
        else:
            posetopic = f"/davis/{side}/pose"
            T_marker_cam0 = np.eye(4)
            T_cam0_cam1 = np.eye(4)


        tss_imgs_us = read_tss_us_from_rosbag(bag, topics[imgtopic_idx])
        assert len(tss_imgs_us) == len(imgs)
        poses, tss_gt_us = read_poses_from_rosbag(bag, posetopic, T_marker_cam0, T_cam0_cam1=T_cam0_cam1)
        t0_evs = read_t0us_evs_from_rosbag(bag, topics[evtopic_idx])
        assert sorted(tss_imgs_us) == tss_imgs_us
        assert sorted(tss_gt_us) == tss_gt_us

        t0_us = np.minimum(np.minimum(tss_gt_us[0], tss_imgs_us[0]), t0_evs)
        tss_imgs_us = [t - t0_us for t in tss_imgs_us]

        # saving tss
        f = open(os.path.join(indir, f"tss_imgs_us_{side}.txt"), 'w')
        for t in tss_imgs_us:
            f.write(f"{t:.012f}\n")
        f.close()

        tss_gt_us = [t - t0_us for t in tss_gt_us]
        write_gt_stamped(poses, tss_gt_us, os.path.join(indir, f"gt_stamped_{side}.txt"))

        # TODO: write events (and also substract t0_evs)
        evs = read_evs_from_rosbag(bag, topics[evtopic_idx], H=H, W=W)
        f = open(os.path.join(indir, f"evs_{side}.txt"), 'w')
        for i in range(evs.shape[0]):
            f.write(f"{(evs[i, 2] - t0_us):.04f} {int(evs[i, 0])} {int(evs[i, 1])} {int(evs[i, 3])}\n")
        f.close()

        ######## [DEBUG] viz undistorted events
        
        
        # outvizfolder = os.path.join(indir, f"evs_{side}_undist")
        # os.makedirs(outvizfolder, exist_ok=True)
        # pbar = tqdm.tqdm(total=len(tss_imgs_us)-1)
        # for (ts_idx, ts_us) in enumerate(tss_imgs_us):
        #     if ts_idx == len(tss_imgs_us) - 1:
        #         break
            
        #     if DELTA_MS is None:
        #         evs_idx = np.where((evs[:, 2] >= ts_us) & (evs[:, 2] < tss_imgs_us[ts_idx+1]))[0]
        #     else:
        #         evs_idx = np.where((evs[:, 2] >= ts_us) & (evs[:, 2] < ts_us + DELTA_MS*1e3))[0]
                
        #     if len(evs_idx) == 0:
        #         print(f"no events in range {ts_us*1e-3} - {tss_imgs_us[ts_idx+1]*1e-3} milisecs")
        #         continue
        #     evs_batch = np.array(evs[evs_idx, :]).copy()


        #     img = render(evs_batch[:, 0], evs_batch[:, 1], evs_batch[:, 3], H, W)
        #     imfnmae = os.path.join(outvizfolder, f"{ts_idx:06d}.png")
        #     cv2.imwrite(imfnmae, img)

        #     if H != 260 and W != 346:
        #         rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]
        #         img = render(rect[:, 0], rect[:, 1], evs_batch[:, 3], H, W)
            
        #         imfnmae = imfnmae.split(".")[0] + "_undist.png"
        #         cv2.imwrite(os.path.join(outvizfolder, imfnmae), img)

        #     pbar.update(1)
        ############ [end DEBUG] viz undistorted events

        print(f"Finshied processing {indir}\n\n")
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP ECD data in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    args = parser.parse_args()

    roots = []
    for root, dirs, files in os.walk(args.indir):
        for f in files:
            if f.endswith(".bag"):
                p = os.path.join(root, f"{f.split('.')[0]}")
                os.makedirs(p, exist_ok=True)
                if p not in roots:
                    roots.append(p)

    
    cors = 3
    assert cors <= 9
    roots_split = np.array_split(roots, cors)

    processes = []
    for i in range(cors):
        p = multiprocessing.Process(target=process_dirs, args=(roots_split[i].tolist(), ))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print(f"Finished processing all RPG scenes")
