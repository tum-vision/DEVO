import numpy as np
import os
import argparse
import cv2
import tqdm
import json
import shutil
import h5py
import hdf5plugin
import glob

from utils.load_utils import EventSlicer
from utils.viz_utils import render

CAM_ID = 0


def match_calib(indir):
    if "loop-floor" in indir or "mocap-desk" in indir or "mocap-desk2" in indir or "skate-easy" in indir:
        return os.path.join(indir, "../camera-calibrationA.json")
    else:
        return os.path.join(indir, "../camera-calibrationB.json")

def process_dir(indir, camId=0):    
    print(f"\n \n Undistorting {indir} cam {camId} intrinsics, rgbs and evs")
    camId = camId
    assert camId == 0 or camId == 1
    if camId == 0:
        imgdir = os.path.join(indir, "left_images")
        imgdirout = os.path.join(indir, "left_images_undistorted")
    if camId == 1:
        imgdir = os.path.join(indir, "right_images")
        imgdirout = os.path.join(indir, "right_images_undistorted")

    img_list = [os.path.join(indir, imgdir, im) for im in sorted(os.listdir(imgdir)) if im.endswith(".jpg")]
    if len(img_list) == 0:
        img_list = [os.path.join(indir, imgdir, im) for im in sorted(os.listdir(imgdir)) if im.endswith(".png")]
    H, W, _ = cv2.imread(img_list[0]).shape

    if not os.path.exists(imgdirout):
        os.makedirs(imgdirout)
    else:
        img_list_undist = [os.path.join(indir, imgdirout, im) for im in sorted(os.listdir(imgdirout)) if im.endswith(".jpg")]
        if len(img_list) == len(img_list_undist):
            print(f"\n\nWARNING **** Images already undistorted. Skipping {indir} ***** \n\n")
            assert os.path.isfile(os.path.join(indir, "rectify_map_left.h5"))
            return

    # transforming intrinsics
    calibfile = match_calib(indir)
    shutil.copy(calibfile, os.path.join(indir, "calibration.json"))
                
    with open(os.path.join(indir, "calibration.json"), 'r') as f:
        calibdata = json.load(f)
        intr_undist = []
        for i in range(4): # loop all 4 cameras (0=left, 1=right, 2=left events, 3=right events)
            K = np.zeros((3,3)) 
            K[0,0] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["fx"]
            K[0,2] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["cx"]
            K[1,1] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["fy"]
            K[1,2] = calibdata["value0"]["intrinsics"][i]["intrinsics"]["cy"]
            K[2, 2] = 1

            k1 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k1"]
            k2 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k2"]
            k3 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k3"]
            k4 = calibdata["value0"]["intrinsics"][i]["intrinsics"]["k4"]
            dist_coeffs = np.asarray([k1, k2, k3, k4])

            W = calibdata["value0"]["resolution"][i][0]
            H = calibdata["value0"]["resolution"][i][1]
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, dist_coeffs, (W, H), np.eye(3), balance=0.5)
            intr_undist.append({"fx": K_new[0,0], "fy": K_new[1,1], "cx": K_new[0,2], "cy": K_new[1,2]})
        
            if i == 2 and camId == 0 or i == 3 and camId == 1:
                # reading event resolution
                xs, ys = np.meshgrid(np.arange(W), np.arange(H))
                xys = np.stack((xs, ys), axis=-1) # (H, W, 2)
                xys_remap = cv2.fisheye.undistortPoints(xys.astype(np.float32), K, dist_coeffs, R=np.eye(3), P=K_new)

                h5outfile = os.path.join(indir, "rectify_map_left.h5") if i == 2 else os.path.join(indir, "rectify_map_right.h5")
                ef_out = h5py.File(h5outfile, 'w') 
                ef_out.clear()
                ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
                ef_out["rectify_map"][:] = xys_remap
                ef_out.close()

                # [debug]: visualize undistortion
                h5file = glob.glob(os.path.join(indir, '*events_left.h5')) if i == 2 else glob.glob(os.path.join(indir, '*events_right.h5'))
                if len(h5file) == 0:
                    continue
                else:
                    h5file = h5file[0]
                ef_in = h5py.File(os.path.join(indir, h5file), "r")
                outvizfolder = os.path.join(indir, "all_evs_left_undist_viz") if i == 2 else os.path.join(indir, "all_evs_right_undist_viz")
                os.makedirs(outvizfolder, exist_ok=True)

                event_slicer = EventSlicer(ef_in)
                N_slices = int((ef_in["events"]["t"][-1]-ef_in["events"]["t"][0])/1e6*20)
                tss_slice_us = np.linspace(ef_in["events"]["t"][0], ef_in["events"]["t"][-1], N_slices)
                print(f"Visualizing undistorted events")
                pbar = tqdm.tqdm(total=len(tss_slice_us)-1)
                for j in range(len(tss_slice_us)-1):
                    start_time_us = tss_slice_us[j]
                    end_time_us = start_time_us + 10000 # visualizing 10ms
                    ev_batch = event_slicer.get_events(start_time_us, end_time_us)
                    if ev_batch is None:
                        continue
                    p = ev_batch['p']
                    x = ev_batch['x']
                    y = ev_batch['y']
                    img = render(x, y, p, H, W)
                    cv2.imwrite(os.path.join(outvizfolder,  "%06d" % j + ".png"), img)
                    rect = xys_remap[y, x]
                    x_rect = rect[..., 0]
                    y_rect = rect[..., 1]
                    img = render(x_rect, y_rect, p, H, W)
                    cv2.imwrite(os.path.join(outvizfolder,  "%06d_undist" % j + ".png"), img)
                    pbar.update(1)
                ef_in.close()

    with open(os.path.join(indir, "calib_undist.json"), 'w') as f:
        calibdata["value0"]["intrinsics_undistorted"] = intr_undist
        json.dump(calibdata, f)
        
    calibdata = calibdata["value0"]
    W = calibdata["resolution"][camId][0]
    H = calibdata["resolution"][camId][1]

    # reading visual camera data
    k1 = calibdata["intrinsics"][camId]["intrinsics"]["k1"]
    k2 = calibdata["intrinsics"][camId]["intrinsics"]["k2"]
    k3 = calibdata["intrinsics"][camId]["intrinsics"]["k3"]
    k4 = calibdata["intrinsics"][camId]["intrinsics"]["k4"]
    dist_coeffs = np.asarray([k1, k2, k3, k4])
    
    K = np.zeros((3,3)) 
    K[0,0] = calibdata["intrinsics"][camId]["intrinsics"]["fx"]
    K[0,2] = calibdata["intrinsics"][camId]["intrinsics"]["cx"]
    K[1,1] = calibdata["intrinsics"][camId]["intrinsics"]["fy"]
    K[1,2] = calibdata["intrinsics"][camId]["intrinsics"]["cy"]
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, dist_coeffs, (W, H), np.eye(3), balance=0) 

    print(f"Undistorting images of camId {camId}")
    pbar = tqdm.tqdm(total=len(img_list))
    for f in img_list:
        image = cv2.imread(f)
        img = cv2.fisheye.undistortImage(image, K, dist_coeffs, Knew=K_new, new_size=(W, H))
        cv2.imwrite(os.path.join(imgdirout, os.path.split(f)[1]), img)
        pbar.update(1)
        # for debugging: 
        # cv2.imwrite(os.path.join(imgdirout,  os.path.split(f)[1][:-4] + "_undist.jpg"),  image)

    if camId == 0:
        shutil.copy(os.path.join(imgdir, "image_exposures_left.txt"), os.path.join(imgdirout, "image_exposures_left.txt"))
        shutil.copy(os.path.join(imgdir, "image_timestamps_left.txt"), os.path.join(imgdirout, "image_timestamps_left.txt"))
    else:
        assert camId == 1
        shutil.copy(os.path.join(imgdir, "image_exposures_right.txt"), os.path.join(imgdirout, "image_exposures_right.txt"))
        shutil.copy(os.path.join(imgdir, "image_timestamps_right.txt"), os.path.join(imgdirout, "image_timestamps_right.txt"))
    print(f"Done undistorting {indir} cam {camId} \n\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resizes images in dir")
    parser.add_argument(
        "--indir", help="Input image directory.", default=""
    )
    parser.add_argument(
        "--camId", help="0 == left visual cam, 1 == right visual cam", default=0, type=int
    )
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.indir):
        for d in dirs:
            if "left_images" in d and "mocap_data.txt" in files:
                process_dir(root, camId=args.camId)
