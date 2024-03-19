import numpy as np
import os
import argparse
import cv2
import tqdm
import shutil
# import yaml
import ruamel.yaml

def prepare_seq_eds(indir, calibstr="calib0"):
    assert calibstr == "calib0" or calibstr == "calib1"
    print(f"Processing {indir}")

    imgdirin = os.path.join(indir, f"images_undistorted_{calibstr}")
    assert os.path.isdir(imgdirin)
    orb3out = os.path.join(indir, f"ORB3_{calibstr}")
    os.makedirs(orb3out, exist_ok=True)
    imgdirout = os.path.join(orb3out, "data")
    os.makedirs(imgdirout, exist_ok=True)

    img_list = sorted(os.listdir(imgdirin))
    img_list = [os.path.join(indir, imgdirin, im) for im in img_list if im.endswith(".png")]
    H, W, _ = cv2.imread(img_list[0]).shape
    assert W == 640
    assert H == 480

    img_list2 = sorted(os.listdir(imgdirout))
    if len(img_list2) > 0:
        if len(img_list2) == len(img_list):
            print(f"Images already copied for {indir}, skipping")
            return

    tss_imgs_us = np.loadtxt(os.path.join(indir, "images_timestamps.txt"), skiprows=0)
    tss_imgs_ns = tss_imgs_us * 1e3
    FPS = 1e9/np.mean(np.diff(tss_imgs_ns))

    with open(os.path.join(orb3out, "images_tss_ns.txt"), 'w') as f:
        for ts_ns in tss_imgs_ns:
            f.write(f"{int(ts_ns)}\n")

    yamloutfn = os.path.join(orb3out, f"EDS_{calibstr}_tum.yaml")
    shutil.copy(f"path/EDS_{calibstr}_tum.yaml", yamloutfn)
    yaml = ruamel.yaml.YAML()
    with open(yamloutfn, 'r') as file:
        SETT = yaml.load(file)
    SETT["Camera.fps"] = int(FPS)
    with open(yamloutfn, 'w') as file:
        yaml.dump(SETT, file)

    gtfn = os.path.join(orb3out, "groundtruth_ns.txt")
    shutil.copy(os.path.join(indir, "stamped_groundtruth.txt"), os.path.join(gtfn))
    gt_s = np.loadtxt(gtfn, skiprows=1)
    gt_ns = gt_s
    gt_ns[:, 0] = gt_ns[:, 0] * 1e9
    np.savetxt(gtfn, gt_ns, fmt="%f", delimiter=",")

    # 2) undistorting images
    pbar = tqdm.tqdm(total=len(img_list))
    for i, f in enumerate(img_list):
        image = cv2.imread(f)
        cv2.imwrite(os.path.join(imgdirout, f"{int(tss_imgs_ns[i]):019d}.png"), image)
        assert (int(tss_imgs_ns[i])-tss_imgs_ns[i]) == 0
        pbar.update(1)

    print(f"Finsh Preparing {indir} for ORB3\n\n")
    
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
                    
    roots = sorted(roots)
    roots = roots 
    for root in roots:
        print(f"Processing {root}")
        prepare_seq_eds(root)

