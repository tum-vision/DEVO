import numpy as np
import os
import argparse
import cv2
import tqdm
import glob
import esim_torch
import torch
import h5py
import tensorflow as tf
import time
import sys
import hdf5plugin

H = 480
W = 640
NBINS = 5

def render(x, y, pol, H, W):
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img

def to_voxel_grid_numpy(events, num_bins, width=W, height=H):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert (num_bins > 0) and (num_bins < 1000)
    assert(width > 0) and (width < 2048)
    assert(height > 0) and (height < 2048)
    events = events.astype(np.float32)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1]
    ys = events[:, 2]
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, (xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height).astype(np.int64), vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, (xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height).astype(np.int64), vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def ignore_files(directory, files):
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def save_evs_to_h5(xs, ys, ts, ps, evs_file_path, Cneg=0.2, Cpos=0.2, refractory_period_ns=0, compr="gzip", compr_lvl=4):
    compression = {
        "compression": compr,
        "compression_opts": compr_lvl,
    }

    with h5py.File(evs_file_path, "w") as f:
        f.create_dataset("x", data=xs, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE)) 
        f.create_dataset("y", data=ys, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE)) 
        f.create_dataset("t", data=ts, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE)) 
        f.create_dataset("p", data=ps, **hdf5plugin.Blosc(cname='zstd', clevel=compr_lvl, shuffle=hdf5plugin.Blosc.SHUFFLE))  
        # TODO: ms_to_idx

    print(f"Saved {len(xs)} events to {evs_file_path}.")
    return

def to_voxel_grid(xs, ys, ts, ps, nb_of_time_bins=5, remapping_maps=None):
    """Returns voxel grid representation of event steam.

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    If event stream is empty, voxel grid will be empty.
    """
    voxel_grid = torch.zeros(nb_of_time_bins,
                          H,
                          W,
                          dtype=torch.float32,
                          device='cpu')

    voxel_grid_flat = voxel_grid.flatten()
    ps = ps.astype(np.int8)
    ps[ps == 0] = -1

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = ts[-1] - ts[0]
    start_timestamp = ts[0]
    features = torch.from_numpy(np.stack([xs.astype(np.float32), ys.astype(np.float32), ts, ps], axis=1))
    x = features[:, 0]
    y = features[:, 1]
    polarity = features[:, 3].float()
    t = (features[:, 2] - start_timestamp) * (nb_of_time_bins - 1) / duration  # torch.float64
    t = t.to(torch.float64)

    if remapping_maps is not None:
        remapping_maps = torch.from_numpy(remapping_maps)
        x, y = remapping_maps[:,y,x]

    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= W-1) \
                       & (lim_y <= H-1) & (lim_t <= nb_of_time_bins-1)

                # we cast to long here otherwise the mask is not computed correctly
                lin_idx = lim_x.long() \
                          + lim_y.long() * W \
                          + lim_t.long() * W * H

                weight = polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    return voxel_grid

def save_voxels_to_h5(voxel, evs_file):
    voxel_float16 = voxel.to(torch.float16)
    with h5py.File(evs_file, "w") as f:
        f.create_dataset("voxel", data=voxel_float16, **hdf5plugin.Blosc(cname='zstd', clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE))

def convert_sequence(root, stereo="left"):
    assert stereo == "left" or stereo == "right"
    upimgs = root.replace(f"image_{stereo}", f"image_{stereo}_up")
    evs_dir = root.replace(f"image_{stereo}", f"evs_{stereo}")
    img_dir = os.path.join(root, "imgs")

    if os.path.isfile(os.path.join(root, "converted.txt")):
        print(f"Already converted {root}")
        return       
    print(f"Converting {root} to {upimgs} and {evs_dir}")  

    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.set_logical_device_configuration(
        gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=40000)]
    )

    if not os.path.exists(upimgs):
        cmd = f"python upsampling/upsample.py --input_dir={root} --output_dir={upimgs}"
        os.system(f"{cmd}")
        print(f"Upsampled {img_dir} to {upimgs}.")
    else:
        num_upimgs = len(glob.glob(os.path.join(upimgs, "imgs/*.png")))
        if num_upimgs > 0:
            num_imgs = len(glob.glob(os.path.join(img_dir, "*.png")))
            assert num_upimgs / 2 > num_imgs
            if not os.path.isfile(os.path.join(upimgs, "timestamps.txt")):
                cmd = f"rm -rf {upimgs}" 
                os.system(f"{cmd}")
                print(f"Removed high-fps images {upimgs} - because not complete (not timestamps.txt found). Recreating:")

                cmd = f"python upsampling/upsample.py --input_dir={root} --output_dir={upimgs}"
                os.system(f"{cmd}")
                print(f"Upsampled {img_dir} to {upimgs}.")
            else:
                print(f"Already upsampled {num_imgs} images to {num_upimgs}")
    time.sleep(5)

    # create events
    C = 0.25
    dC = 0.09
    Cneg = np.random.uniform(C-dC, C+dC)
    Cpos = np.random.uniform(C-dC, C+dC)

    os.makedirs(evs_dir, exist_ok=True)
    cmd = f"touch {evs_dir}/C.txt"
    os.system(f"{cmd}")
    cmd = f"echo {Cneg} {Cpos} > {evs_dir}/C.txt"
    os.system(f"{cmd}")
    refractory_period_ns = 0  # TODO: sample refractory?
    
    simulator = esim_torch.ESIM(
        contrast_threshold_neg=Cneg,
        contrast_threshold_pos=Cpos,
        refractory_period_ns=refractory_period_ns,  
    )
    
    image_files = sorted(glob.glob(os.path.join(upimgs, "imgs/*.png")))
    tss_ns = (np.loadtxt(os.path.join(upimgs, "timestamps.txt"), dtype=np.float64)*1e9).astype(np.int64)
    tss_ns = torch.from_numpy(tss_ns).cuda()
    fps_imgs_s = np.loadtxt(os.path.join(img_dir, "../fps.txt"), dtype=np.float64).item()
    N_images = len(glob.glob(os.path.join(img_dir, "*.png")))
    tss_imgs_ns = (np.arange(start=0, stop=N_images)*1e9/fps_imgs_s).astype(np.int64) # [0, N_images-1]
    assert np.abs(tss_ns[0].item() - tss_imgs_ns[0]) < 10000000  # 10ms
    assert np.abs(tss_ns[-1].item() - tss_imgs_ns[-1]) < 10000000 # 10ms
    print(f"dbegin: {tss_ns[0].item() - tss_imgs_ns[0]}, dend: {tss_ns[-1].item() - tss_imgs_ns[-1]}")

    # save img_tss and img_up_tss
    cmd = f"cp {upimgs}/timestamps.txt {evs_dir}/tss_upimgs_sec.txt"
    os.system(f"{cmd}")
    f = open(os.path.join(evs_dir, "tss_imgs_sec.txt"), "w")
    for ts_img_ns in tss_imgs_ns:
        f.write(f"{ts_img_ns/1e9}\n")
    f.close()
    
    pbar = tqdm.tqdm(total=len(image_files)-1)
    num_events = 0

    img_right_counter = 1 # start at 1 because the first image is not useds
    xs, ys, ts, ps = [], [], [], []
    os.makedirs(os.path.join(evs_dir, "h5"), exist_ok=True)
    os.makedirs(os.path.join(evs_dir, "viz"), exist_ok=True)
    for image_file, ts_ns in zip(image_files, tss_ns):
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        log_image = np.log(image.astype("float32") / 255 + 1e-5)
        log_image = torch.from_numpy(log_image).cuda()

        sub_events = simulator.forward(log_image, ts_ns)

        # for the first image, no events are generated, so this needs to be skipped
        if sub_events is None: # TODO: handle case when no events are generated if img_ct_right > 1
            continue

        sub_events = {k: v.cpu() for k, v in sub_events.items()}    
        num_events += len(sub_events['t'])
 
        # do something with the events
        # np.savez(os.path.join(outdir, "%010d.npz" % counter), **sub_events)
        x = sub_events["x"].numpy().astype(np.int16) if sub_events else np.empty(0, dtype=np.int16)
        y = sub_events["y"].numpy().astype(np.int16) if sub_events else np.empty(0, dtype=np.int16)
        t = sub_events["t"].numpy().astype(np.int64) if sub_events else np.empty(0, dtype=np.int64)
        p = sub_events["p"].numpy().astype(np.int8) if sub_events else np.empty(0, dtype=np.int8)
        if t.max() <= tss_imgs_ns[img_right_counter] or img_right_counter == N_images - 1:
            xs.append(x)
            ys.append(y)
            ts.append(t)
            ps.append(p)
            if img_right_counter == N_images - 1:
                assert t.max() - tss_imgs_ns[img_right_counter] < 100000 # 100us = 0.1ms
                print(f"diff: {t.max() - tss_imgs_ns[img_right_counter]}ns")
        else:
            idx = np.where(t > tss_imgs_ns[img_right_counter])[0][0]
            xs.append(x[:idx])
            ys.append(y[:idx])
            ts.append(t[:idx])
            ps.append(p[:idx])

            xs = np.concatenate(np.array(xs, dtype=object)).astype(np.uint16)
            ys = np.concatenate(np.array(ys, dtype=object)).astype(np.uint16)
            ts = np.concatenate(np.array(ts, dtype=object)).astype(np.int64)
            ps = np.concatenate(np.array(ps, dtype=object)).astype(np.int8)

            evs_file = os.path.join(evs_dir, "h5", "%010d.h5" % img_right_counter)
            voxel = to_voxel_grid(xs, ys, ts, ps, nb_of_time_bins=NBINS)
            save_voxels_to_h5(voxel, evs_file)

            # save_evs_to_h5(xs, ys, ts, ps, evs_file_path=evs_file, Cneg=Cneg, Cpos=Cpos, refractory_period_ns=refractory_period_ns)
            img = render(xs, ys, ps, H=H, W=W)
            cv2.imwrite(os.path.join(evs_dir, "viz", "%010d.png" % img_right_counter), img)

            img_right_counter += 1
            xs = [x[idx:]]
            ys = [y[idx:]]
            ts = [t[idx:]]
            ps = [p[idx:]]

        pbar.set_description(f"Num events generated: {num_events}")
        pbar.update(1)
    
    time.sleep(5)
    print(f"img_right_counter = {img_right_counter}, N_images = {N_images}")
    if len(xs) > 0:
        if len(xs[0]) > 1:
            xs = np.concatenate(np.array(xs, dtype=object)).astype(np.uint16)
            ys = np.concatenate(np.array(ys, dtype=object)).astype(np.uint16)
            ts = np.concatenate(np.array(ts, dtype=object)).astype(np.int64)
            ps = np.concatenate(np.array(ps, dtype=object)).astype(np.int8)

            print(f"Saving last {len(xs)} events from {ts[0]}nsec to {ts[-1]}nsec.")
            print(f"tss_imgs_ns[-1]: {tss_imgs_ns[-1]}, tss_imgs_ns[-2]: {tss_imgs_ns[-2]}")
            # TODO: assert these are the correct events

            evs_file = os.path.join(evs_dir, "h5", "%010d.h5" % img_right_counter)
            voxel = to_voxel_grid(xs, ys, ts, ps, nb_of_time_bins=NBINS)
            save_voxels_to_h5(voxel, evs_file)

            img = render(xs, ys, ps, H=H, W=W)
            cv2.imwrite(os.path.join(evs_dir, "viz", "%010d.png" % img_right_counter), img)
            img_right_counter += 1

    if img_right_counter == N_images:
        print(f"Sucees: img_right_counter = {img_right_counter}, N_images - 1 = {N_images}. ")
    else:
        print(f"\n\n\n\nError**********! ")
        print(f"img_right_counter = {img_right_counter}, N_images = {N_images}")
        print(f"Error**********!\n\n\n\n")

    simulator.reset()
    time.sleep(5)

    # remove upsampled images
    cmd = f"rm -rf {upimgs}"
    os.system(f"{cmd}")
    print(f"Removed high-fps images {upimgs}.")

    cmd = f"touch {root}/converted.txt"
    os.system(f"{cmd}")
    return

def main():
    parser = argparse.ArgumentParser(description="Raw to png images in dir")
    parser.add_argument("--dirsfile", help="Input raw dir.", default="/DIRECTORY/test.txt")

    args = parser.parse_args()
    assert ".txt" in args.dirsfile
    print(f"config file = {args.dirsfile}")

    # ROOTS = []
    # for root, dirs, files in os.walk("/DIRECTORY_FOR_TARTAN/tartan/"):
    #     for name in files:
    #         if np.any([".zip" in f for f in files]):
    #             assert "depth_left.zip" in files
    #             # assert "image_left.zip" in files
    #             print(os.path.join(root, name))
                
    #             img_zip_file = os.path.join(root, 'image_left.zip')
    #             imgdir = os.path.join(root, 'image_left')
    #             if not os.path.exists(imgdir):
    #                 os.makedirs(os.path.join(root, "image_left"), exist_ok=True)
    #                 cmd = f"unzip {img_zip_file} -d {imgdir}"
    #                 os.system(f"{cmd}")
    #                 cmd = f"rm {img_zip_file}" 
    #                 os.system(f"{cmd}")
    #                 print(f"Extracted and removed {img_zip_file}.")
                
    #             for r, d, fs in os.walk(imgdir):
    #                 if "pose_left.txt" in fs and "pose_right.txt" in fs:
    #                     cmd = f"mv {r}/image_left/ {r}/imgs/"
    #                     if os.path.isdir(f"{r}/imgs"):
    #                         with open(os.path.join(r, f"{r}/fps.txt"), "w") as f:
    #                             f.write(f"10\n")
    #                         assert not os.path.isdir(f"{r}/image_left*")
    #                         continue
    #                     os.system(f"{cmd}")

    #                     cmd = f"touch {r}/fps.txt"
    #                     os.system(f"{cmd}")
    #                     with open(os.path.join(r, f"{r}/fps.txt"), "w") as f:
    #                         f.write(f"10\n")

    #             print(f"Put fps.txt (10fps) files in subdirs of {imgdir}. Created 'imgs' subdirs")
    #             evs_dir = os.path.join(root, 'evs_left')
    #             os.makedirs(evs_dir, exist_ok=True)

    #             for r, d, fs in os.walk(imgdir):
    #                 if "pose_left.txt" in fs and "pose_right.txt" in fs:
    #                     if len(ROOTS) > 0:
    #                         assert r not in ROOTS
    #                     ROOTS.append(r)
        
    # print(f"Collected all commands and folders")

    # Split Creation
    # file = open("/DIRECTORY/roots.txt", "w")
    # for R in ROOTS:
    #     file.write(R + "\n")
    # file.close()

    # splits = np.array_split(np.array(ROOTS), 8)
    # for i, split in enumerate(splits):
    #     file = open("/DIRECTORY/roots"+f"_{i}.txt", "w")
    #     for s in split.tolist():
    #         file.write(s + "\n")
    #     file.close()
     

    file = open(f"{args.dirsfile}", "r")
    ROOTS = file.read().splitlines()
    file.close()
    print(f"convert_tartan.py: Processing {len(ROOTS)} dirs: {ROOTS}")

    print('A', sys.version)
    print('B', torch.__version__)
    print('C', torch.cuda.is_available())
    print('D', torch.backends.cudnn.enabled)
    device = torch.device('cuda')
    print('E', torch.cuda.get_device_properties(device))
    print('F', torch.tensor([1.0, 2.0]).cuda())

    cmd = "nvcc --version"
    os.system(f"{cmd}")
    cmd = "module load cuda/11.2"
    os.system(f"{cmd}")
    cmd = "module load cudnn"
    os.system(f"{cmd}")

    for i in range(len(ROOTS)):
        print(f"\n\nconvert_tartan.py: Start processing {ROOTS[i]}")
        convert_sequence(ROOTS[i], stereo="left")
        print(f"convert_tartan.py: Finished processing {ROOTS[i]}\n\n")


if __name__ == "__main__":
    main()
