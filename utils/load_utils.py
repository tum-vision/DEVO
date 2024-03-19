
import torch
import numpy as np
import glob
import cv2
import os.path as osp
import torch.utils.data as data
import multiprocessing
import time
import h5py
import os
import hdf5plugin # required for import of blosc
import json
import torch.nn.functional as F

import torchvision

from utils.event_utils import EventSlicer, to_voxel_grid, RemoveHotPixelsVoxel
from utils.viz_utils import visualize_voxel, visualize_N_voxels, render
from utils.transform_utils import transform_rescale

def load_intrinsics_tumvie(path, camID=0):
    with open(os.path.join(path, "calib_undist.json"), 'r') as f:
        calibdata = json.load(f)["value0"]
    calibdata = calibdata["intrinsics_undistorted"][camID]
    fx, fy, cx, cy = calibdata["fx"], calibdata["fy"], calibdata["cx"], calibdata["cy"]
    intrinsics = [fx, fy, cx, cy]
    return intrinsics

def load_rmap_tumvie(path, side="left", H=720, W=1280):
    h5file = glob.glob(osp.join(path, f"rectify_map_{side}.h5"))[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    assert rectify_map.shape == (H, W, 2)
    rmap.close()
    return rectify_map

def change_intrinsics_resize(intrinsics, H, W, H_orig=720, W_orig=1280):
    fx, fy, cx, cy = intrinsics
    fx = fx * W / W_orig
    fy = fy * H / H_orig
    cx = cx * W / W_orig
    cy = cy * H / H_orig
    intrinsics = [fx, fy, cx, cy]
    return intrinsics

def read_batch_as_voxel(evs_slicer, t0_us, t1_us, rectify_map, trafos, Horig, Worig, Nbins=5):
    ev_batch = evs_slicer.get_events(t0_us, t1_us)
    if ev_batch is None:
        return None
    if len(ev_batch["t"]) == 0:
        print(f"lens: {len(ev_batch['x'])}, {len(ev_batch['y'])}, {len(ev_batch['t'])}, {len(ev_batch['p'])}\n")
        return None

    rect = rectify_map[ev_batch["y"], ev_batch["x"]]
    voxel = to_voxel_grid(rect[..., 0], rect[..., 1], ev_batch["t"], ev_batch["p"], H=Horig, W=Worig, nb_of_time_bins=Nbins)

    if trafos is not None:
        for t in trafos:
            voxel = t(voxel)
    # visualize_voxel(voxel.detach(), EPS=1e-3)
    return voxel

def get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig, Worig):
    data_list = []
    for i, ts_us in enumerate(tss_imgs_us): 
        t0_us, t1_us = ts_us, ts_us + dT_ms*1e3

        voxel = read_batch_as_voxel(evs_slicer, t0_us, t1_us, rectify_map, trafos, Horig, Worig)
        if voxel is None:
            print(f"Found no events in {(t0_us)/1e6:.3f}secs to {(t1_us)/1e6:.3f}secs at frame-idx {i}.jpg")
            continue

        intrinsics = torch.as_tensor(intrinsics).clone()
        data_list.append((voxel, intrinsics, (t0_us+t1_us)/2))
    return data_list


def get_real_data_list_parallel(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig, Worig, return_dict):
    data_list = get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig, Worig)
    return_dict.update({tss_imgs_us[0]: data_list})
    

def tumvie_evs_iterator(scenedir, camID=2, stride=1, rectify_map=None, H=720, W=1280, dT_ms=None, timing=False, parallel=False, cors=16):
    assert camID == 2 or camID == 3
    side = "left" if camID == 2 else "right"
    
    intrinsics = load_intrinsics_tumvie(scenedir, camID=camID)
    rectify_map = load_rmap_tumvie(scenedir, side=side)

    h5file = glob.glob(osp.join(scenedir, f"*events_{side}.h5"))[0]
    evs = h5py.File(h5file, "r")
    evs_slicer = EventSlicer(evs)

    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"{side}_images_undistorted", f"image_timestamps_{side}.txt")))
    if dT_ms is None:
        dT_ms = np.diff(tss_imgs_us).mean()/1e3   
    assert dT_ms > 3 and dT_ms < 100
    tss_imgs_us = tss_imgs_us[::stride]

    trafos = []
    if H != 720 or W != 1280:
        resize = torchvision.transforms.Resize((H, W))
        intrinsics = change_intrinsics_resize(intrinsics, H, W, Horig=720, Worig=1280)
        print(f"Warning: Resizing tumvie voxels to ({H}, {W}). Visualize and check")
    else:
        resize = lambda x: x
    trafos.append(resize)

    hotpixfilter = True
    if hotpixfilter:
        trafos.append(RemoveHotPixelsVoxel(num_stds=6))

    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    if not parallel:
        data_list = get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig=720, Worig=1280)
    else:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        processes = []
        return_dict = multiprocessing.Manager().dict()
        for i in range(cors):
            p = multiprocessing.Process(target=get_real_data_list_parallel, args=(evs_slicer, tss_imgs_us_split[i].tolist(), intrinsics, rectify_map, trafos, dT_ms, 720, 1280, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        
        if timing:
            t0sort = torch.cuda.Event(enable_timing=True)
            t1sort = torch.cuda.Event(enable_timing=True)
            t0sort.record()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])

        if timing:
            t1sort.record()
            torch.cuda.synchronize()
            print(f"Sorted {len(data_list)} tumvie voxels in {t0sort.elapsed_time(t1sort)/1e3} secs")

    if timing:  
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} TUMVIE voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} TUMVIE voxels, imstart={0}, imstop={-1}, stride={stride}, dT_ms={dT_ms} on {scenedir}")

    evs.close()

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us

def load_rmap_eds(scenedir, calib1=False):
    calibstr = "calib0" if not calib1 else "calib1"
    h5file = glob.glob(os.path.join(scenedir, f'rectify_map_{calibstr}.h5'))[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])
    assert rectify_map.shape == (480, 640, 2)
    rmap.close()
    return rectify_map

def load_intrinsics_eds(calib1=False):
    if not calib1:
        intrinsics = [562.9412231445312, 563.5700073242188, 310.53467429134616, 215.59711647292897]
        # RGB: [713.6517944335938, 737.5368041992188, 288.29385382423607, 226.97368855930836]
    elif calib1:
        intrinsics = [548.6773071289062, 551.0106201171875, 310.9592609123247, 218.11182443004145]
        # RGB: [704.6842041015625, 729.3246459960938, 286.39991203466707, 227.79612335843012]
    return intrinsics

def get_imstart_imstop_eds(indir):
    imstart, imstop = 0, -1

    return imstart, imstop

def eds_evs_iterator(scenedir, calib1=False, stride=1, timing=False, H=480, W=640, parallel=False, cors=16):
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
    
    intrinsics = load_intrinsics_eds(calib1=calib1)
    rectify_map = load_rmap_eds(scenedir, calib1=calib1)

    h5file = glob.glob(osp.join(scenedir, "events.h5"))[0]
    evs = h5py.File(h5file, "r") # x, y, t, p
    evs_slicer = EventSlicer(evs)

    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, "images_timestamps_us.txt")))
    dT_ms = np.diff(tss_imgs_us).mean()/2e3
    # if "02_rocket_earth_light" in scenedir:
    #     dT_ms = 2.0*dT_ms
    print(f"Mean dT_ms: {dT_ms}")
    # assert dT_ms > 3 and dT_ms < 60
    # assert abs(tss_imgs_us[0] - (evs["t"][0])) < 3e6 
    # assert abs(tss_imgs_us[-1] - (evs["t"][-1])) < 3e6
    imstart, imstop = 0, -1
    tss_imgs_us = tss_imgs_us[imstart:imstop:stride]

    trafos = []
    if H != 480 or W != 640:
        resize = torchvision.transforms.Resize((H, W))
        print(f"Warning: Resizing EDS voxels to ({H}, {W}). Visualize and check")
        intrinsics = change_intrinsics_resize(intrinsics, H, W, Horig=480, Worig=640)
    else:
        resize = lambda x: x
    trafos.append(resize)

    hotpixfilter = True
    if hotpixfilter:
        trafos.append(RemoveHotPixelsVoxel(num_stds=10))

    if not parallel:
        data_list = get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig=480, Worig=640)
    else:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        processes = []
        return_dict = multiprocessing.Manager().dict()
        for i in range(cors):
            p = multiprocessing.Process(target=get_real_data_list_parallel, args=(evs_slicer, tss_imgs_us_split[i].tolist(), intrinsics, rectify_map, trafos, dT_ms, 480, 640, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        
        if timing:
            t0sort = torch.cuda.Event(enable_timing=True)
            t1sort = torch.cuda.Event(enable_timing=True)
            t0sort.record()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])

        if timing:
            t1sort.record()
            torch.cuda.synchronize()
            print(f"Sorted {len(data_list)} EDS voxels in {t0sort.elapsed_time(t1sort)/1e3} secs")

    evs.close()

    if timing:  
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} EDS voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} EDS voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {scenedir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us

def eds_evs_loader(scenedir, calib1=False, stride=1, H=480, W=640, parallel=False, cors=16, timing=False):    
    intrinsics = load_intrinsics_eds(calib1=calib1)
    rectify_map = load_rmap_eds(scenedir, calib1=calib1)

    h5file = glob.glob(osp.join(scenedir, "events.h5"))[0]
    evs = h5py.File(h5file, "r") # x, y, t, p
    evs_slicer = EventSlicer(evs)

    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, "images_timestamps_us.txt")))
    dT_ms = np.diff(tss_imgs_us).mean()/1e3
    print(f"Mean dT_ms: {dT_ms}")
    # assert dT_ms > 3 and dT_ms < 60
    # assert abs(tss_imgs_us[0] - (evs["t"][0])) < 3e6 
    # assert abs(tss_imgs_us[-1] - (evs["t"][-1])) < 3e6
    imstart, imstop = 0, -1
    tss_imgs_us = tss_imgs_us[imstart:imstop:stride]

    trafos = []
    if H != 480 or W != 640:
        resize = torchvision.transforms.Resize((H, W))
        print(f"Warning: Resizing EDS voxels to ({H}, {W}). Visualize and check")
        intrinsics = change_intrinsics_resize(intrinsics, H, W, Horig=480, Worig=640)
    else:
        resize = lambda x: x
    trafos.append(resize)

    hotpixfilter = True
    if hotpixfilter:
        trafos.append(RemoveHotPixelsVoxel(num_stds=10))

    if not parallel:
        data_list = get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig=480, Worig=640)
    else:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        processes = []
        return_dict = multiprocessing.Manager().dict()
        for i in range(cors):
            p = multiprocessing.Process(target=get_real_data_list_parallel, args=(evs_slicer, tss_imgs_us_split[i].tolist(), intrinsics, rectify_map, trafos, dT_ms, 480, 640, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        
        if timing:
            t0sort = torch.cuda.Event(enable_timing=True)
            t1sort = torch.cuda.Event(enable_timing=True)
            t0sort.record()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])

    evs.close()
    print(f"Preloaded {len(data_list)} EDS voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {scenedir}")

    return data_list



def read_image_to_tensor(fname):
    return torch.from_numpy(cv2.imread(fname)).permute(2,0,1)

def video_iterator(imagedir, tss_file=None, ext=".png", intrinsics=[320, 320, 320, 240], stride=1, timing=False, scale=1.0):
    imfiles = sorted(glob.glob(osp.join(imagedir, "*{}".format(ext))))

    if tss_file is None:
        tss_imgs_us = np.arange(len(imfiles))
    else:
        tss_imgs_us = sorted(np.loadtxt(tss_file))

    imfiles = imfiles[::stride]
    tss_imgs_us = tss_imgs_us[::stride]
    assert len(imfiles) == len(tss_imgs_us)
    
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    data_list = []
    for (ts_us, imfile) in zip(tss_imgs_us, imfiles):
        image = read_image_to_tensor(imfile)
        intrinsics = torch.as_tensor(intrinsics)
        if scale != 1.0:
            image, _, _, intrinsics = transform_rescale(scale, voxels=image, intrinsics=intrinsics[None])
        data_list.append((image, intrinsics.squeeze(), ts_us))

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} images in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} images, imstart={0}, imstop={-1}, stride={stride} on {imagedir}")

    for (image, intrinsics, ts_us) in data_list:
        yield image.cuda(), intrinsics.cuda(), ts_us


def voxel_read(voxel_file):
    h5 = h5py.File(voxel_file, 'r')
    voxel = h5['voxel'][:]
    # assert voxel.dtype == np.float32 # (5, 480, 640)
    h5.close()
    return voxel

def read_voxel_data_from_list(voxfiles, tss_us, intrinsics, return_dict, scale=1.0):
    fx, fy, cx, cy = intrinsics
    
    data_list = []
    for (imfile, ts_us) in zip(voxfiles, tss_us):
        img = torch.from_numpy(voxel_read(imfile))
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        if scale != 1.0:
            img, _, _, intrinsics = transform_rescale(scale, voxels=img, intrinsics=intrinsics[None])
        data_list.append((img, intrinsics.squeeze(), ts_us))
    return_dict[int(os.path.splitext(voxfiles[0].split("/")[-1])[0])] = data_list


def voxel_iterator_parallel(voxeldir, tss_file=None, ext=".h5", intrinsics=[320, 320, 320, 240], stride=1, timing=True, CORS=16, scale=1.0):
    voxfiles = sorted(glob.glob(osp.join(voxeldir, "*{}".format(ext))))

    if tss_file is None:
        tss_imgs_us = np.arange(len(voxfiles))
    else:
        tss_imgs_us = sorted(np.loadtxt(tss_file))
     
    voxfiles = voxfiles[::stride]
    tss_imgs_us = tss_imgs_us[::stride]
    
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    sublists_voxfiles = np.array_split(voxfiles, CORS)
    tss_imgs_us_split = np.array_split(tss_imgs_us, CORS)
    processes = []
    return_dict = multiprocessing.Manager().dict()
    for i in range(CORS):
        p = multiprocessing.Process(target=read_voxel_data_from_list, args=(sublists_voxfiles[i].tolist(), tss_imgs_us_split[i].tolist(), intrinsics, return_dict, scale))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    
    if timing:
        t0sort = torch.cuda.Event(enable_timing=True)
        t1sort = torch.cuda.Event(enable_timing=True)
        t0sort.record()

    keys = np.array(return_dict.keys())
    order = np.argsort(keys)
    data_list = []
    for k in keys[order]:
        data_list.extend(return_dict[k])
    
    if timing:
        t1sort.record()
        torch.cuda.synchronize()
        print(f"Sorted {len(voxfiles)} voxels in {t0sort.elapsed_time(t1sort)/1e3} secs")
    
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} voxels,stride={stride} on {voxeldir}")

    for (image, intrinsics, ts_us) in data_list:
        yield image.cuda(), intrinsics.cuda(), ts_us



def voxel_iterator(voxeldir, tss_file=None, ext=".h5", intrinsics=[320, 320, 320, 240], stride=1, timing=False, scale=1.0):
    voxfiles = sorted(glob.glob(osp.join(voxeldir, "*{}".format(ext))))
    fx, fy, cx, cy = intrinsics

    if tss_file is None:
        tss_imgs_us = np.arange(len(voxfiles))
    else:
        tss_imgs_us = sorted(np.loadtxt(tss_file))

    voxfiles = voxfiles[::stride]
    tss_imgs_us = tss_imgs_us[::stride]

    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    data_list = []
    for (ts_us, voxfile) in zip(tss_imgs_us, voxfiles):
        voxel = torch.from_numpy(voxel_read(voxfile))
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        if scale != 1.0:
            voxel, _, _, intrinsics = transform_rescale(scale, voxels=voxel, intrinsics=intrinsics[None])
        data_list.append((voxel, intrinsics.squeeze(), ts_us))

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} voxels, imstart={0}, imstop={-1}, stride={stride}, {voxeldir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us


def read_video_data_from_list(imfiles, tss_us, intrinsics, return_dict, scale=1.0):
    fx, fy, cx, cy = intrinsics
    
    data_list = []
    for (imfile, ts_us) in zip(imfiles, tss_us):
        img = read_image_to_tensor(imfile)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        if scale != 1.0:
            img, _, _, intrinsics = transform_rescale(scale, voxels=img, intrinsics=intrinsics[None])
        data_list.append((img, intrinsics.squeeze(), ts_us))

    indx = os.path.splitext(imfiles[0].split("/")[-1])[0]
    if "frame_" in indx:  
        indx = indx.split("_")[-1]
    if "image_" in indx:  # for FPV dataset
        indx = indx.split("_")[-1]
    return_dict[int(indx)] = data_list
  

def video_iterator_parallel(imagedir, tss_file=None, ext=".png", intrinsics=[320, 320, 320, 240], stride=1, timing=False, cors=16, scale=1.0, tss_gt_us=None): 
    imfiles = sorted(glob.glob(osp.join(imagedir, "*{}".format(ext))))

    if tss_file is None:
        tss_imgs_us = np.arange(len(imfiles))
    else:
        tss_imgs_us = sorted(np.loadtxt(tss_file))

    imstart = 0
    imstop = -1
    if tss_gt_us is not None: # fix for FPV
        dT_imgs = tss_imgs_us[-1]-tss_imgs_us[0]
        dT_gt = tss_gt_us[-1]-tss_gt_us[0]
        if (dT_imgs - dT_gt) > 5*1e6 and (tss_gt_us[0] - tss_imgs_us[0]) > 5e6:
            imstart = np.where(tss_imgs_us > tss_gt_us[0])[0][0]
            imstop = np.where(tss_imgs_us < tss_gt_us[-1])[0][-1]
            print(f"Start reading images from {imstart}, {imstop}, due to much shorter GT")
    
    imfiles = imfiles[imstart:imstop:stride]
    tss_imgs_us = tss_imgs_us[imstart:imstop:stride]
    
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    data_list = []

    sublists_imfiles = np.array_split(imfiles, cors)
    tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
    processes = []
    return_dict = multiprocessing.Manager().dict()
    for i in range(cors):
        p = multiprocessing.Process(target=read_video_data_from_list, args=(sublists_imfiles[i].tolist(), tss_imgs_us_split[i].tolist(), intrinsics, return_dict, scale))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    
    if timing:
        ssort = torch.cuda.Event(enable_timing=True)
        endsort = torch.cuda.Event(enable_timing=True)
        ssort.record()

    keys = np.array(return_dict.keys())
    order = np.argsort(keys)
    data_list = []
    for k in keys[order]:
        data_list.extend(return_dict[k])
    
    if timing:
        endsort.record()
        torch.cuda.synchronize()
        print(f"Sorted {len(imfiles)} images in {ssort.elapsed_time(endsort)/1e3} secs")
    
        end.record()
        torch.cuda.synchronize()
        dt = start.elapsed_time(end)/1e3
        print(f"Preloaded {len(data_list)} images in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} images, imstart={imstart}, imstop={imstop}, stride={stride} on {imagedir}")

    for (image, intrinsics, ts_us) in data_list:
        yield image.cuda(), intrinsics.cuda(), ts_us


def load_mvsec_traj(scenedir, side="left"):
    from utils.pose_utils import poses_hom_to_quatlist
    gt_fname = os.path.join(scenedir, scenedir[:-5].split("/")[-1]+"_gt.hdf5")
    datain = h5py.File(gt_fname, 'r')

    traj_hf = datain["davis"][side]["pose"][:] # (N, 4, 4)
    traj_hf = poses_hom_to_quatlist(traj_hf) # (N, 7)
    tss_traj_us = datain["davis"][side]["pose_ts"][:].astype(np.float64)*1e6 # (N,)

    datain.close()

    return np.array(tss_traj_us.tolist()), np.array(traj_hf)

def load_eds_traj(path):
    traj_ref = np.loadtxt(path, delimiter=" ", skiprows=1)
    tss_gt_us = traj_ref[:, 0].copy()
    assert np.all(tss_gt_us == sorted(tss_gt_us))
    assert traj_ref.shape[0] > 0
    assert traj_ref.shape[1] == 8

    return tss_gt_us, traj_ref[:, 1:]


def load_tumvie_traj(path):
    traj_ref = np.loadtxt(path, delimiter=" ", skiprows=1)
    tss_gt_us = traj_ref[:, 0].copy()
    assert np.all(tss_gt_us == sorted(tss_gt_us))
    assert traj_ref.shape[0] > 0
    assert traj_ref.shape[1] == 8

    return tss_gt_us, traj_ref[:, 1:]

def load_intrinsics_ecd(path):
    path_undist = osp.join(path, "calib_undist.txt")
    
    intrinsics = np.loadtxt(path_undist)
    assert len(intrinsics) == 4
    return intrinsics

def load_intrinsics_rpg(p):  
    intrinsics = np.loadtxt(p)
    assert len(intrinsics) == 4
    return intrinsics

def load_ecd_gt(path, skiprows=0):
    traj_ref = np.loadtxt(path, delimiter=" ", skiprows=skiprows)
    tss_gt_us = traj_ref[:, 0].copy() * 1e6
    assert np.all(tss_gt_us == sorted(tss_gt_us))
    assert traj_ref.shape[0] > 0
    assert traj_ref.shape[1] == 8

    return tss_gt_us, traj_ref[:, 1:]

def load_gt_us(path, skiprows=0):
    traj_ref = np.loadtxt(path, delimiter=" ", skiprows=skiprows)
    tss_gt_us = traj_ref[:, 0].copy() 
    assert np.all(tss_gt_us == sorted(tss_gt_us))
    assert traj_ref.shape[0] > 0
    assert traj_ref.shape[1] == 8

    return tss_gt_us, traj_ref[:, 1:]

def read_ecd_tss(p, idx=0):
    f = open(p, "r")
    tss_secs = []
    for line in f.readlines():
        if line.startswith("#"):
            continue
        tss_secs.append(float(line.split(" ")[idx]))
    f.close()

    tss_us = 1e6 * np.asarray(tss_secs)
    assert np.all(sorted(tss_us) == tss_us)
    return tss_us

def get_ecd_data(tss_imgs_us, evs, intrinsics, rectify_map, DELTA_MS=None, H=180, W=240, return_dict=None):
    print(f"Delta {DELTA_MS} ms")
    data_list = []
    for (ts_idx, ts_us) in enumerate(tss_imgs_us):
        if ts_idx == len(tss_imgs_us) - 1:
            break
        
        if DELTA_MS is None:
            t0_us, t1_us = ts_us, tss_imgs_us[ts_idx+1]
        else:
            t0_us, t1_us = ts_us, ts_us + DELTA_MS*1e3
        evs_idx = np.where((evs[:, 0] >= t0_us) & (evs[:, 0] < t1_us))[0]
             
        if len(evs_idx) == 0:
            print(f"no events in range {ts_us*1e-3} - {tss_imgs_us[ts_idx+1]*1e-3} milisecs")
            continue
        evs_batch = np.array(evs[evs_idx, :]).copy()

        if rectify_map is not None:
            rect = rectify_map[evs_batch[:, 2].astype(np.int32), evs_batch[:, 1].astype(np.int32)]
            voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 0], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        else:
            voxel = to_voxel_grid(evs_batch[:, 1], evs_batch[:, 2], evs_batch[:, 0], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        # visualize_voxel(voxel)
        # img = render(evs_batch[:, 1], evs_batch[:, 2], evs_batch[:, 3], 180, 240) # 
        data_list.append((voxel, intrinsics, min((t0_us+t1_us)/2, tss_imgs_us[ts_idx+1])))

    if return_dict is not None:
        return_dict.update({tss_imgs_us[0]: data_list})
    else:
        return data_list

def read_rmap(rect_file, H=180, W=240):
    h5file = glob.glob(rect_file)[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    assert rectify_map.shape == (H, W, 2)
    rmap.close()
    return rectify_map

def compute_rmap_ecd(scenedir, H=180, W=240):
    intrinsics = np.loadtxt(osp.join(scenedir, "calib.txt"))
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics

    K_evs =  np.zeros((3,3))        
    K_evs[0,0] = fx
    K_evs[0,2] = cx 
    K_evs[1,1] = fy
    K_evs[1,2] = cy
    K_evs[2, 2] = 1
    dist_coeffs_evs = np.asarray([k1, k2, p1, p2, k3])

    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(K_evs, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
    
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float32")
    term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
    points = cv2.undistortPointsIter(coords, K_evs, dist_coeffs_evs, np.eye(3), K_new_evs, criteria=term_criteria)
    rectify_map = points.reshape((H, W, 2))

    # 4) Create rectify map for events
    h5outfile = os.path.join(scenedir, f"rectify_map.h5")
    ef_out = h5py.File(h5outfile, 'w')
    ef_out.clear()
    ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
    ef_out["rectify_map"][:] = rectify_map
    ef_out.close()

    return rectify_map, K_new_evs

def ecd_evs_iterator(scenedir, DELTA_MS=None, H=180, W=240, timing=False, parallel=False, cors=8):
    if DELTA_MS is not None:
        assert DELTA_MS > 5 and DELTA_MS < 1000

    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    evs_file = glob.glob(osp.join(scenedir, "events.txt"))
    assert len(evs_file) == 1
    evs = np.asarray(np.loadtxt(evs_file[0], delimiter=" ")) # (N, 4) with [ts_sec, x, y, p]
    evs[:, 0] = evs[:, 0] * 1e6

    rect_file = osp.join(scenedir, "rectify_map.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    intrinsics = load_intrinsics_ecd(scenedir)
    fx, fy, cx, cy = intrinsics 
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    tss_imgs_us = sorted(np.loadtxt(osp.join(scenedir, "tss_us.txt")))

    if parallel:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        evs_split = split_evs_list_by_tss_split(evs, tss_imgs_us_split)

        processes = []
        return_dict = multiprocessing.Manager().dict()
        for i in range(cors):
            p = multiprocessing.Process(target=get_ecd_data, args=(tss_imgs_us_split[i].tolist(), evs_split[i], intrinsics, rectify_map, DELTA_MS, H, W, return_dict))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])
    else:
        data_list = get_ecd_data(tss_imgs_us, evs, intrinsics, rectify_map, DELTA_MS, H, W)

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} ECD-voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} ECD-voxels from {scenedir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us


def split_evs_list_by_tss_split(evs, tss_imgs_us_split):
    cors = len(tss_imgs_us_split)
    evs_splits = []
    for i in range(cors-1):
        mask_evs_batch = (evs[:, 0] >= tss_imgs_us_split[i][0]) & (evs[:, 0] < tss_imgs_us_split[i+1][0])
        evs_splits.append(evs[mask_evs_batch, :])
    mask_evs_batch = (evs[:, 0] >= tss_imgs_us_split[-1][0]) & (evs[:, 0] <= tss_imgs_us_split[-1][-1])
    evs_splits.append(evs[mask_evs_batch, :])
    return evs_splits

def rpg_evs_iterator(scenedir, side="left", stride=1, dT_ms=None, H=180, W=240, timing=False, parallel=False, cors=8):
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    intrinsics = load_intrinsics_rpg(os.path.join(scenedir, f"calib_undist_{side}.txt"))
    fx, fy, cx, cy = intrinsics 
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    evs_file = glob.glob(osp.join(scenedir, f"evs_{side}.txt"))
    assert len(evs_file) == 1
    evs = np.asarray(np.loadtxt(evs_file[0], delimiter=" ")) # (N, 4) with [ts_usecs, x, y, p]

    rect_file = osp.join(scenedir, f"rectify_map_{side}.h5")
    rectify_map = None if "simulation_3planes" in scenedir else read_rmap(rect_file, H=H, W=W)

    tss_file = os.path.join(scenedir, f"tss_imgs_us_{side}.txt")
    tss_imgs_us = sorted(np.loadtxt(tss_file))
    if dT_ms is None:
        dT_ms = np.diff(tss_imgs_us).mean()/1e3   
    assert dT_ms > 3 and dT_ms < 1000
    tss_imgs_us = tss_imgs_us[::stride]
 
    if parallel:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        evs_splits = split_evs_list_by_tss_split(evs, tss_imgs_us_split)

        processes = []
        return_dict = multiprocessing.Manager().dict()
        for i in range(cors):
            p = multiprocessing.Process(target=get_ecd_data, args=(tss_imgs_us_split[i].tolist(), evs_splits[i], intrinsics, rectify_map, dT_ms, H, W, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])
    else:
        data_list = get_ecd_data(tss_imgs_us, evs, intrinsics, rectify_map, dT_ms, H, W)

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} RPG-voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} RPG-voxels, imstart={0}, imstop={-1}, stride={stride}, dT_ms={dT_ms} on {scenedir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us

def mvsec_evs_iterator(scenedir, side="left", stride=1, dT_ms=None, timing=False, H=260, W=346):
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    intrinsics = np.loadtxt(os.path.join(scenedir, f"calib_undist_{side}.txt"))
    fx, fy, cx, cy = intrinsics
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    assert len(h5in) == 1
    datain = h5py.File(h5in[0], 'r')

    num_imgs = datain["davis"][side]["image_raw"].shape[0]
    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))
    assert num_imgs == len(tss_imgs_us)

    rect_file = osp.join(scenedir, f"rectify_map_{side}.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    event_idxs = datain["davis"][side]["image_raw_event_inds"]
    all_evs = datain["davis"][side]["events"][:]
    evidx_left = 0
    data_list = []
    for img_i in range(num_imgs):        
        evid_nextimg = event_idxs[img_i]
        evs_batch = all_evs[evidx_left:evid_nextimg][:]
        evidx_left = evid_nextimg
        rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]

        voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 2], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        # visualize_voxel(voxel)
        data_list.append((voxel, intrinsics, tss_imgs_us[img_i]))


    datain.close()

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} MVSEC-voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} MVSEC-voxels, imstart={0}, imstop={-1}, stride={1}, dT_ms={dT_ms} on {scenedir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us

def mvsec_evs_loader(scenedir, side="left", stride=1, H=260, W=346):
    intrinsics = np.loadtxt(os.path.join(scenedir, f"calib_undist_{side}.txt"))
    fx, fy, cx, cy = intrinsics
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    h5in = glob.glob(os.path.join(scenedir, f"*_data.hdf5"))
    assert len(h5in) == 1
    datain = h5py.File(h5in[0], 'r')

    num_imgs = datain["davis"][side]["image_raw"].shape[0]
    tss_imgs_us = sorted(np.loadtxt(os.path.join(scenedir, f"tss_imgs_us_{side}.txt")))
    assert num_imgs == len(tss_imgs_us)

    rect_file = osp.join(scenedir, f"rectify_map_{side}.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    event_idxs = datain["davis"][side]["image_raw_event_inds"]
    all_evs = datain["davis"][side]["events"][:]
    evidx_left = 0
    data_list = []
    for img_i in range(num_imgs):       
        evid_nextimg = event_idxs[img_i]
        evs_batch = all_evs[evidx_left:evid_nextimg][:]
        evidx_left = evid_nextimg
        rect = rectify_map[evs_batch[:, 1].astype(np.int32), evs_batch[:, 0].astype(np.int32)]

        voxel = to_voxel_grid(rect[..., 0], rect[..., 1], evs_batch[:, 2], evs_batch[:, 3], H=H, W=W, nb_of_time_bins=5)
        data_list.append((voxel, intrinsics, tss_imgs_us[img_i]))

    datain.close()

    print(f"Preloaded {len(data_list)} MVSEC-voxels, imstart={0}, imstop={-1}, stride={1} on {scenedir}")

    return data_list

def get_imstart_imstop_vector(indir):
    imstart, imstop = 0, -1

    if "corner_slow" in indir:
        imstart = 30
        imstop = 1180
    elif "robot_normal" in indir:
        imstart = 40
    elif "robot_fast" in indir:
        imstart = 30
        imstop = 901
    elif "desk_normal" in indir:
        imstart = 65
    elif "desk_fast" in indir:
        imstart = 25
        imstop = 1380
    elif "sofa_normal" in indir:
        imstart = 120
        imstop = 2700
    elif "sofa_fast" in indir:
        imstart = 50
        imstop = 1200
    elif "mountain_normal" in indir:
        imstart = 40
    elif "mountain_fast" in indir:
        imstart = 15
        imstop =  1290
    elif "hdr_normal" in indir:
        imstart = 30
    elif "hdr_fast" in indir:
        imstart = 35
    elif "corridors_dolly" in indir:
        imstart = 115
    # elif "corridors_walk" in indir:
    #     imstart = 40
    elif "school_dolly" in indir:
        imstart = 80
        imstop = 3160
    elif "school_scooter" in indir:
        imstart = 20
        imstop = 1290
    elif "units_dolly" in indir:
        imstart = 20
        imstop = 5750
    elif "units_scooter" in indir:
        imstart = 10
        imstop = 2790
 
    return imstart, imstop

def vector_evs_iterator(indir, side="left", stride=1, dT_ms=None, timing=False, H=480, W=640, cors=4):
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    intrinsics = np.loadtxt(os.path.join(indir, f"calib_undist_evs_{side}.txt"))
    assert len(intrinsics) == 4
    rectify_map = read_rmap(os.path.join(indir, f"rectify_map_{side}.h5"), H=H, W=W)

    seq = indir.split("/")[-1]
    fnameh5 = os.path.join(indir, f"{seq}1.synced.{side}_event.hdf5")
    datain = h5py.File(fnameh5, 'r') # (events, ms_to_idx, t_offset)
    evs_slicer = EventSlicer(datain)

    trafos = None
    tss_imgs_us = np.loadtxt(os.path.join(indir, f"tss_imgs_us_{side}.txt"))

    if dT_ms is None:
        dT_ms = np.mean(np.diff(tss_imgs_us)) / 1e3 # 33.3
    if "fast" in indir:
        dT_ms = dT_ms / 2.0 # 16.67
    else:
        dT_ms = dT_ms * 2.0 # 66.67
    assert dT_ms > 0 and dT_ms < 1000

    imstart, imstop = get_imstart_imstop_vector(indir)
    tss_imgs_us = tss_imgs_us[imstart:imstop:stride]
    data_list = get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig=H, Worig=W)

    datain.close()

    # # [DEBUG]
    # # perform center crop to avoid strong lens flare
    # dx = 90
    # dy = 15
    # for i, (voxel, intr, ts_us) in enumerate(data_list):
    #     # import matplotlib.pyplot as plt
    #     # plt.switch_backend('TkAgg')
    #     # # visualize_voxel(voxel)
    #     # v = torch.clone(voxel)
    #     # v[:, :dy, :] =  0 * v[:, :dy,  :]
    #     # v[:, -dy:, :] =  0 * v[:, -dy:, :]
    #     # v[:, :, :dx] =  0 * v[:, :, :dx]
    #     # v[:, :, -dx:] =  0 * v[:, :, -dx:]
    #     # v2 = torch.clone(voxel)
    #     # v2 = voxel[:, dy:-dy, dx:-dx]
    #     # v3 = F.interpolate(v2.unsqueeze(0), size=(480, 640), mode='nearest').squeeze(0)
    #     # v4 = F.interpolate(v2.unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False).squeeze(0)
    #     # v5 = F.interpolate(v2.unsqueeze(0), size=(480, 640), mode='bicubic', align_corners=True).squeeze(0)
    #     # visualize_voxel(voxel, v, v3, v4, v5)

    #     # [mode1]: cut and upsize
    #     voxel = voxel[:, dy:-dy, dx:-dx]
    #     intr[2] -= dx
    #     intr[3] -= dy
    #     voxel = F.interpolate(voxel.unsqueeze(0), size=(480, 640), mode='bicubic', align_corners=True).squeeze(0)
    #     scale_x = (640/(640-2*dx))
    #     scale_y = (480/(480-2*dy))
    #     intr[0] *= scale_x
    #     intr[1] *= scale_y
    #     intr[2] *= scale_x
    #     intr[3] *= scale_y

    #     # [mode2]: multiply by zeros
    #     # voxel[:, :dy, :] =  0 * voxel[:, :dy,  :]
    #     # voxel[:, -dy:, :] =  0 * voxel[:, -dy:, :]
    #     # voxel[:, :, :dx] =  0 * voxel[:, :, :dx]
    #     # voxel[:, :, -dx:] =  0 * voxel[:, :, -dx:]
    #     data_list[i] = (voxel, intr, ts_us)
    # # end [DEBUG]

    if timing:  
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} Vector-dvoxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} Vector-voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {indir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us

def compute_rmap_vector(Kevsdist, dist_coeffs_evs, scenedir, side, H=480, W=640):
    K_new_evs, roi = cv2.getOptimalNewCameraMatrix(Kevsdist, dist_coeffs_evs, (W, H), alpha=0, newImgSize=(W, H))
    
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H))).reshape((2, -1)).astype("float32")
    term_criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100, 0.001)
    points = cv2.undistortPointsIter(coords, Kevsdist, dist_coeffs_evs, np.eye(3), K_new_evs, criteria=term_criteria)
    rectify_map = points.reshape((H, W, 2))

    # # 4) Create rectify map for events
    h5outfile = os.path.join(scenedir, f"rectify_map_{side}.h5")
    ef_out = h5py.File(h5outfile, 'w')
    ef_out.clear()
    ef_out.create_dataset('rectify_map', shape=(H, W, 2), dtype="<f4")
    ef_out["rectify_map"][:] = rectify_map
    ef_out.close()

    return rectify_map, K_new_evs


def get_imstart_imstop_hku(indir):
    imstart, imstop = 0, -1

    if "hdr_tran_rota" in indir:
        imstart = 135
        imstop = 3230
    elif "HDR_slow" in indir:
        imstart = 240
        imstop = 4150
    elif "HDR_circle" in indir:
        imstart = 155
        imstop = 2115   
    elif "hdr_agg" in indir:
        imstart = 145
        imstop = 3600   
    elif "dark_normal" in indir:
        imstart = 150
        imstop = 2805   
    elif "aggressive_walk" in indir:
        imstart = 150
        imstop = 2385
    elif "aggressive_translation" in indir:
        imstart = 165
        imstop = 1795
    elif "aggressive_translation" in indir:
        imstart = 165
        imstop = 1795   
    elif "aggressive_small_flip" in indir:
        imstart = 150
        imstop = 1585
    elif "aggressive_rotation" in indir:
        imstart = 157
        imstop = 1660   

    return imstart, imstop

def hku_evs_iterator(indir, side="left", stride=1, timing=False, dT_ms=None, H=260, W=346):
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    intrinsics = np.loadtxt(os.path.join(indir, f"calib_undist_{side}.txt"))
    assert len(intrinsics) == 4
    rectify_map = read_rmap(os.path.join(indir, f"rectify_map_{side}.h5"), H=H, W=W)

    fnameh5 = os.path.join(indir, f"evs_{side}.h5")
    datain = h5py.File(fnameh5, 'r') # (events, ms_to_idx)
    evs_slicer = EventSlicer(datain)

    tss_imgs_us = np.loadtxt(os.path.join(indir, f"tss_imgs_us_{side}.txt"))

    trafos = []
    hotpixfilter = False
    if hotpixfilter:
        trafos.append(RemoveHotPixelsVoxel(num_stds=10))
    if dT_ms is None:
        dT_ms = np.mean(np.diff(tss_imgs_us)) / 1e3

    imstart, imstop = 0, -1  
    # [DEBUG]
    imstart, imstop = get_imstart_imstop_hku(indir)
    del_idxs = None
    if "HDR_circle" in indir:
        del_idxs = [1349, 1350, 1351, 1352, 1353, 1354]
    elif "HDR_slow" in indir:
        del_idxs = [3238, 3239, 3240, 3241, 3242]
    else:
        tss_imgs_us = tss_imgs_us[imstart:imstop:stride]
    
    if del_idxs is not None:
        del_idxs.extend(np.arange(0, imstart).tolist())
        del_idxs.extend(np.arange(imstop, len(tss_imgs_us)).tolist())
        tss_imgs_us = np.delete(tss_imgs_us, del_idxs)
        tss_imgs_us = tss_imgs_us[::stride]
    # end [DEBUG]

    data_list = get_real_data_list(evs_slicer, tss_imgs_us, intrinsics, rectify_map, trafos, dT_ms, Horig=H, Worig=W)

    datain.close()

    if timing:  
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} HKU voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} HKU voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {indir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us


# def hku_evs_loader(indir, side, stride=1, timing=False, dT_ms=None, H=260, W=346):
#     if timing:
#         t0 = torch.cuda.Event(enable_timing=True)
#         t1 = torch.cuda.Event(enable_timing=True)
#         t0.record()

#     intrinsics = np.loadtxt(os.path.join(indir, f"calib_undist_{side}.txt"))
#     assert len(intrinsics) == 4
#     rectify_map = read_rmap(os.path.join(indir, f"rectify_map_{side}.h5"), H=H, W=W)

#     fnameh5 = os.path.join(indir, f"evs_{side}.h5")
#     datain = h5py.File(fnameh5, 'r') # (events, ms_to_idx)
#     evs_slicer = EventSlicer(datain)

#     tss_imgs_us = np.loadtxt(os.path.join(indir, f"tss_imgs_us_{side}.txt"))

#     trafos = []
#     hotpixfilter = False
#     if hotpixfilter:
#         trafos.append(RemoveHotPixelsVoxel(num_stds=10))
#     if dT_ms is None:
#         dT_ms = np.mean(np.diff(tss_imgs_us)) / 1e3

#     imstart, imstop = get_imstart_imstop_hku(indir)
#     data_list = get_real_data_list(evs_slicer, tss_imgs_us[imstart:imstop:stride], intrinsics, rectify_map, trafos, dT_ms, Horig=H, Worig=W)

#     datain.close()

#     if timing:  
#         t1.record()
#         torch.cuda.synchronize()
#         dt = t0.elapsed_time(t1)/1e3
#         print(f"Preloaded {len(data_list)} HKU voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
#     print(f"Preloaded {len(data_list)} HKU voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {indir}")

#     return data_list
    
def fpv_evs_iterator(scenedir, stride=1, timing=False, dT_ms=None, H=260, W=346, parallel=False, cors=4, tss_gt_us=None):
    if timing:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

    evs_file = glob.glob(osp.join(scenedir, "events.txt"))
    assert len(evs_file) == 1
    evs = np.asarray(np.loadtxt(evs_file[0], delimiter=" ")) # (N, 4) with [ts_sec, x, y, p]
    evs[:, 0] = evs[:, 0] * 1e6

    t_offset_us = np.loadtxt(os.path.join(scenedir, "t_offset_us.txt")).astype(np.int64)
    evs[:, 0] -= t_offset_us

    rect_file = osp.join(scenedir, "rectify_map.h5")
    rectify_map = read_rmap(rect_file, H=H, W=W)

    intrinsics = load_intrinsics_ecd(scenedir)
    fx, fy, cx, cy = intrinsics 
    intrinsics = torch.from_numpy(np.array([fx, fy, cx, cy]))

    tss_imgs_us = sorted(np.loadtxt(osp.join(scenedir, "images_timestamps_us.txt")))
    imstart = 0
    imstop = -1
    if tss_gt_us is not None: # fix for FPV
        dT_imgs = tss_imgs_us[-1]-tss_imgs_us[0]
        dT_gt = tss_gt_us[-1]-tss_gt_us[0]
        if (dT_imgs - dT_gt) > 5*1e6 and (tss_gt_us[0] - tss_imgs_us[0]) > 5e6:
            imstart = np.where(tss_imgs_us > tss_gt_us[0])[0][0]
            imstop = np.where(tss_imgs_us < tss_gt_us[-1])[0][-1]
            print(f"Start reading Voxel from {imstart}, {imstop}, due to much shorter GT")

    if dT_ms is None:
        dT_ms = np.mean(np.diff(tss_imgs_us)) / 1e3
    assert dT_ms > 3 and dT_ms < 200

    tss_imgs_us = tss_imgs_us[imstart:imstop:stride]

    if parallel:
        tss_imgs_us_split = np.array_split(tss_imgs_us, cors)
        evs_split = split_evs_list_by_tss_split(evs, tss_imgs_us_split)

        processes = []
        return_dict = multiprocessing.Manager().dict()      
        for i in range(cors):
            p = multiprocessing.Process(target=get_ecd_data, args=(tss_imgs_us_split[i].tolist(), evs_split[i], intrinsics, rectify_map, dT_ms, H, W, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

        keys = np.array(return_dict.keys())
        order = np.argsort(keys)
        data_list = []
        for k in keys[order]:
            data_list.extend(return_dict[k])
    else:
        data_list = get_ecd_data(tss_imgs_us, evs, intrinsics, rectify_map, dT_ms, H, W)

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"Preloaded {len(data_list)} FPV-UZH voxels in {dt} secs, e.g. {len(data_list)/dt} FPS")
    print(f"Preloaded {len(data_list)} FPV-UZH voxels, imstart={imstart}, imstop={imstop}, stride={stride}, dT_ms={dT_ms} on {scenedir}")

    for (voxel, intrinsics, ts_us) in data_list:
        yield voxel.cuda(), intrinsics.cuda(), ts_us


def get_calib_fpv(indir):
    # equidistant model = fisheye, MVSEC has the same
    if "indoor_45_" in indir:
        K = np.array([173.07989681517137, 173.0734479068749, 163.31033691005516, 134.99889292308214])
        D = np.array([-0.03252275347038443, 0.0010042799356776398, -0.0048537750326187136, 0.0014604134198771906])
        T_cam_imu = np.array(([[0.9999641031275889, 0.003197881415389814, -0.007846401129833277, 0.001265030971654739], 
                            [-0.003216308945288942, 0.9999920967707336, -0.002337039332281246, -0.0025652081547025674], 
                            [0.007838865543278494, 0.0023621918900207225, 0.9999664855566258, -0.022231533861925983], 
                            [0.0, 0.0, 0.0, 1.0]]))        
    elif "indoor_forward_" in indir:
        K = np.array([172.98992850734132, 172.98303181090185, 163.33639726024606, 134.99537889030861])
        D = np.array([-0.027576733308582076, -0.006593578674675004, 0.0008566938165177085, -0.00030899587045247486])
        T_cam_imu = np.array(([[0.9999711474430529, 0.0013817010649267755, -0.007469617365767657, 0.00018050225881571712],
                            [-0.0014085305353606873, 0.9999925720306121, -0.00358774655345255, -0.004316353415695194],
                            [0.007464604688444933, 0.0035981642219379494, 0.9999656658561218, -0.027547385763471585],
                            [0.0, 0.0, 0.0, 1.0]]))
    elif "outdoor_forward_" in indir:
        K = np.array([174.23979032083346, 174.11105443010973, 163.91078563399876, 140.9726709818771])
        D = np.array([-0.03560363132286886, 0.001974723646350411, -0.0045671620060236855, 0.0011707729112974909])
        T_cam_imu = np.array([[ 0.9998829655327196, 0.005335413966337045, -0.014338360969823338, -0.0015224098391112568],
                              [-0.005432624310654592, 0.9999624656424586, -0.006749362884958196, -0.006621897399791399],
                              [ 0.014301812143655866, 0.00682646790524808, 0.9998744208676132, -0.023154837302635834],
                              [ 0.0, 0.0, 0.0, 1.0]])
    else:
        raise NotImplementedError(f"Unknown sequence {indir}")
    
    Kout = np.eye(3)
    Kout[0, 0] = K[0]
    Kout[1, 1] = K[1]
    Kout[0, 2] = K[2]
    Kout[1, 2] = K[3]
    return Kout, D, T_cam_imu