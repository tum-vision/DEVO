import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import functools
import operator
import h5py
import hdf5plugin

from ..lietorch import SE3
from .base import RGBDDataset, EVSDDataset
from .utils import is_converted, scene_in_split

class TartanAir(RGBDDataset):
    """ Derived class for TartanAir RGBD dataset """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAir, self).__init__(name='TartanAir', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir RGBD dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/image_left'))
        scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in scenes]
        scenes = functools.reduce(operator.concat, scenes)
        for scene in tqdm(sorted(scenes)):
            if not scene_in_split(scene, self.train_split):
                continue
            
            images = sorted(glob.glob(osp.join(scene, 'imgs/*.png')))
            assert len(images) > 0
            depths = sorted(glob.glob(osp.join(scene.replace("image_left", "depth_left"), 'depth_left/*.npy')))
            assert len(images) == len(depths)

            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            assert poses.shape[0] == len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            print(f"Added {scene} to TartanAir RGBD dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth


class TartanAirE2VID(RGBDDataset):
    """ Derived class for TartanAir e2v dataset """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAirE2VID, self).__init__(name='TartanAirE2VID', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAirE2VID dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/e2v'))
        scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in scenes]
        scenes = functools.reduce(operator.concat, scenes)
        for scene in tqdm(sorted(scenes)):
            if not scene_in_split(scene, self.train_split):
                continue

            images = sorted(glob.glob(osp.join(scene, 'e2calib/*.png')))
            assert len(images) > 0
            depthdir = scene.replace("/e2v/", "/depth_left/").replace("/datasets/tartan-e2v/", "/datasets/tartan/")
            depths = sorted(glob.glob(osp.join(depthdir, 'depth_left/*.npy')))[1:]
            assert len(images) == len(depths)

            scene_tartan = scene.replace("/e2v/", "/image_left/").replace("/datasets/tartan-e2v/", "/datasets/tartan/")
            poses = np.loadtxt(osp.join(scene_tartan, 'pose_left.txt'), delimiter=' ')
            poses = poses[1:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)
            assert poses.shape[0] == len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

            print(f"Added {scene} to TartanAir RGBD dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth


class TartanAirEVS(EVSDDataset):
    """ Derived class for TartanAir event + depth dataset (EVSD) """
    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAirEVS, self).__init__(name='TartanAirEVS', **kwargs)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir EVSD dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/evs_left'))
        scenes = [glob.glob(osp.join(s, '*/*/*/*')) for s in scenes]
        scenes = functools.reduce(operator.concat, scenes)
        for scene in tqdm(sorted(scenes)):
            if not is_converted(scene):
                print(f"Skipping {scene}. Not fully converted")
                continue

            if not scene_in_split(scene, self.train_split):
                continue

            voxels = sorted(glob.glob(osp.join(scene, 'h5/*.h5')))
            assert len(voxels) > 0
            depths = sorted(glob.glob(osp.join(scene.replace("evs_left", "depth_left"), 'depth_left/*.npy')))[1:] # No event voxel at first timestamp t=0
            assert len(voxels) == len(depths)

            # [simon] poses are c2w, did thorough viz and data_type.md]
            poses = np.loadtxt(osp.join(scene.replace('evs_left', 'image_left'), 'pose_left.txt'), delimiter=' ')[1:] # No event voxel at first timestamp t=0
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]] # NED (z,x,y) to (x,y,z) camera frame
            poses[:,:3] /= TartanAirEVS.DEPTH_SCALE
            intrinsics = [TartanAirEVS.calib_read()] * len(voxels)
            assert poses.shape[0] == len(voxels)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics) # graph is dict of {frameIdx: (co-visible frames, distance)}

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'voxels': voxels, 'depths': depths,
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
            
            print(f"Added {scene} to TartanAir EVDS dataset")

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def voxel_read(voxel_file):
        h5 = h5py.File(voxel_file, 'r')
        voxel = h5['voxel'][:]
        # assert voxel.dtype == np.float32 # (5, 480, 640)
        h5.close()
        return voxel

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAirEVS.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        # visualize_depth_map(depth)
        return depth