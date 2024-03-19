import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from typing import Any, List, Tuple, Optional, Dict
import copy
from torchvision.transforms import InterpolationMode
import math
import torchvision.transforms.functional as Fv
from devo.lietorch import SE3
from scipy.spatial.transform import Rotation as R


from utils.viz_utils import visualize_voxel, visualize_depth_map, visualize_pose

class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2/3.14),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomInvert(p=0.1),
            transforms.ToTensor()])

        self.max_scale = 0.5

    def spatial_transform(self, images, depths, poses, intrinsics):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        scale = 1
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(0.0, max_scale)

        intrinsics = scale * intrinsics

        ht1 = int(scale * ht)
        wd1 = int(scale * wd)

        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), recompute_scale_factor=False)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, poses, depths, intrinsics

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, depths, intrinsics):
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        return self.spatial_transform(images, depths, poses, intrinsics)


######################## Voxel-related Fcts #############################
def voxel_color_jitter(voxels, EPS = 1e-4):
    # jitter by small value (way smaller than avg. (-2.1, 2.1) value), e.g. 1e-4

    # [DEBUG]
    # typical values of abs(voxels[voxels>0]).mean() = (2.1) and <0: (-2.1)
    # self.posmean.append(voxels[voxels>0].mean().item())
    # self.negmean.append(voxels[voxels<0].mean().item())

    N, bins, H, W = voxels.shape  # (N, 5, 480, 640)
    voxels = voxels + (torch.rand_like(voxels)-0.5)*2 * EPS
    return voxels


class EVSDAugmentor:
    """ perform augmentation on EVS-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.max_scale = 0.25

        # [DEBUG]
        # self.posmean = [] 
        # self.negmean = []


    def voxel_spatial_transform(self, voxels, poses, depths, intrinsics, fix_scale=None):
        """ cropping and resizing """
        # voxels: (N, bins, H, W)
        ht, wd = voxels.shape[2:]
        
        # [DEBUG]
        # import matplotlib
        # matplotlib.use('Qt5Agg')
        # visualize_voxel(voxels[0])

        # [SCALING]
        # TODO: toggle for voxels
        max_scale = self.max_scale
        if fix_scale is None:
            scale = 1
            min_scale = np.log2(np.maximum(
                (self.crop_size[0] + 1) / float(ht),
                (self.crop_size[1] + 1) / float(wd)))
            if np.random.rand() < 0.8:
                scale = 2 ** np.random.uniform(min_scale, max_scale)
        else:
            scale = fix_scale
            min_scale = np.log2(fix_scale)
            if min_scale < max_scale:
                scale = 2 ** np.random.uniform(min_scale, max_scale)

        intrinsics = scale * intrinsics
        depths = depths.unsqueeze(dim=1)

        voxels = F.interpolate(voxels, scale_factor=scale, mode='bilinear',
            align_corners=False, recompute_scale_factor=True)
        depths = F.interpolate(depths, scale_factor=scale, recompute_scale_factor=True)#

        # [DEBUG]
        # visualize_depth_map(depths[0])
        # visualize_voxel(voxels[0])
        # [end SCALING]


        # [CROPPING]
        # always perform center crop (TODO: try non-center crops)
        y0 = (voxels.shape[2] - self.crop_size[0]) // 2
        x0 = (voxels.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        voxels = voxels[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        # [DEBUG]
        # visualize_voxel(voxels[-1])
        # visualize_depth_map(depths[-1])
        # [end CROPPING]

        depths = depths.squeeze(dim=1)

        return voxels, poses, depths, intrinsics


    def voxel_color_transform(self, voxels):
        """ voxel value trafos """
        voxels = voxel_color_jitter(voxels)
        return voxels

    def __call__(self, voxels, poses, depths, intrinsics):
        # images (n_frames,c,h,w)
        # poses (n_frames,7) (no SE3)
        # depths (n_frames,h,w)
        # intrinsics (n_frames,4)
        
        fix_scale = None
        voxels = self.voxel_color_transform(voxels)
        return self.voxel_spatial_transform(voxels, poses, depths, intrinsics, fix_scale)