import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum
from torchvision.ops import batched_nms

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4Evs
from .blocks import GradientClip, GatedResidual, SoftAgg
from .selector import Scorer, SelectionMethod, PatchSelector

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

from utils.voxel_utils import std, rescale, voxel_augment
from utils.viz_utils import visualize_voxel, visualize_N_voxels, visualize_scorer_map

DIM = 384 # default 384

class Update(nn.Module):
    def __init__(self, p, dim=DIM):
        super(Update, self).__init__()
        self.dim = dim

        self.c1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

        self.c2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))
        
        self.norm = nn.LayerNorm(dim, eps=1e-3)

        self.agg_kk = SoftAgg(dim)
        self.agg_ij = SoftAgg(dim)

        self.gru = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
            nn.LayerNorm(dim, eps=1e-3),
            GatedResidual(dim),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(dim, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """
        net = net + inp + self.corr(corr)
        net = self.norm(net) # (b,edges,384)

        
        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)
        weights = self.w(net)

        return net, (self.d(net), weights, None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3, dim_inet=DIM, dim_fnet=128, dim=32, patch_selector=SelectionMethod.SCORER):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.dim_inet = dim_inet # dim of context extractor and hidden state (update operator)
        self.dim_fnet = dim_fnet # dim of matching extractor
        self.patch_selector = patch_selector.lower()
        self.fnet = BasicEncoder4Evs(output_dim=self.dim_fnet, dim=dim, norm_fn='instance') # matching-feature extractor
        self.inet = BasicEncoder4Evs(output_dim=self.dim_inet, dim=dim, norm_fn='none') # context-feature extractor
        if self.patch_selector == SelectionMethod.SCORER:
            self.scorer = Scorer(5)

    def __event_gradient(self, images):
        images = images.sum(dim=2) # sum over bins
        dx = images[...,:-1,1:] - images[...,:-1,:-1]
        dy = images[...,1:,:-1] - images[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, return_color=False, scorer_eval_mode="multi", scorer_eval_use_grid=True):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0 # (1, 15, 128, 120, 160)
        imap = self.inet(images) / 4.0 # (1, 15, 384, 120, 160)

        b, n, c, h, w = fmap.shape # (1, 15, 128, 120, 160)
        P = self.patch_size

        # Patch selection
        if self.patch_selector == SelectionMethod.GRADIENT:
            # bias patch selection towards regions with high gradient
            g = self.__event_gradient(images) # gradient map (b,n_frames,h/4-1,w/4-1)
            
            if self.training:
                patch_selector_fn = PatchSelector("3xrandom")
            else:
                patch_selector_fn = PatchSelector(scorer_eval_mode, grid=scorer_eval_use_grid)
            
            x, y = patch_selector_fn(g, patches_per_image) # TODO use g[...,1:,1:]?
                        
            x = x.clamp(min=1, max=w-2)
            y = y.clamp(min=1, max=h-2)
        elif self.patch_selector == SelectionMethod.RANDOM:
            # random sampling
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")
        elif self.patch_selector == SelectionMethod.SCORER:
            scores = self.scorer(images) # (1, 15, 118, 158)
            scores = torch.sigmoid(scores)
            
            if self.training:
                x = torch.randint(0, w-2, size=[n, 3*patches_per_image], device="cuda")
                y = torch.randint(0, h-2, size=[n, 3*patches_per_image], device="cuda")

                coords = torch.stack([x, y], dim=-1).float() # (n_frames,3*patches_per_image,2)
                scores = altcorr.patchify(scores[0,:,None], coords, 0).view(n, 3 * patches_per_image) # extract patches of scorer map
                
                vx, ix = torch.sort(scores, dim=1) # sort by score (n_frames,3*patches_per_image)
                x = x + 1
                y = y + 1
                x = torch.gather(x, 1, ix[:, -patches_per_image:]) # choose patch idx with largest score
                y = torch.gather(y, 1, ix[:, -patches_per_image:])
                scores = vx[:, -patches_per_image:].contiguous().view(n,patches_per_image)
            else:            
                patch_selector_fn = PatchSelector(scorer_eval_mode, grid=scorer_eval_use_grid)
                x, y = patch_selector_fn(scores, patches_per_image)
                coords = torch.stack([x, y], dim=-1).float() # (b*n,patches_per_image,2)
                scores = altcorr.patchify(scores[0,:,None], coords, 0).view(n, patches_per_image) # extract weights of scorer map
                
                x += 1
                y += 1
        else:
            print(f"{self.patch_selector} not implemented")
            raise NotImplementedError
        
        coords = torch.stack([x, y], dim=-1).float() # in range (H//4, W//4)
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, self.dim_inet, 1, 1) # [B, n_images*n_patches_per_image, dim_inet, 1, 1]
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, self.dim_fnet, P, P) # [B, n_images*n_patches_per_image, dim_fnet, 3, 3]

        if return_color:
            clr = altcorr.patchify(images[0].abs().sum(dim=1,keepdim=True), 4*(coords + 0.5), 0).clamp(min=0,max=255).view(b, -1, 1)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device) # [B, n_images, 3, H//4, W//4]
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P) # [B, n_images*n_patches_per_image, 3, 3, 3]

        index = torch.arange(n, device="cuda").view(n, 1) # [n_images, 1]
        index = index.repeat(1, patches_per_image).reshape(-1) # [15, 80] => [15*80, 1] => [15*80]

        if self.training:
            if self.patch_selector == SelectionMethod.SCORER:
                return fmap, gmap, imap, patches, index, scores
        else:
            if return_color:
                return fmap, gmap, imap, patches, index, clr
            
        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class eVONet(nn.Module):
    def __init__(self, P=3, use_viewer=False, dim_inet=DIM, dim_fnet=128, dim=32, patch_selector=SelectionMethod.SCORER, norm="std2", randaug=False):
        super(eVONet, self).__init__()
        self.P = P
        self.dim_inet = dim_inet # dim of context extractor and hidden state (update operator)
        self.dim_fnet = dim_fnet # dim of matching extractor
        self.patch_selector = patch_selector
        self.patchify = Patchifier(patch_size=self.P, dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=dim, patch_selector=patch_selector)
        self.update = Update(self.P, self.dim_inet)
        
        self.dim = dim # dim of the first layer in extractor
        self.RES = 4.0
        self.norm = norm
        self.randaug = randaug


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, plot_patches=False, patches_per_image=80):
        """ Estimates SE3 between pair of voxel grids """
        
        # images (b,n_frames,c,h,w)
        # poses (b,n_frames)
        # disps (b,n_frames,h,w)
        
        b, n, v, h, w = images.shape

        # Normalize event voxel grids (rescale, std)
        if self.norm == 'none':
            pass
        elif self.norm == 'rescale' or self.norm == 'norm':
            # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise (by default)
            images = rescale(images)
        elif self.norm == 'standard' or self.norm == 'std':
            # Data standardization of events (voxel-wise)
            images = std(images, sequence=False)
        elif self.norm == 'standard2' or self.norm == 'std2':
            # Data standardization of events (sequence-wise by default)
            images = std(images)
        else:
            print(f"{self.norm} not implemented")
            raise NotImplementedError

        if self.training and self.randaug:
            if np.random.rand() < 0.33:
                if self.norm == 'rescale' or self.norm == 'norm':
                    images = voxel_augment(images, rescaled=True)
                elif 'std' in self.norm:
                    images = voxel_augment(images, rescaled=False)
                else:
                    print(f"{self.norm} not implemented")
                    raise NotImplementedError

        if plot_patches:
            plot_data = []

        intrinsics = intrinsics / self.RES
        if disps is not None:
            disps = disps[:, :, 1::4, 1::4].float()
            
        if self.patch_selector == SelectionMethod.SCORER:
            fmap, gmap, imap, patches, ix, scores = self.patchify(images, patches_per_image=patches_per_image, disps=disps)
        else:
            fmap, gmap, imap, patches, ix = self.patchify(images, patches_per_image=patches_per_image, disps=disps)
        # 1200 patches / 15 imgs = 80 patches per image
        # ix are image indices, i.e. simply (n_images, 80).flatten() = 15*80 = 1200 = n_patches
        # patches is (B, n_patches, 3, 3, 3), where (:, n_patches, 0, :, :) are x-coords, (:, n_patches, 1, :, :) are y-coords, (:, n_patches, 2, :, :) are depths 
        
        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        # first 8 images for initialization
        # kk are indixes for (first 8) patches/ixs of shape (1200*8/15)*8 = (640*8) = (5120)
        # jj are indices for (first 8) images/poses/intr in range (0, 7) of shape (640)*8 = (5120)
        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing="ij")
        ii = ix[kk] # here, ii are image indices for initialization (5120)

        imap = imap.view(b, -1, self.dim_inet) # (b,n_patches,dim_inet) = (b,1200,384)
        net = torch.zeros(b, len(kk), self.dim_inet, device="cuda", dtype=torch.float) # init hidden state
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), self.dim_inet, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous() # (B,edges,P,P,2) -> (B,edges,2,P,P)

            corr = corr_fn(kk, jj, coords1)
            # corr (b,edges,p*p*7*7*2)
            # delta, weights (b,edges,2)
            # net (b,edges,384)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta # (B, edges, 2)

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2) # k.sum() = (close_edges), i.e. > 0 and <= 2

            if self.patch_selector == SelectionMethod.SCORER:
                coords_full = pops.transform(Gs, patches, intrinsics, ii, jj, kk) # p_ij (B,close_edges,P,P,2)
                coords_gt_full, valid_full = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk, valid=True)
                coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k]) # p_ij (B,close_edges,P,P,2)
                coords_gt, valid = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)
                
                k = (dij > 0) & (dij <= 16) # default 16
                traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl, scores, valid_full[0,k], coords_full.detach()[0,k], coords_gt_full.detach()[0,k], weight.detach()[0,k], kk[k], dij[k]))
            else:
                coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k]) # p_ij (B,close_edges,P,P,2)
                coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], valid=True)
                
                traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))
            
            if plot_patches:
                coords_gt = pops.transform(Ps, patches_gt, intrinsics, ii, jj, kk)
                coords1_gt = coords_gt.permute(0, 1, 4, 2, 3).contiguous()
                coordsAll = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
                coordsAll = coordsAll.permute(0, 1, 4, 2, 3).contiguous() 
                plot_data.append((ii, jj, patches, coordsAll, coords1_gt))

        if plot_patches:
            traj.append(plot_data)        
        return traj