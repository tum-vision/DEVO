import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

# from .net import VONet # TODO add net.py
from .enet import eVONet
from .utils import *
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")

from utils.viz_utils import visualize_voxel


class DEVO:
    def __init__(self, cfg, network, evs=False, ht=480, wd=640, viz=False, viz_flow=False, dim_inet=384, dim_fnet=128, dim=32):
        self.cfg = cfg
        self.evs = evs

        self.dim_inet = dim_inet
        self.dim_fnet = dim_fnet
        self.dim = dim
        # TODO add patch_selector
        
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False # TODO timing in param

        self.viz_flow = viz_flow
        
        self.n = 0      # active keyframes/frames (every frames == keyframe)
        self.m = 0      # number active patches
        self.M = self.cfg.PATCHES_PER_FRAME     # (default: 96)
        self.N = self.cfg.BUFFER_SIZE           # max number of keyframes (default: 2048)

        self.ht = ht    # image height
        self.wd = wd    # image width

        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0 # how often this network is called __call__()

        self.flow_data = {}

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda") # 3 channels = (x, y, depth)
        self.patches_gt_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, self.dim_inet, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, self.dim_fnet, self.P, self.P, **kwargs)

        ht = int(ht // RES)
        wd = int(wd // RES)

        self.fmap1_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 1), int(wd // 1), **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, self.dim_fnet, int(ht // 4), int(wd // 4), **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, self.dim_inet, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            print(f"Loading from {network}")
            checkpoint = torch.load(network)
            # TODO infer dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim
            self.network = VONet(patch_selector=self.cfg.PATCH_SELECTOR) if not self.evs else \
                eVONet(dim_inet=self.dim_inet, dim_fnet=self.dim_fnet, dim=self.dim, patch_selector=self.cfg.PATCH_SELECTOR)
            if 'model_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['model_state_dict'])
            else:
                # legacy
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if "update.lmbda" not in k:
                        new_state_dict[k.replace('module.', '')] = v
                self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.dim_inet = self.network.dim_inet
        self.dim_fnet = self.network.dim_fnet
        self.dim = self.network.dim
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.poses_,
            self.points_,
            self.colors_,
            intrinsics_)

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)
    
    @property
    def patches_gt(self):
        return self.patches_gt_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.dim_inet)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, self.dim_fnet, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        print("keyframes", self.n)
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        if self.is_initialized:
            poses = [self.get_pose(t) for t in range(self.counter)]
            poses = lietorch.stack(poses, dim=0)
            poses = poses.inv().data.cpu().numpy()
        else:
            print(f"Warning: Model is not initialized. Using Identity.") # eval still runs bug
            id = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            poses = np.array([id for t in range(self.counter)])
            poses[:, :3] = poses[:, :3] + np.random.randn(self.counter, 3) * 0.01 # small random trans

        tstamps = np.array(self.tlist, dtype=np.float64)

        if self.viewer is not None:
            self.viewer.join()

        return poses, tstamps
    
    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]]) 
        # TODO: self.ix.shape = self.M*self.N
        # self.ix is filled dynamically

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.dim_inet, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):
        # described in 3.3. Keyframing DPVO paper
        # "after each update, compute flow_mag <t-5, t-3> and remove <t-4> if less than 64px"
        i = self.n - self.cfg.KEYFRAME_INDEX - 1 # t-5, KF_INDEX = 4 per default
        j = self.n - self.cfg.KEYFRAME_INDEX + 1 # t-3
        m = self.motionmag(i, j) + self.motionmag(j, i) 
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX # scalar
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP) # store relative pose between <t-5, t-4>

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.patches_gt_[i] = self.patches_gt_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1 # remove frame
            self.m -= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        coords = self.reproject()

        with autocast(enabled=True):
            
            corr = self.corr(coords)
            ctx = self.imap[:,self.kk % (self.M * self.mem)]
            with Timer("other", enabled=self.enable_timing):
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

        lmbda = torch.as_tensor([1e-4], device="cuda")
        weight = weight.float()
            
            # [DEBUG]
            # dij = (self.ii - self.jj).abs()
            # k = (dij > 0) & (dij <= 4)
            # print("BA weights mean", weight[0, k].mean().item())
            # print("BA weights std", weight[0, k].std().item())
            # print("BA weights max", weight[0, k].max().item())
            # print("BA weights min", weight[0, k].min().item())
            # [DEBUG]
        target = coords[...,self.P//2,self.P//2] + delta.float()

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")
            
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]

    def flow_viz_step(self):
        # [DEBUG]
        # dij = (self.ii - self.jj).abs()
        # assert (dij==0).sum().item() == len(torch.unique(self.kk)) 
        # [DEBUG]

        coords_est = pops.transform(SE3(self.poses), self.patches, self.intrinsics, self.ii, self.jj, self.kk) # p_ij (B,close_edges,P,P,2)
        self.flow_data[self.counter-1] = {"ii": self.ii, "jj": self.jj, "kk": self.kk,\
                                          "coords_est": coords_est, "img": self.image_, "n": self.n}

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.image_)
        # plt.show()
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME  # default: 13
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics, scale=1.0):
        """ track new frame """

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image)

        if self.viz_flow:
            self.image_ = image.detach().cpu().permute((1, 2, 0)).numpy()

        if not self.evs:
            image = 2 * (image[None,None] / 255.0) - 0.5 
        else:
            image = image[None,None]

            # [DEBUG]
            # import matplotlib
            # matplotlib.use('Qt5Agg')
            # visualize_voxel(image[0][0].detach().cpu(), EPS=1e-3)
            # i2 = image[image!=0]
            # print("stats before norm", i2.min().item(), i2.max().item(), i2.mean().item(), i2.std().item(), i2.median().item())
            
            if self.n == 0:
                nonzero_ev = (image != 0.0)
                zero_ev = ~nonzero_ev
                num_nonzeros = nonzero_ev.sum().item()
                num_zeros = zero_ev.sum().item()
                # [DEBUG]
                # print("nonzero-zero-ratio", num_nonzeros, num_zeros, num_nonzeros / (num_zeros + num_nonzeros))
                if num_nonzeros / (num_zeros + num_nonzeros) < 2e-2: # TODO eval hyperparam (add to config.py)
                    print(f"skip voxel at {tstamp} due to lack of events!")
                    return

            b, n, v, h, w = image.shape
            flatten_image = image.view(b,n,-1)
            
            if self.cfg.NORM.lower() == 'none':
                pass
            elif self.cfg.NORM.lower() == 'rescale' or self.cfg.NORM.lower() == 'norm':
                # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise
                # Preserve pos-neg inequality (quantity only)
                pos = flatten_image > 0.0
                neg = flatten_image < 0.0
                vx_max = torch.Tensor([1]).to("cuda") if pos.sum().item() == 0 else flatten_image[pos].max(dim=-1, keepdim=True)[0]
                vx_min = torch.Tensor([1]).to("cuda") if neg.sum().item() == 0 else flatten_image[neg].min(dim=-1, keepdim=True)[0]
                # [DEBUG]
                # print("vx_max", vx_max.item())
                # print("vx_min", vx_min.item())
                if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                    # no information for at least one polarity
                    print(f"empty voxel at {tstamp}!")
                    return
                flatten_image[pos] = flatten_image[pos] / vx_max
                flatten_image[neg] = flatten_image[neg] / -vx_min
            elif self.cfg.NORM.lower() == 'standard' or self.cfg.NORM.lower() == 'std':
                # Data standardization of events only
                # Does not preserve pos-neg inequality
                # see https://github.com/uzh-rpg/rpg_e2depth/blob/master/utils/event_tensor_utils.py#L52
                nonzero_ev = (flatten_image != 0.0)
                num_nonzeros = nonzero_ev.sum(dim=-1)
                if torch.all(num_nonzeros > 0):
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array

                    mean = torch.sum(flatten_image, dim=-1, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
                    stddev = torch.sqrt(torch.sum(flatten_image ** 2, dim=-1, dtype=torch.float32) / num_nonzeros - mean ** 2)
                    mask = nonzero_ev.type_as(flatten_image)
                    flatten_image = mask * (flatten_image - mean[...,None]) / stddev[...,None]
            else:
                print(f"{self.cfg.NORM} not implemented")
                raise NotImplementedError

            image = flatten_image.view(b,n,v,h,w)

            # [DEBUG]
            # import matplotlib
            # matplotlib.use('Qt5Agg')
            # visualize_voxel(image[0][0].detach().cpu(), EPS=1e-3)
            # i2 = image[image!=0]
            # print(f"stats after norm={self.cfg.NORM}", i2.min().item(), i2.max().item(), i2.mean().item(), i2.std().item(), i2.median().item())

        if image.shape[-1] == 346:
            image = image[..., 1:-1] # hack for MVSEC, FPV,...
    
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(image.detach().cpu().numpy()[0, 0, :1, ...].transpose(1, 2, 0))
        # plt.show()

        # TODO patches with depth is available (val)
        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, 
                    return_color=True,
                    scorer_eval_mode=self.cfg.SCORER_EVAL_MODE,
                    scorer_eval_use_grid=self.cfg.SCORER_EVAL_USE_GRID)

        self.patches_gt_[self.n] = patches.clone()

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES
        
        # color info for visualization
        if not self.evs:
            clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
            self.colors_[self.n] = clr.to(torch.uint8)
        else:
            clr = (clr[0,:,[0,0,0]] + 0.5) * (255.0 / 2)
            self.colors_[self.n] = clr.to(torch.uint8)
            

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1

        if self.n > 0 and not self.is_initialized:
            thres = 2.0 if scale == 1.0 else scale ** 2 # TODO adapt thres for lite version
            if self.motion_probe() < thres: # TODO: replace by 8 pixels flow criterion (as described in 3.3 Initialization)
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1 # add one (key)frame
        self.m += self.M # add patches per (key)frames to patch number

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()
        
        elif self.is_initialized:
            self.update()
            self.keyframe()

        if self.viz_flow:
            self.flow_viz_step()