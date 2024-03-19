import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.gumbel as gumbel
import numpy as np
import enum

from torchvision.ops import batched_nms
from . import altcorr


EPSILON = np.finfo(np.float32).tiny

class SelectionMethod(str, enum.Enum):
    RANDOM = "random"
    GRADIENT = "gradient"
    SCORER = "scorer"

class Scorer(nn.Module):
    def __init__(self, bins=5) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(bins, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3,),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3),
            nn.MaxPool2d(kernel_size=4, stride=4))
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        b, n, c1, h1, w1 = x.shape # voxels (batch,n_frames,bins,h,w)
        x = x.view(b*n, c1, h1, w1)
        scores = self.scorer(x)
        _, c2, h2, w2 = scores.shape
        return scores.view(b, n, h2, w2)
        

class PatchSelector():
    def __init__(self, method="multi", grid=False, **kwargs) -> None:
        self.method = method.lower()
        self.grid = grid
        self.GRID = 2
        self.KERNEL_SIZE = 4
        self.NMS_RADIUS = 1.5
        self.NMS_IOU = 0.4
        
    def _grid(self, scores):
        b, n, h1, w1 = scores.shape
        scores_grid = scores.view(b*n,1,h1,w1) # -> (b*n,1,h1,w1)
        # assert h1 % self.GRID == 0 and w1 % self.GRID == 0
        h2, w2 = (h1//self.GRID, w1//self.GRID)
        size = (h2,w2)
        scores_grid = F.unfold(scores_grid, kernel_size=size, stride=size) # (B*N,1*h2*w2,GRID*GRID)
        # scores_grid = F.softmax(scores_grid, dim=-2)
        scores_grid = scores_grid.view(b,n,h2,w2,self.GRID**2) # (b,n,h2,w2,GRID*GRID)
        return scores_grid
    
    def _grid2_coords_up(self, idx, scores_grid):
        # assert self.GRID == 2
        b, n, h2, w2, _ = scores_grid.shape
        x = (idx % w2) # (b*n,patches_per_image/(GRID*GRID),GRID*GRID)
        y = torch.div(idx, w2, rounding_mode='floor')
        x[...,1] = x[...,1] + w2
        y[...,2] = y[...,2] + h2
        x[...,3] = x[...,3] + w2
        y[...,3] = y[...,3] + h2
        x = x.contiguous().view(b*n,-1) # (b*n,patches_per_image)
        y = y.contiguous().view(b*n,-1) # (b*n,patches_per_image)
        return (x,y)
    
    def _grid2_idx_up(self, idx, scores_grid):
        # assert self.GRID == 2
        b, n, h2, w2, _ = scores_grid.shape
        # idx (b*n,patches_per_image/(GRID*GRID),GRID*GRID)
        (x,y) = self._grid2_coords_up(idx, scores_grid)
        idx = 2*w2*y + x
        idx = idx.view(b*n,-1) # (b*n,patches_per_image)
        return idx
    
    def _3xrandom(self, scores, patches_per_image):
        b, n, h, w = scores.shape
        x = torch.randint(0, w, size=[n, 3*patches_per_image], device="cuda")
        y = torch.randint(0, h, size=[n, 3*patches_per_image], device="cuda")

        coords = torch.stack([x, y], dim=-1).float() # (n_frames,3*patches_per_image,2)
        scores = altcorr.patchify(scores[0,:,None], coords, 0).view(n, 3 * patches_per_image) # extract patches of scorer map
        
        _, ix = torch.sort(scores, dim=1) # sort by score (n_frames,3*patches_per_image)
        x = x + 1
        y = y + 1
        x = torch.gather(x, 1, ix[:, -patches_per_image:]) # choose patch idx with largest score
        y = torch.gather(y, 1, ix[:, -patches_per_image:])
        return (x,y)
    
    def _multi(self, scores, patches_per_image):
        """ avg pooled multinomial sampling

        Args:
            scores (tensor): (b,n,h,w) sensor of scores (non neg) 
            patches_per_image (int): number of sampling patches per image

        Returns:
            (tensor,tensor): Tuple((b*n,patches_per_image),(b*n,patches_per_image)) x,y coords
        """
        b, n, h, w = scores.shape
        # 1) avg pooling
        avg_scores = F.avg_pool2d(scores, kernel_size=self.KERNEL_SIZE, stride=self.KERNEL_SIZE)
        _, _, h1, w1 = avg_scores.shape
        
        # 2) multinomial sampling (with grid)
        if self.grid:
            # assert patches_per_image % (self.GRID**2) == 0
            scores_grid = self._grid(avg_scores) # (b,n,h2,w2,GRID*GRID)
            _, _, h2, w2, _ = scores_grid.shape
            scores_t = scores_grid.view(b*n,h2*w2,(self.GRID**2)).transpose(-2,-1).contiguous().view(b*n*(self.GRID**2),h2*w2) # (b,n,h2*w2,GRID*GRID) -> (b,n,GRID*GRID,h2*w2) -> (b*n*GRID*GRID,h2*w2)
            scores_t += 1e-7 # to fulfill non-zero sum
            idx = torch.multinomial(scores_t, patches_per_image//(self.GRID**2)) # (b*n*GRID*GRID,patches_per_image/(GRID*GRID))
            idx = idx.view(b*n,self.GRID**2,-1).transpose(-2,-1) # -> (b*n,patches_per_image/(GRID*GRID),GRID*GRID)
            idx = self._grid2_idx_up(idx, scores_grid) # (b*n,patches_per_image)
        else:
            avg_scores = avg_scores.view(b*n,-1) # (b*n,h1*w1)
            idx = torch.multinomial(avg_scores, patches_per_image) # (b*n,patches_per_image)
        
        # 3) multinomial sampling for index
        idx_gather = idx[...,None].repeat(1,1,self.KERNEL_SIZE*self.KERNEL_SIZE) # -> (b*n,patches_per_image,KERNEL_SIZE*KERNEL_SIZE)
        avg_windows = F.unfold(scores.view(b*n,1,h,w), kernel_size=self.KERNEL_SIZE, stride=self.KERNEL_SIZE, padding=1).transpose(-2,-1) # -> (b*n,#windows,1*KERNEL_SIZE*KERNEL_SIZE)
        avg_windows_multi = torch.gather(avg_windows, -2, idx_gather) # (b*n,patches_per_image,KERNEL_SIZE*KERNEL_SIZE)
        avg_windows_multi += 1e-7 # to fulfill non-zero sum
        idx_offset = torch.multinomial(avg_windows_multi.flatten(0,1), 1).view(b*n,patches_per_image) # (b*n*patches_per_image,1) -> (b*n,patches_per_image)
        x_offset = (idx_offset % self.KERNEL_SIZE) # (b*n,patches_per_image)
        y_offset = torch.div(idx_offset, self.KERNEL_SIZE, rounding_mode='floor') # (b*n,patches_per_image)
        
        x = (idx % w1) # (b*n,patches_per_image)
        y = torch.div(idx, w1, rounding_mode='floor') # (b*n,patches_per_image)
        x = self.KERNEL_SIZE * x + x_offset
        y = self.KERNEL_SIZE * y + y_offset
        
        return (x,y)
    
    def _topk(self, scores, patches_per_image):
        """ pooled topk sampling
        
        Args:
            scores (tensor): (b,n,h,w) sensor of scores (non neg) 
            patches_per_image (int): number of sampling patches per image
        Returns:
            (tensor,tensor): Tuple((b*n,patches_per_image),(b*n,patches_per_image)) x,y coords
        """
        b, n, h, w = scores.shape
        # 1) max pooling (with indices)
        max_windows = F.unfold(scores.view(b*n,1,h,w), kernel_size=self.KERNEL_SIZE, stride=self.KERNEL_SIZE) # (b*n,1*KERNEL_SIZE*KERNEL_SIZE,#windows)
        max_scores, max_idx = torch.max(max_windows, dim=-2) # (b*n,#windows)
        max_scores = max_scores.view(b,n,h//self.KERNEL_SIZE,w//self.KERNEL_SIZE) # (b,n,h1,w1)
        max_idx = max_idx.view(b*n,-1) # (b*n,#windows) = (b*n,h1*w1)
        _, _, h1, w1 = max_scores.shape
        
        # 2) topk sampling (with grid)
        if self.grid:
            # assert patches_per_image % (self.GRID**2) == 0
            scores_grid = self._grid(max_scores) # (b,n,h2,w2,GRID*GRID)
            _, _, h2, w2, _ = scores_grid.shape
            # select patches_per_image max elem in scorer-map (scores)
            _, idx = torch.topk(scores_grid.view(b*n,h2*w2,-1), patches_per_image//(self.GRID**2), dim=-2) # (b*n,patches_per_image/(GRID*GRID),GRID*GRID)
            idx = self._grid2_idx_up(idx, scores_grid) # (b*n,patches_per_image)
        else:
            _, idx = torch.topk(max_scores.view(b*n,h1*w1), patches_per_image, dim=-1) # (b*n,patches_per_image)
        
        # 3) compute indices
        idx_offset = torch.gather(max_idx, 1, idx)
        offset_x = (idx_offset % self.KERNEL_SIZE)
        offset_y = torch.div(idx_offset, self.KERNEL_SIZE, rounding_mode='floor')
        
        # offset_x = (offset_idx % self.KERNEL_SIZE).clamp(min=1,max=KERNEL_SIZE-2)
        # offset_y = torch.div(idx_offset, self.KERNEL_SIZE, rounding_mode='floor').clamp(min=1,max=KERNEL_SIZE-2)
        x = (idx % w1) # (b*n,patches_per_image)
        y = torch.div(idx, w1, rounding_mode='floor') # (b*n,patches_per_image)
        x = self.KERNEL_SIZE * x + offset_x
        y = self.KERNEL_SIZE * y + offset_y
        
        return (x,y)
    
    def _nms(self, scores, patches_per_image):
        """ pooled nms sampling
        
        Args:
            scores (tensor): (b,n,h,w) sensor of scores (non neg) 
            patches_per_image (int): number of sampling patches per image
        Returns:
            (tensor,tensor): Tuple((b*n,patches_per_image),(b*n,patches_per_image)) x,y coords
        """
        b, n, h, w = scores.shape
        # 1) max pooling (with indices)
        max_scores, max_idx = F.max_pool2d(scores, kernel_size=self.KERNEL_SIZE, stride=self.KERNEL_SIZE, return_indices=True) # (b,n,h1,w1)
        max_idx = max_idx.view(b*n,-1) # (b*n,h1*w1)
        _, _, h1, w1 = max_scores.shape
        
        # 2) nms
        # boxes (patches_per_image*n_frames,4)
        boxes_center_x = (max_idx % w) # (b*n,h1*w1)
        boxes_center_y = torch.div(max_idx, w, rounding_mode='floor') # (b*n,h1*w1)
        boxes_x1 = (boxes_center_x.float() - self.NMS_RADIUS).clamp(min=0.0)
        boxes_y1 = (boxes_center_y.float() - self.NMS_RADIUS).clamp(min=0.0)
        
        boxes_x1y1 = torch.stack([boxes_x1, boxes_y1], dim=-1) # (b*n,h1*w1,2)
        boxes_x2y2 = boxes_x1y1 + torch.as_tensor([2*self.NMS_RADIUS,2*self.NMS_RADIUS], device="cuda")[None,None]
        boxes = torch.cat([boxes_x1y1, boxes_x2y2], dim=-1) # (b*n,h1*w1,2*2)
        boxes = boxes.view(b*n*h1*w1,4) # (b*n*h1*w1,4)
        if self.grid:
            h2 = h1 / 2
            w2 = w1 / 2
            boxes_x1_half = (boxes_x1.view(-1) < w2) # (b*n*h1*w1)
            boxes_y1_half = (boxes_y1.view(-1) < h2) # (b*n*h1*w1)
            quad1 = boxes_x1_half & boxes_y1_half
            quad2 = ~boxes_x1_half & boxes_y1_half
            quad3 = boxes_x1_half & ~boxes_y1_half
            quad4 = ~boxes_x1_half & ~boxes_y1_half
            idx_boxes = torch.arange(0, b*n*4, 4, device="cuda").view(-1,1).repeat(1,h1*w1).view(-1) # (b*n*h1*w1)
            idx_boxes[quad2] += 1
            idx_boxes[quad3] += 2 
            idx_boxes[quad4] += 3 
        else:
            idx_boxes = torch.arange(b*n, device="cuda").view(-1,1).repeat(1,h1*w1).view(-1) # (b*n*h1*w1)
        idx_keep = batched_nms(boxes, max_scores.view(-1), idx_boxes, self.NMS_IOU)

        if self.grid:
            idx_boxes = torch.div(idx_boxes, 4, rounding_mode='floor')
        scores_keep = max_scores.view(-1)[idx_keep]
        boxes_keep = boxes[idx_keep]
        boxes_center_x_keep = boxes_center_x.view(-1)[idx_keep]
        boxes_center_y_keep = boxes_center_y.view(-1)[idx_keep]
        
        # TODO how to get topk scores per frame without loop
        x = torch.empty(0,patches_per_image, device="cuda")
        y = torch.empty(0,patches_per_image, device="cuda")
        for f in range(b*n):
            mask = idx_boxes[idx_keep] == f
            # top_scores = torch.cat((top_scores, scores_keep[mask][:patches_per_image][None]))
            # TODO handle if not enough patches keeped
            x = torch.cat((x, boxes_center_x_keep[mask][:patches_per_image][None]))
            y = torch.cat((y, boxes_center_y_keep[mask][:patches_per_image][None]))
    
        return (x,y)
    
    def __call__(self, scores, patches_per_image):
        """ Call specific method

        Args:
            scores (tensor): (b,n,h,w) sensor of scores (non neg) 
            patches_per_image (int): number of sampling patches per image

        Returns:
            (tensor,tensor): Tuple((b*n,patches_per_image),(b*n,patches_per_image)) x,y coords
        """
        _, _, h, w = scores.shape # (b,n,h,w)
        # preprocessing
        factor = (self.GRID * self.KERNEL_SIZE) if self.grid else self.KERNEL_SIZE
        padding_h = (factor - (h % factor)) % factor
        padding_w = (factor - (w % factor)) % factor
        padding_h_top, padding_h_bottom = (padding_h // 2, padding_h // 2) if (padding_h % 2) == 0 else (padding_h // 2, padding_h // 2 + 1)
        padding_w_left, padding_w_right = (padding_w // 2, padding_w // 2) if (padding_w % 2) == 0 else (padding_w // 2, padding_w // 2 + 1)
        
        pad = (padding_w_left, padding_w_right, padding_h_top, padding_h_bottom)
        # scores_padded = F.pad(scores, pad, mode="constant", value=0)
        constant_pad = nn.ConstantPad2d(pad, value=0)
        scores_padded = constant_pad(scores)
        _, _, hp, wp = scores_padded.shape # (b,n,hp,wp)
        # assert (hp % factor) == 0 and (wp % factor) == 0
        
        (x,y) = getattr(self, f"_{self.method}")(scores_padded, patches_per_image)
        
        # postprocessing
        x = (x - padding_w_left).clamp(min=0, max=w-1)
        y = (y - padding_h_top).clamp(min=0, max=h-1)
        return (x,y)

