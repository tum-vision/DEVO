import numpy as np
import torch
import torchvision.transforms.functional as f


def std(voxs, sequence=True):
    # Data standardization of events (sequence-wise by default)
    # Do not preserve pos-neg inequality, but ensure pos-neg equality
    # see https://github.com/uzh-rpg/rpg_e2depth/blob/master/utils/event_tensor_utils.py#L52
    b, n, c, h, w = voxs.shape
    if sequence:
        flatten_voxs = voxs.view(b,-1)
    else:
        flatten_voxs = voxs.view(b,n,-1)
    nonzero_ev = (flatten_voxs != 0.0)
    num_nonzeros = nonzero_ev.sum(dim=-1)
    
    if torch.all(num_nonzeros > 0):
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array

        mean = torch.sum(flatten_voxs, dim=-1, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
        stddev = torch.sqrt(torch.sum(flatten_voxs ** 2, dim=-1, dtype=torch.float32) / num_nonzeros - mean ** 2)
        mask = nonzero_ev.type_as(flatten_voxs)
        flatten_voxs = mask * (flatten_voxs - mean[...,None]) / stddev[...,None]
    
    return flatten_voxs.view(b,n,c,h,w)


def rescale(voxs, sequence=True):
    # Normalize (rescaling) neg events into [-1,0) and pos events into (0,1] sequence-wise (by default)
    # Preserve pos-neg inequality
    b, n, c, h, w = voxs.shape
    if sequence:
        flatten_voxs = voxs.view(b,-1)
    else:
        flatten_voxs = voxs.view(b,n,-1)
    
    pos = flatten_voxs > 0.0
    neg = flatten_voxs < 0.0
    vx_max = torch.Tensor([1e-5], device="cuda") if pos.sum().item() == 0 else flatten_voxs[pos].max(dim=-1)[0]
    vx_min = torch.Tensor([1e-5], device="cuda") if neg.sum().item() == 0 else flatten_voxs[neg].min(dim=-1)[0]
    # [DEBUG]
    # print("vx_max", vx_max.item())
    # print("vx_min", vx_min.item())
    flatten_voxs[pos] = flatten_voxs[pos] / vx_max
    flatten_voxs[neg] = flatten_voxs[neg] / -vx_min
    # assert flatten_voxs.max() <= 1.0
    # assert flatten_voxs.min() >= -1.0

    return flatten_voxs.view(b,n,c,h,w)


def evs2rgb(voxs):
    pos_voxs = voxs.clone()
    neg_voxs = voxs.clone()
    pos_voxs[voxs < 0.0] = 0.0
    neg_voxs[voxs > 0.0] = 0.0
    # [DEBUG]
    assert pos_voxs.min().item() >= 0.0 and pos_voxs.max().item() <= 1.0
    assert neg_voxs.max().item() <= 0.0 and neg_voxs.min().item() >= -1.0
    neg_voxs *= -1.0
    green_channel = torch.zeros_like(pos_voxs)
    rgb_images = torch.stack((neg_voxs, green_channel, pos_voxs), dim=-3)
    return rgb_images


def rgb2evs(rgb_images):
    pos_voxs = rgb_images[...,2,:,:]
    neg_voxs = -rgb_images[...,0,:,:]
    voxs = pos_voxs + neg_voxs
    return voxs


def _augment(voxs, op=None, factor=None):
    evs_images = evs2rgb(voxs)
    
    # to Uint8
    evs_images = (255*evs_images).to(torch.uint8)
    evs_images_flatten = evs_images.flatten(0,2)

    if op is None:
        pass
    else:
        if factor is None:
            evs_images_flatten = op(evs_images_flatten)
        else:
            evs_images_flatten = op(evs_images_flatten, factor)
    
    # to float32
    evs_images = evs_images_flatten.view(*evs_images.shape)
    evs_images = evs_images.to(torch.float32)/255
    
    voxs = rgb2evs(evs_images)
    return voxs


def _aug_ops():
    ops = [f.adjust_brightness, f.adjust_contrast, f.invert, f.posterize, f.adjust_saturation, f.adjust_sharpness, f.solarize]
    return ops


def _aug_factors(num_bins):
    factors = [
        torch.linspace(0.1, 0.2, num_bins), # Brightness
        torch.linspace(0.05, 0.2, num_bins), # Contrast
        torch.tensor(0.0), # Invert
        8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), # Posterization
        torch.linspace(0.05, 0.2, num_bins), # Saturation
        torch.linspace(0.9, 2.0, num_bins), # Sharpness
        torch.linspace(0, 30, num_bins).round().int(), # Solarize
    ]
    return factors


def voxel_augment(voxs, rescaled=False, num_bins=10):
    # 1) rescale
    if not rescaled:
        voxs = rescale(voxs)

    # 2) augment
    ops = _aug_ops()
    factors = _aug_factors(num_bins)
    op_rand = torch.randint(len(ops), (1,)).item()
    factor_rand = torch.randint(num_bins, (1,)).item()
    op = ops[op_rand]
    op_factor = factors[op_rand]
    if op_factor.ndim > 0:
        factor = op_factor[factor_rand]
    else:
        factor = None
    
    voxs = _augment(voxs, op=op, factor=factor)

    # 3) std
    return std(voxs)