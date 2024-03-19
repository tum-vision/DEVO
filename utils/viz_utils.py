import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
import os
from datetime import datetime
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1].astype(np.int32), x[mask1].astype(np.int32)] = pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255] 
    return img


def prepare_vox_for_plot_numpy(voxi, EPS=1e-3):
    voxi = voxi.copy()
    voxi[np.bitwise_and(voxi<EPS, voxi>0)] = 0
    voxi[np.bitwise_and(voxi>-EPS, voxi<0)] = 0
    voxi[voxi<0] = -1
    voxi[voxi>0] = 1
    return voxi

def prepare_vox_for_plot(voxs, fi, EPS=1e-3):
    voxi = torch.clone(voxs[0, fi, ...]).detach().cpu()

    voxi[torch.bitwise_and(voxi<EPS, voxi>0)] = 0
    voxi[torch.bitwise_and(voxi>-EPS, voxi<0)] = 0
    voxi[voxi<0] = -1
    voxi[voxi>0] = 1
    return voxi

def select_rand_frame_pairs(N, num_frame_pairs=3):
    starts = np.sort(np.random.randint(0, N-1, size=(num_frame_pairs, 1)), axis=1) # in range (0, N-2)
    ends = np.sort(np.random.randint(starts+1, N, size=(num_frame_pairs, 1)), axis=1) # in range (starts, N-1)
    assert np.all(ends > starts)
    return np.concatenate((starts, ends), axis=1)

# TODO: offline script for this
def plot_patch_following(images, patch_data, evs=True, outdir=None, num_frame_pairs=5, patch_thickness = 5):
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)
    EPS = 1e-3

    N = images.shape[1]
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    if evs: # pick random (but constant) bin for voxel
        bin = np.random.randint(0, images.shape[2]) 

    rand_frame_pairs = select_rand_frame_pairs(N, num_frame_pairs=num_frame_pairs)
    for i in range(rand_frame_pairs.shape[0]):
        fi, fj = rand_frame_pairs[i, 0], rand_frame_pairs[i, 1]
        delta_f = fj - fi

        fig = plt.figure(figsize=(20, 20))
        ncols = 4
        nrows = (delta_f+2) // ncols + 1
        
        ax = fig.add_subplot(nrows, ncols, 1)
        if evs:
            vox = prepare_vox_for_plot(images, fi, EPS)
            ax.imshow(vox[bin], cmap=cmap, norm=norm)
        else:
            ax.imshow(images[0, fi].detach().cpu())

        # 1) Show whole image
        fhost = rand_frame_pairs[0, 0]
        hostpatches = patch_data[fhost][2].detach().cpu().numpy()
        xs_hosts = hostpatches[0, fhost*80:fhost*80+80, 0, :, :] * 4.
        ys_hosts = hostpatches[0, fhost*80:fhost*80+80, 1, :, :] * 4.
        # ds_hosts = hostpatches[0, :, 2, :, :] # TODO: depth
        ax.scatter(xs_hosts, ys_hosts, s=patch_thickness, c='green')
        # 1a) mark the random host-patch with black rectangle        
        rand_patch = np.random.randint(fhost*80, fhost*80+80)
        x_host = hostpatches[0, rand_patch, 0, :, :] * 4.
        y_host = hostpatches[0, rand_patch, 1, :, :] * 4.
        rect = patches.Rectangle((x_host.min()-1, y_host.min()-1), x_host.max()-x_host.min()+2, y_host.max()-y_host.min()+2, linewidth=3, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        # 2) Show random patch with DP**2
        ax = fig.add_subplot(nrows, ncols, 2)
        DP = 15
        # ax.scatter(x_host, y_host, s=patch_thickness, c='green')
        ax.scatter(x_host-x_host.min()+DP, y_host-y_host.min()+DP, s=patch_thickness, c='green')
        
        ymin, ymax = int(y_host[0, 0]-DP), int(y_host[2, 0]+DP)
        xmin, xmax = int(x_host[0, 0]-DP), int(x_host[0, 2]+DP)
        if evs:
            ax.imshow(vox[bin, ymin:ymax, xmin:xmax], cmap=cmap, norm=norm)
        else:
            ax.imshow(images[0, fi, ymin:ymax, xmin:xmax].detach().cpu())

        all_coords_gt = patch_data[fhost][4].detach().cpu().numpy()
        coords_gt = all_coords_gt[0, rand_patch, :, :, :]
        xs_gt = coords_gt[0, :, :] * 4.
        ys_gt = coords_gt[1, :, :] * 4.
        ax.scatter(xs_gt-xs_gt.min()+DP, ys_gt-ys_gt.min()+DP, s=patch_thickness, c='orange')

        # 3) Follow random patch
        for ploti, fidx in enumerate(range(fi+1, fj+1)):
            ax = fig.add_subplot(nrows, ncols, ploti+3)
            
            all_coords = patch_data[fidx][3].detach().cpu().numpy() # (B, edges, 2, P, P)
            coords = all_coords[0, rand_patch, :, :, :]  
            xs_proj = coords[0, :, :] * 4.
            ys_proj = coords[1, :, :] * 4.
            ax.scatter(xs_proj-xs_proj.min()+DP, ys_proj-ys_proj.min()+DP, s=patch_thickness, c='green')

            ymin, ymax = int(ys_proj[0, 0]-DP), int(ys_proj[2, 0]+DP)
            xmin, xmax = int(xs_proj[0, 0]-DP), int(xs_proj[0, 2]+DP)
            if evs:
                vox = prepare_vox_for_plot(images, fidx, EPS)
                ax.imshow(vox[bin, ymin:ymax, xmin:xmax], cmap=cmap, norm=norm)
            else:
                ax.imshow(images[0, fidx, ymin:ymax, xmin:xmax].detach().cpu())
            
            all_coords_gt = patch_data[fidx][4].detach().cpu().numpy()
            coords_gt = all_coords_gt[0, rand_patch, :, :, :]
            xs_gt = coords_gt[0, :, :] * 4.
            ys_gt = coords_gt[1, :, :] * 4.
            ax.scatter(xs_gt-xs_gt.min()+DP, ys_gt-ys_gt.min()+DP, s=patch_thickness, c='orange')
            
        # plt.show()
        plt.tight_layout()

        if outdir is not None:
            fig.savefig(outdir + f"patch_following_{fi}_{fj}.png")


# TODO: offline script for this
def plot_patch_following_all(images, patch_data, evs=True, outdir=None, num_frame_pairs=3, patch_thickness=4):
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)
    EPS = 1e-3
    N = images.shape[1]
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    if evs: # pick random (but constant) bin for voxel
        bin = np.random.randint(0, images.shape[2])

    rand_frame_pairs = select_rand_frame_pairs(N, num_frame_pairs=num_frame_pairs) # (N, num_fpairs)
    for i in range(rand_frame_pairs.shape[0]):
        fi, fj = rand_frame_pairs[i, 0], rand_frame_pairs[i, 1]
        delta_f = fj - fi

        fig = plt.figure(figsize=(20, 20))
        ncols = 4
        nrows = (delta_f+1) // ncols + 1
        
        ax = fig.add_subplot(nrows, ncols, 1)
        if evs:
            vox = prepare_vox_for_plot(images, fi, EPS)
            ax.imshow(vox[bin], cmap=cmap, norm=norm)
        else:
            ax.imshow(images[0, fi].detach().cpu())

        fhost = rand_frame_pairs[0, 0]
        hostpatches = patch_data[fhost][2].detach().cpu().numpy()
        xs_hosts = hostpatches[0, fhost*80:fhost*80+80, 0, :, :] * 4.
        ys_hosts = hostpatches[0, fhost*80:fhost*80+80, 1, :, :] * 4.
        ax.scatter(xs_hosts, ys_hosts, s=patch_thickness, c='green')
        
        for ploti, fidx in enumerate(range(fi+1, fj+1)):
            ii, jj = patch_data[fidx][0].detach().cpu().numpy(), patch_data[fidx][1].detach().cpu().numpy()
            ax = fig.add_subplot(nrows, ncols, ploti+2)
            
            if evs:
                vox = prepare_vox_for_plot(images, fidx, EPS)
                ax.imshow(vox[bin], cmap=cmap, norm=norm)
            else:
                ax.imshow(images[0, fidx].detach().cpu())

            all_coords = patch_data[fidx][3].detach().cpu().numpy() # (B, edges, 2, P, P)
            k = (ii == fi) & (jj == fidx) # (80)
            coords = all_coords[0, k, :, :, :]  # (80, 2, P, P)
            xs_proj = coords[:, 0, :, :] * 4.
            ys_proj = coords[:, 1, :, :] * 4.
            ax.scatter(xs_proj, ys_proj, s=patch_thickness, c='green')
            
            all_coords_gt = patch_data[fidx][4].detach().cpu().numpy()
            k = (ii == fi) & (jj == fidx)
            coords_gt = all_coords_gt[0, k, :, :, :]
            xs_gt = coords_gt[:, 0, :, :] * 4.
            ys_gt = coords_gt[:, 1, :, :] * 4.
            ax.scatter(xs_gt, ys_gt, s=patch_thickness, c='orange')
            
        # plt.show()
        plt.tight_layout()

        if outdir is not None:
            fig.savefig(outdir + f"patch_following_{fi}_{fj}.png")
    

@torch.no_grad()
def viz_flow_inference(outdir, flow_data, debug=False, patch_thickness=4, line_width=4):
    # flow_data: dict of {fidx: (ii, jj, coords_est)}
    plt.switch_backend('Agg') #   Agg  Qt5Agg
    outvizdir = os.path.join(outdir, "flow_viz")
    os.makedirs(outvizdir, exist_ok=True)
    
    frame_idxs = np.array(list(flow_data.keys()))
    P = flow_data[frame_idxs[0]]["coords_est"].shape[2]

    channels = flow_data[frame_idxs[0]]["img"].shape[-1]
    is_event_data = channels > 3
    if is_event_data: 
        colors = ['red', 'white', 'blue']
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.Normalize(vmin=-1, vmax=1)
        EPS = 1e-3
        bin = np.random.randint(0, channels)

    for counter, (fidx, flow_i) in enumerate(flow_data.items()):
        if fidx == 0:
            continue
        # if counter <= 8:
        #     continue

        ii, jj, kk = flow_i["ii"].detach().cpu().numpy(), flow_i["jj"].detach().cpu().numpy(), flow_i["kk"].detach().cpu().numpy()
        # assert flow_i["n"] == ii.max() + 1
        ii, jj = ii+fidx-ii.max(), jj+fidx-jj.max()
        coords_fidx = flow_i["coords_est"][0, ...].detach().cpu().numpy() * 4.0 # (768, 3, 3, 2)
        # [DEBUG]
        if debug:
            dij = abs(ii - jj)
            assert (dij==0).sum() == len(np.unique(kk)) 

        # center plot at index 4
        fig = plt.figure(figsize=(50, 30))
        nrows = 4
        ncols = 4

        PATCH_LIFETIME = 13
        N_plots = 11 # 15 a bit too much, 8 or 11 good
        fcenter_global = max(fidx - (N_plots) // 2 - 1, 0)
        fcenter_global = frame_idxs[np.argmin(abs(frame_idxs - fcenter_global))]

        xss, yss = [], []
        for di in range(-N_plots//2, N_plots//2+1):
            nth_idx = np.where((frame_idxs == fcenter_global))[0][0]
            fidx_before = frame_idxs[nth_idx+di]
            hosted_in_fcenter_proj_to_before = (ii == fcenter_global) & (jj == fidx_before)
            xs = coords_fidx[hosted_in_fcenter_proj_to_before, :, :, 0]
            ys = coords_fidx[hosted_in_fcenter_proj_to_before, :, :, 1]
            if len(xs) == 0:
                continue
            xss.append(xs) # appending (96, 3, 3)
            yss.append(ys)

        # plotting patches (starting from fidx, backward in time)
        for i in range(N_plots):
            ax = fig.add_subplot(nrows, ncols, N_plots-i)

            # plot the image
            nth_idx = np.where((frame_idxs == fidx))[0][0]
            fidx_before = frame_idxs[nth_idx-i-1]
            # print(f"(fidx_before, fcenter_global, fidx) = ({fidx_before}, {fcenter_global}, {fidx})")
            if fidx_before > fidx:
               continue
            
            if is_event_data:
                vox = prepare_vox_for_plot_numpy(flow_data[fidx_before]["img"], EPS=EPS)
                ax.imshow(vox[:, :, bin], cmap=cmap, norm=norm)
            else:
                ax.imshow(flow_data[fidx_before]["img"])
            
            if fidx_before == fcenter_global:
                ax.set_title(f"Center:{fidx_before}", fontsize=30, fontweight='bold')
                hosted_in_fcenter = (ii == fcenter_global) & (jj == fcenter_global)
                #if hosted_in_fcenter.sum() == 0:
                #    print("Error!")
                ax.scatter(coords_fidx[hosted_in_fcenter, :, : , 0], coords_fidx[hosted_in_fcenter, :, : , 1], s=patch_thickness, c="black")
                if len(xss) > 0:
                    ax.plot(np.stack(xss)[:, :, P//2, P//2], np.stack(yss)[:, :, P//2, P//2], linewidth=line_width)
                    ax.grid(False)
                    continue
            else:
                ax.set_title(f"t{fidx_before}", fontsize=30, fontweight='bold')

            hosted_in_fcenter_proj_to_before = (ii == fcenter_global) & (jj == fidx_before)
            xs = coords_fidx[hosted_in_fcenter_proj_to_before, :, :, 0]
            ys = coords_fidx[hosted_in_fcenter_proj_to_before, :, :, 1]
            xss.append(xs)
            yss.append(ys)
            ax.scatter(xs, ys, s=patch_thickness, c="black")
            ax.grid(False)

            # [DEBUG]
            if debug:
                assert len(np.unique(kk)) == 96*(flow_i["n"])
                assert len(ii) == 96*flow_i["n"]**2
                assert len(ii) == len(kk)
                assert len(ii) == len(jj)
                # assert len(np.unique(kk)) == 96*(flow_data[fidx_before]["n"]) # stride must be 1?
                assert (dij > PATCH_LIFETIME).sum().item() == 0
                for uniqi in np.unique(ii):
                    assert 96*flow_data[fidx]["n"] <= (ii==uniqi).sum()
                    if 96*flow_data[fidx]["n"] != (ii==uniqi).sum():
                        print(f"W")
            # [DEBUG]

        if is_event_data:
            bin = np.random.randint(0, channels)

        plt.tight_layout()
        if outvizdir is not None:
            fig.savefig(os.path.join(outvizdir, f"t{fidx:05d}.png"))
            plt.close()


@torch.no_grad()
def viz_flow_beforeFidx_inference(outdir, flow_data, debug=False, patch_thickness=4, line_width=4):
    # flow_data: dict of {fidx: (ii, jj, coords_est)}
    plt.switch_backend('Agg') #   Agg  Qt5Agg
    outvizdir = os.path.join(outdir, "flow_viz")
    os.makedirs(outvizdir, exist_ok=True)
    
    frame_idxs = np.array(list(flow_data.keys()))
    P = flow_data[frame_idxs[0]]["coords_est"].shape[2]

    channels = flow_data[frame_idxs[0]]["img"].shape[-1]
    is_event_data = channels > 3
    if is_event_data: 
        colors = ['red', 'white', 'blue']
        cmap = plt.cm.colors.ListedColormap(colors)
        norm = plt.Normalize(vmin=-1, vmax=1)
        EPS = 1e-3
        bin = np.random.randint(0, channels)

    for fidx, flow_i in flow_data.items():
        if fidx == 0:
            continue
        if flow_data[fidx]["n"] == 1:
            continue
        ii, jj, kk = flow_i["ii"].detach().cpu().numpy(), flow_i["jj"].detach().cpu().numpy(), flow_i["kk"].detach().cpu().numpy()
        coords_fidx = flow_i["coords_est"][0, ...].detach().cpu().numpy() * 4.0 # (768, 3, 3, 2)
        # [DEBUG]
        if debug:
            dij = abs(ii - jj)
            assert (dij==0).sum() == len(np.unique(kk)) 

        img = flow_i["img"]
        # center plot at index 4
        fig = plt.figure(figsize=(50, 30))

        nrows = 4
        ncols = 4
        PATCH_LIFETIME = 13
        N_plots = PATCH_LIFETIME + 1

        ax0 = fig.add_subplot(nrows, ncols, N_plots) 

        if is_event_data:
            vox = prepare_vox_for_plot_numpy(img, EPS=EPS)
            ax0.imshow(vox[:, :, bin], cmap=cmap, norm=norm)
        else:
            ax0.imshow(img)

        if debug:
            hosted_in_fidx = (ii == flow_data[fidx]["n"]-1) & (jj == flow_data[fidx]["n"]-1)
            ax0.scatter(coords_fidx[hosted_in_fidx, :, : , 0], coords_fidx[hosted_in_fidx, :, : , 1], s=patch_thickness, c="black")

        xss, yss = [], []
        for i in range(flow_data[fidx]["n"]-1):
            hosted_in_fidx_proj_to_before = (ii == flow_data[fidx]["n"]-1) & (jj == i)
            xs = coords_fidx[hosted_in_fidx_proj_to_before, :, :, 0]
            ys = coords_fidx[hosted_in_fidx_proj_to_before, :, :, 1]
            if len(xs) == 0:
                continue
            xss.append(xs) # appending (96, 3, 3)
            yss.append(ys)
        if len(xss) > 0:
            ax0.plot(np.stack(xss)[:, :, P//2, P//2], np.stack(yss)[:, :, P//2, P//2], linewidth=line_width)
        ax0.set_title(f"t{fidx}", fontsize=30, fontweight='bold')
        ax0.grid(False)
        
        # plotting patches
        plot_N_before_frames = min(flow_data[fidx]["n"], PATCH_LIFETIME) - 1
        for i in range(plot_N_before_frames): # TODO: fix oldest frame (sometimes no patches are plots)
            ax = fig.add_subplot(nrows, ncols, N_plots-i-1)

            nth_idx = np.where((frame_idxs == fidx))[0][0]
            fidx_before = frame_idxs[nth_idx-i-1]
            ax.set_title(f"t{fidx_before}", fontsize=30, fontweight='bold')
            if is_event_data:
                vox = prepare_vox_for_plot_numpy(flow_data[fidx_before]["img"], EPS=EPS)
                ax.imshow(vox[:, :, bin], cmap=cmap, norm=norm)
            else:
                ax.imshow(flow_data[fidx_before]["img"])

            hosted_in_fidx_proj_to_before = (ii == flow_data[fidx]["n"]-1) & (jj == flow_data[fidx]["n"]-2-i) 
            xs = coords_fidx[hosted_in_fidx_proj_to_before, :, :, 0]
            ys = coords_fidx[hosted_in_fidx_proj_to_before, :, :, 1]
            ax.scatter(xs, ys, s=patch_thickness, c="black")
            ax.grid(False)

            # [DEBUG]
            if debug:
                assert len(np.unique(kk)) == 96*(flow_data[fidx]["n"])
                assert len(ii) == 96*flow_data[fidx]["n"]**2
                assert len(ii) == len(kk)
                assert len(ii) == len(jj)
                # assert len(np.unique(kk)) == 96*(flow_data[fidx_before]["n"]) # stride must be 1?
                assert (dij > PATCH_LIFETIME).sum().item() == 0
                for uniqi in np.unique(ii):
                    assert 96*flow_data[fidx]["n"] <= (ii==uniqi).sum()
                    if 96*flow_data[fidx]["n"] != (ii==uniqi).sum():
                        print(f"W")
            # [DEBUG]

        if is_event_data:
            bin = np.random.randint(0, channels)

        plt.tight_layout()
        if outvizdir is not None:
            fig.savefig(os.path.join(outvizdir, f"t{fidx+1:05d}.png"))
            plt.close()



@torch.no_grad()
def plot_patches_hostii_targetjj(fid_host, fid_target, patch_data, coord_idx, ax=None, color='green', patch_thickness=4):
    # patch_data is: list of ((ii, jj, patches, coordsAll, coords1_gt))
    assert coord_idx == 3 or coord_idx == 4 # (3 = estimate, 4 = gt)
    ii, jj = patch_data[fid_host][0].detach().cpu().numpy(), patch_data[fid_host][1].detach().cpu().numpy()
    all_coords = patch_data[fid_host][coord_idx].detach().cpu().numpy() # (B, edges, 2, P, P)

    k = (ii == fid_host) & (jj == fid_target) # (80)
    coords = all_coords[0, k, :, :, :]  # (80, 2, P, P)
    xs_proj = coords[:, 0, :, :] * 4.
    ys_proj = coords[:, 1, :, :] * 4.
    
    if ax is None:
        P = coords.shape[-1]
        return xs_proj[:, P//2, P//2], ys_proj[:, P//2, P//2] # , np.repeat(color, k.sum()).tolist()
    else:
        ax.scatter(xs_proj, ys_proj, s=patch_thickness, c=color)

def make_bold_frame_subplot(ax):
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

def plot_flow_tartan_train(images, patch_data, evs=True, outdir=None, num_frame_pairs=3, patch_thickness=4):
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)
    EPS = 1e-3
    N = images.shape[1]
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    coord_idx, coord_idx_gt = 3, 4

    fids = np.linspace(0, N-1, N).astype(np.uint16)
    B = 2
    nrows = images.shape[2] if evs else 1
    ncols = 2*B+2
    for fidx_center in range(N):
        fig = plt.figure(figsize=(45, 30))
        fids_valid = np.array([(t <= fidx_center + B) and (t >= fidx_center - B) for t in fids])

        fids_noncenter_valid = np.array([(t <= fidx_center + B) and (t >= fidx_center - B) and (t != fidx_center) for t in fids])
        fidxs_noncenter_valid = np.where(fids_noncenter_valid)[0]

        delta_plotidx_noncenter = []
        for fid in fids:
            if fid < fidx_center and fid >= fidx_center - B:
                delta_plotidx_noncenter.append(fid - fidx_center)
            elif fid > fidx_center and fid <= fidx_center + B:
                delta_plotidx_noncenter.append(fid - fidx_center + 1)
            else:
                continue
        assert len(fidxs_noncenter_valid) == len(delta_plotidx_noncenter)

        for fid in fids:
            if fid >= fidx_center - B and fid <= fidx_center + B:
                if evs:
                    vox = prepare_vox_for_plot(images, fid, EPS)
                    for bin in range(images.shape[2]):
                        if fid == fidx_center:
                            #colors_valid = ['orange', 'yellow', 'green', 'purple']#colors_valid_gt = ['black']
                            xss, yss, xssgt, yssgt = [], [], [], []
                            for fidj, valid in enumerate(fids_valid):
                                if valid:
                                    xs, ys = plot_patches_hostii_targetjj(fidx_center, fidj, patch_data, coord_idx, patch_thickness=1) # color=colors_valid[fidj%len(colors_valid)]
                                    xsgt, ysgt = plot_patches_hostii_targetjj(fidx_center, fidj, patch_data, coord_idx_gt, patch_thickness=1)
                                    if len(xs) == 0 or len(xsgt) == 0:
                                        continue
                                    xss.append(xs)
                                    yss.append(ys)
                                    xssgt.append(xsgt)
                                    yssgt.append(ysgt)
                            xss = np.stack(xss) # (4, 80)
                            yss = np.stack(yss)

                            ax = fig.add_subplot(nrows, ncols, ncols*bin+B+1)
                            make_bold_frame_subplot(ax)
                            ax.imshow(vox[bin], cmap=cmap, norm=norm)
                            ax.plot(xss, yss)

                            ax = fig.add_subplot(nrows, ncols, ncols*bin+B+2)
                            make_bold_frame_subplot(ax)
                            ax.imshow(vox[bin], cmap=cmap, norm=norm)
                            ax.plot(xssgt, yssgt)
                        #else:
                        counter_noncenter = 0
                        for j, pidx in enumerate(delta_plotidx_noncenter):
                            #if valid:
                            ax = fig.add_subplot(nrows, ncols, ncols*bin+B+1+pidx)
                            ax.imshow(vox[bin], cmap=cmap, norm=norm)
                            plot_patches_hostii_targetjj(fidx_center, fidxs_noncenter_valid[j], patch_data, coord_idx, ax,  color='green', patch_thickness=patch_thickness)
                            plot_patches_hostii_targetjj(fidx_center, fidxs_noncenter_valid[j], patch_data, coord_idx_gt, ax, color='orange', patch_thickness=1)
                            counter_noncenter += 1
                else:
                    img = images[0, fid].detach().cpu().numpy().transpose((1, 2, 0))
                    img = img / (img.max() + 1e-6)
                    if fid == fidx_center:
                        xss, yss, xssgt, yssgt = [], [], [], []
                        for fidj, valid in enumerate(fids_valid):
                            if valid:
                                xs, ys = plot_patches_hostii_targetjj(fidx_center, fidj, patch_data, coord_idx, patch_thickness=1) # color=colors_valid[fidj%len(colors_valid)]
                                xsgt, ysgt = plot_patches_hostii_targetjj(fidx_center, fidj, patch_data, coord_idx_gt, patch_thickness=1)
                                if len(xs) == 0 or len(xsgt) == 0:
                                    continue
                                xss.append(xs)
                                yss.append(ys)
                                xssgt.append(xsgt)
                                yssgt.append(ysgt)
                        xss = np.stack(xss) # (4, 80)
                        yss = np.stack(yss)

                        ax = fig.add_subplot(nrows, ncols, B+1)
                        make_bold_frame_subplot(ax)
                        ax.imshow(img, cmap=cmap, norm=norm)
                        ax.plot(xss, yss)

                        ax = fig.add_subplot(nrows, ncols, B+2)
                        make_bold_frame_subplot(ax)
                        ax.imshow(img, cmap=cmap, norm=norm)
                        ax.plot(xssgt, yssgt)
                    #else:
                    counter_noncenter = 0
                    for j, pidx in enumerate(delta_plotidx_noncenter):
                        #if valid:
                        ax = fig.add_subplot(nrows, ncols, B+1+pidx)
                        ax.imshow(img, cmap=cmap, norm=norm)
                        plot_patches_hostii_targetjj(fidx_center, fidxs_noncenter_valid[j], patch_data, coord_idx, ax,  color='green', patch_thickness=patch_thickness)
                        plot_patches_hostii_targetjj(fidx_center, fidxs_noncenter_valid[j], patch_data, coord_idx_gt, ax, color='orange', patch_thickness=1)
                        counter_noncenter += 1
            
        # plt.show()
        plt.tight_layout()
        
        if outdir is not None:
            fig.savefig(outdir + f"patch_time_{fidx_center+1}.png")
            plt.close()



def plot_patch_depths_all(images, patch_data, disps, evs=True, outdir=None, num_frame_pairs=3, patch_thickness=4):
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)
    EPS = 1e-3
    N = images.shape[1]
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    if evs: # pick random (but constant) bin for voxel
        bin = np.random.randint(0, images.shape[2])

    rand_frame_pairs = select_rand_frame_pairs(N, num_frame_pairs=num_frame_pairs) # (N, num_fpairs)
    for i in range(rand_frame_pairs.shape[0]):
        fi, fj = rand_frame_pairs[i, 0], rand_frame_pairs[i, 1]
        delta_f = fj - fi

        fig = plt.figure(figsize=(20, 20))
        ncols = 4
        nrows = 2*(delta_f+1) // ncols + 1
        
        ax = fig.add_subplot(nrows, ncols, 1)
        if evs:
            vox = prepare_vox_for_plot(images, fi, EPS)
            ax.imshow(vox[bin], cmap=cmap, norm=norm)
        else:
            ax.imshow(images[0, fi].detach().cpu())

        fhost = rand_frame_pairs[0, 0]
        hostpatches = patch_data[fhost][2].detach().cpu().numpy()
        xs_hosts = hostpatches[0, fhost*80:fhost*80+80, 0, :, :] * 4.
        ys_hosts = hostpatches[0, fhost*80:fhost*80+80, 1, :, :] * 4.
        zs_hosts = hostpatches[0, fhost*80:fhost*80+80, 2, :, :] * 4.
        zcolors = np.digitize(zs_hosts, np.percentile(zs_hosts, [33, 66])) # zs_hosts.min(), zs_hosts.max(), zs_hosts.mean(), np.mean(zs_hosts)
        ax.scatter(xs_hosts, ys_hosts, s=patch_thickness, c=zcolors)

        ax = fig.add_subplot(nrows, ncols, 2)
        ax.imshow(disps[0, fhost].detach().cpu(), cmap='gray')
        ax.scatter(xs_hosts, ys_hosts, s=patch_thickness, c=zcolors)
        
        for ploti, fidx in enumerate(range(fi+1, fj+1)):
            ii, jj = patch_data[fidx][0].detach().cpu().numpy(), patch_data[fidx][1].detach().cpu().numpy()
            ax = fig.add_subplot(nrows, ncols, 2*ploti+3)
            
            if evs:
                vox = prepare_vox_for_plot(images, fidx, EPS)
                ax.imshow(vox[bin], cmap=cmap, norm=norm)
            else:
                ax.imshow(images[0, fidx].detach().cpu())

            all_coords = patch_data[fidx][3].detach().cpu().numpy() # (B, edges, 2, P, P)
            k = (ii == fi) & (jj == fidx) # (80)
            coords = all_coords[0, k, :, :, :]  # (80, 2, P, P)
            xs_proj = coords[:, 0, :, :] * 4.
            ys_proj = coords[:, 1, :, :] * 4.

            # plot sampled est-depth
            est_patches = patch_data[fidx][2].detach().cpu().numpy() # estimated depths
            est_disps = est_patches[0, fidx*80:fidx*80+80, 2:3, :, :]
            zcolors = np.digitize(est_disps, np.percentile(est_disps, [33, 66])) # est_disps.min(), est_disps.max(), est_disps.mean(), np.mean(est_disps)
            ax.scatter(xs_proj, ys_proj, s=patch_thickness, c=zcolors)

            ax = fig.add_subplot(nrows, ncols, 2*ploti+4)
            ax.imshow(disps[0, fidx].detach().cpu(), cmap='gray')
            ax.scatter(xs_proj, ys_proj, s=patch_thickness, c=zcolors)

        if outdir is not None:
            fig.savefig(outdir + f"depth_{fi}_{fj}.png")



def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

# visualize events
def visualize_sparse_voxel(voxel):
    bins = voxel.shape[0]

    plt.figure(figsize=(20, 20))
    for i in range(bins):
        plt.subplot(1, bins, i+1)
        plt.spy(abs(voxel[i]))
    plt.show()


def visualize_N_voxels(voxels_in, EPS=1e-3, idx_plot_vox=[0]):
    # cmaps
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)

    voxels = torch.clone(voxels_in)
    N = voxels.shape[0]
    assert N > sorted(idx_plot_vox)[-1]

    if len(idx_plot_vox) > 7:
       N = 7
       idx_plot_vox = idx_plot_vox[:N]
       print(f"Only plotting first {N} voxels")

    voxels = voxels[idx_plot_vox]
    bins = voxels.shape[1]

    voxels[torch.bitwise_and(voxels<EPS, voxels>0)] = 0
    voxels[torch.bitwise_and(voxels>-EPS, voxels<0)] = 0

    voxels[voxels<0] = -1
    voxels[voxels>0] = 1

    fig = plt.figure(figsize=(20, 20))
    for i in range(N*bins):
        ax = fig.add_subplot(N, bins, i+1)
        ax.imshow(voxels[i%N][i%bins], cmap=cmap, norm=norm)    
    plt.tight_layout() 
    plt.show()


def visualize_voxel(*voxel_in, EPS=1e-3, save=False, folder="results/voxels"):
    # cmaps
    colors = ['red', 'white', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    norm = plt.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(20, 20))
    for i, vox in enumerate(voxel_in):
        voxel = torch.clone(vox)
        bins = vox.shape[0]

        voxel[torch.bitwise_and(voxel<EPS, voxel>0)] = 0
        voxel[torch.bitwise_and(voxel>-EPS, voxel<0)] = 0

        voxel[voxel<0] = -1
        voxel[voxel>0] = 1
        for j in range(bins):
            plt.subplot(len(voxel_in), bins, i*bins + j + 1)
            plt.imshow(voxel[j], cmap=cmap, norm=norm)
            ax = plt.gca()
            ax.grid(False)
    if save:
        os.makedirs(folder, exist_ok=True)
        str = datetime.today().strftime('%Y-%m-%d_%H:%M:%S.%f')
        plt.axis('off')
        plt.savefig(f'{folder}/{str}.png', bbox_inches='tight', transparent=True, pad_inches=0)
    else:    
        plt.show()
    plt.close()


# visualize inverse depth map  cmap='plasma'
def visualize_depth_map(depth):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.grid(False)
    if depth.shape[0] == 1:
        plt.imshow(depth[0], cmap='plasma')
    else:
        plt.imshow(depth, cmap='plasma')
    plt.show()


def visualize_pose(*poses, plot_axes="xy"):
    """Visualize poses in 3D using evo"""
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(poses)))
    plot_mode = getattr(plot.PlotMode, plot_axes)
    plt.switch_backend('Qt5Agg')
    fig = plt.figure(figsize=(8, 8))
    ax = plot.prepare_axis(fig, plot_mode=plot_mode)

    for i, pose in enumerate(poses):
        tstamps = np.arange(len(pose))
        traj = PoseTrajectory3D(positions_xyz=pose[:,:3], orientations_quat_wxyz=pose[:,3:], timestamps=tstamps)
        plot.traj(ax, plot_mode, traj, '-', colors[i], f"traj_{i}")
    plt.show()

def visualize_scorer_map(scores, save=False, folder="results/scorer"):
    # scores (h,w)
    # plt.switch_backend('Qt5Agg')
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.grid(False)
    if scores.shape[0] == 1:
        plt.imshow(scores[0], cmap='gray', interpolation='nearest')
    else:
        plt.imshow(scores, cmap='gray', interpolation='nearest')
    # plt.colorbar()
    # plt.title("scorer map")
    if save:
        os.makedirs(folder, exist_ok=True)
        str = datetime.today().strftime('%Y-%m-%d_%H:%M:%S.%f')
        plt.axis('off')
        plt.savefig(f'{folder}/{str}.png', bbox_inches='tight', transparent=True, pad_inches=0)
    else:    
        plt.show()
    plt.close()