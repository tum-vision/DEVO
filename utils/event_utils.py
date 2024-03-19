import numpy as np


import math
from typing import Dict, Tuple

import h5py
import numpy as np
from numba import jit
import torch

# from https://github.com/uzh-rpg/DSEC/blob/main/scripts/utils/eventslicer.py
class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        if 'events/x' in self.h5f.keys():
            for dset_str in ['p', 'x', 'y', 't']:
                self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]
        else:
            for dset_str in ['p', 'x', 'y', 't']:
                self.events[dset_str] = self.h5f['{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms = np.maximum(t_start_ms, 0)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        #if t_end_ms_idx is None:
        #    t_end_ms_idx = self.ms2idx(t_end_ms-1)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]
    


def to_voxel_grid(xs, ys, ts, ps, H=480, W=640, nb_of_time_bins=5, remapping_maps=None):
    """Returns voxel grid representation of event steam. (5, H, W)

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


class RemoveHotPixelsVoxel:
    """Remove hot pixels."""

    def __init__(self, num_stds=10, num_hot_pixels=None):
        self.num_stds = num_stds
        self.num_hot_pixels = num_hot_pixels

    # Detect hot pixel function from Simon
    def detect_hot_pixels(self, voxel, num_hot_pixels=None, num_stds=None):
        voxel_flatten = voxel.flatten() # (2*H*W)

        if num_hot_pixels is not None:
            hot_pixel_inds = torch.atleast_1d(torch.argsort(voxel_flatten)[len(voxel_flatten)-int(num_hot_pixels):])
        else:
            mean, std = torch.mean(voxel), torch.std(voxel)
            threshold_filter = mean + num_stds * std
            hot_pixel_inds = torch.atleast_1d(torch.squeeze(np.argwhere(abs(voxel_flatten) > threshold_filter))) # (num_hotpixels)
            # print(f"Found {hot_pixel_inds.shape} hotpixels with num_stds={num_stds}")

        # assert len(torch.unique(hot_pixel_inds)) == len(hot_pixel_inds)
        return hot_pixel_inds

    def __call__(self, x):
        hot_pixel_inds = self.detect_hot_pixels(x, num_stds=self.num_stds, num_hot_pixels=self.num_hot_pixels)
        hot_pixel_index_2d = np.asarray(np.unravel_index(hot_pixel_inds, x.shape)).T
        x[hot_pixel_index_2d[:, 0], hot_pixel_index_2d[:, 1], hot_pixel_index_2d[:, 2]] = 0
        return x
    
def compute_ms_to_idx(tss_ns, ms_start=0):
    """
    evs_ns: (N, 4)
    idx_start: Integer
    ms_start: Integer
    """

    ms_to_ns = 1000000
    # tss_sorted, _ = torch.sort(tss_ns) 
    # assert torch.abs(tss_sorted != tss_ns).sum() < 500

    ms_end = int(math.floor(tss_ns.max()) / ms_to_ns)
    assert ms_end >= ms_start
    ms_window = np.arange(ms_start, ms_end + 1, 1).astype(np.uint64)
    ms_to_idx = np.searchsorted(tss_ns, ms_window * ms_to_ns, side="left", sorter=np.argsort(tss_ns))
    
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]] >= ms*ms_to_ns) for ms in ms_window]))
    assert np.all(np.asarray([(tss_ns[ms_to_idx[ms]-1] < ms*ms_to_ns) for ms in ms_window if ms_to_idx[ms] >= 1]))
    
    return ms_to_idx

def write_evs_arr_to_h5(evs, h5outfile, xidx=0, yidx=1, tssidx=2, polidx=3):
    ef_out = h5py.File(h5outfile, 'w')
    ef_out.clear()
    num_events = evs.shape[0]
    event_grp = ef_out.create_group('/events')
    event_grp.create_dataset('p', shape=(num_events,), dtype='|u1')
    event_grp.create_dataset('t', shape=(num_events,), dtype='<u4')
    event_grp.create_dataset('x', shape=(num_events,), dtype='<u2')
    event_grp.create_dataset('y', shape=(num_events,), dtype='<u2')
    event_grp["x"][:] = evs[:, xidx]
    event_grp["y"][:] = evs[:, yidx]
    event_grp["t"][:] = evs[:, tssidx]
    event_grp["p"][:] = evs[:, polidx]

    ms_to_idx = compute_ms_to_idx(evs[:, tssidx]*1e3)
    ef_out.create_dataset('ms_to_idx', shape=len(ms_to_idx), dtype="<u8")
    ef_out["ms_to_idx"][:] = ms_to_idx

    ef_out.close()