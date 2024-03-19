import pickle
import os
import os.path as osp

from .tartan import TartanAir, TartanAirEVS, TartanAirE2VID


def dataset_factory(dataset_list, **kwargs):
    """ create a combined dataset """

    from torch.utils.data import ConcatDataset

    dataset_map = { 
        'tartan': (TartanAir, ),
        'tartan_evs': (TartanAirEVS, ),
        'tartan_e2vid':  (TartanAirE2VID, ),
    }
    
    if not all(x in dataset_map for x in dataset_list):
        ValueError("dataset_list {dataset_list} should be a subset of {dataset_map}.")
    
    db_list = []
    for key in dataset_list:
        # cache datasets for faster future loading
        db = dataset_map[key][0](**kwargs)

        print("Dataset {} has {} images".format(key, len(db)))
        db_list.append(db)

    return ConcatDataset(db_list)
