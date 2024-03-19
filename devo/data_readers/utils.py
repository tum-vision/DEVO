import os
import pickle
import numpy as np

def get_scene_path(dataroot, scene, modality="image_left"):
    seq_name = scene.split("/")[0]
    seq_diffic = scene.split("/")[2]
    seq_num = scene.split("/")[3]
    sp = os.path.join(dataroot, seq_name, seq_diffic, modality, seq_name, seq_name, seq_diffic, seq_num)
    assert os.path.exists(sp), sp
    return sp

def save_scene_info(scene_info, name):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    fgraph_dest_dir = os.path.join(cur_path, '../../fgraph')
    if not os.path.exists(fgraph_dest_dir):
        os.makedirs(fgraph_dest_dir)
    os.chmod(fgraph_dest_dir, 0o777)
    
    fname = os.path.join(fgraph_dest_dir, f'{name}.pickle')

    with open(fname, 'wb') as cachefile:
        pickle.dump((scene_info,), cachefile)
    os.chmod(fgraph_dest_dir, 0o555)

def is_converted(scene):
    return os.path.isfile(os.path.join(scene.replace('evs_left', 'image_left'), 'converted.txt'))

def scene_in_split(scene, train_split, verbose=True):
    if not any(x in scene for x in train_split):
        if verbose:
            print(f"Not adding {scene}, since scene is not in requested split")
        return False
    else:
        return True

def check_train_val_split(train, val, strict=True, name=None):
    assert len(train) > 0
    assert len(val) > 0
    if strict:
        assert len(set(train).intersection(set(val))) == 0
    else:
        intersect = set(train).intersection(set(val))
        for s in intersect:
            if name is None:
                print(f"\nWARNING: {s} is in both train and val split!!!\n")
            else:
                print(f"\nWARNING: {s} is in both train and {name}-val split!!!\n")

def load_splitfile(splitfile):
    with open(splitfile, 'r') as f:
        split = f.read().split()
    assert len(split) > 0
    return split


def seqs_in_scene_info(split, scene_info):
    splits_in_sinfo = True

    if split is not None:
        for seq in split:
            sum = np.sum([seq in sinf for sinf in scene_info.keys()])
            if sum == 0:
                print(f"Sequence {seq} not in scene_info")
                splits_in_sinfo = False
                break
            else:
                assert sum == 1

    return splits_in_sinfo