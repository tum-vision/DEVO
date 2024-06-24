
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

def invert_trafo(rot, trans):
    # invert transform from w2cam (esim, colmap) to cam2w
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3
    assert trans.shape[0] == 3

    rot_ = rot.transpose()
    trans_ = -1.0 * np.matmul(rot_, trans)

    check_rot(rot_)
    return rot_, trans_

def get_hom_trafos(rots_3_3, trans_3_1):
    N = rots_3_3.shape[0]
    assert rots_3_3.shape == (N, 3, 3)

    if trans_3_1.shape == (N, 3):
        trans_3_1 = np.expand_dims(trans_3_1, axis=-1)
    else:
        assert trans_3_1.shape == (N, 3, 1)
    
    pose_N_4_4 = np.zeros((N, 4, 4))
    hom = np.array([0,0,0,1]).reshape((1, 4)).repeat(N, axis=0).reshape((N, 1, 4))

    pose_N_4_4[:N, :3, :3] = rots_3_3  # (N, 3, 3)
    pose_N_4_4[:N, :3, 3:4] = trans_3_1 # (N, 3, 1)
    pose_N_4_4[:N, 3:4, :] = hom # (N, 1, 4)

    # pose_N_3_4 = np.asarray([np.concatenate((r, t), axis=1) for r, t in zip(rots_3_3, trans_3_1)])
    # pose_N_4_4 = np.asarray([np.vstack((p, np.asarray([0, 0, 0, 1]))) for p in pose_N_3_4])
    return pose_N_4_4


def check_rot(rot, right_handed=True, eps=1e-6):
    """
    Input: 3x3 rotation matrix
    """
    assert rot.shape[0] == 3
    assert rot.shape[1] == 3

    assert np.allclose(rot.transpose() @ rot, np.eye(3), atol=1e-6)
    assert np.linalg.det(rot) - 1 < eps * 2

    if right_handed:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) - 1.0) < 1e-3
    else:
        assert np.abs(np.dot(np.cross(rot[:, 0], rot[:, 1]), rot[:, 2]) + 1.0) < 1e-3

def quatList_to_poses_hom_and_tss(quat_list_us):
    """
    quat_list: [[t, px, py, pz, qx, qy, qz, qw], ...]
    """
    tss_all_poses_us = [t[0] for t in quat_list_us]

    all_rots = [R.from_quat(rot[4:]).as_matrix() for rot in quat_list_us]
    all_trans = [trans[1:4] for trans in quat_list_us]
    all_trafos = get_hom_trafos(np.asarray(all_rots), np.asarray(all_trans))

    return tss_all_poses_us, all_trafos

def poses_hom_to_quatlist(poses_hom, tss=None):
    """
    poses_hom: np.array (N, 4, 4)
    """    
    N = poses_hom.shape[0]
    assert poses_hom.shape == (N, 4, 4)
    if tss is not None:
        assert len(tss) == N

    quatlist = []
    for i, p in enumerate(poses_hom):
        px, py, pz = p[:3, 3]
        qx, qy, qz, qw = R.from_matrix(p[:3, :3]).as_quat()
        if tss is not None:
            quatlist.append([tss[i], px, py, pz, qx, qy, qz, qw])
        else: 
            quatlist.append([px, py, pz, qx, qy, qz, qw])

    return quatlist


def interpolate_traj_at_tss(traj_hf_us, tss_traj_us, tss_imgs_us, POSE_FREQ=150):
    fpss_traj = 1e6 / np.diff(tss_traj_us)
    # assert np.allclose(fpss_traj.mean(), POSE_FREQ, atol=5) # mocap is 120-150Hz
    print(f"POSE_FREQ: {POSE_FREQ}, fpss_traj: {fpss_traj.mean()}")

    assert np.all(sorted(tss_traj_us) == tss_traj_us)
    assert (np.diff(tss_imgs_us)<0).sum() < 10

    # copy first pose to first images
    before_first_pose = np.where(tss_imgs_us < tss_traj_us[0])[0]
    if len(before_first_pose) > 0:
        assert len(before_first_pose) <= 6, f"data problem check! {len(before_first_pose)}"
        for i in reversed(before_first_pose):
            tss_traj_us = np.insert(tss_traj_us, 0, tss_imgs_us[i])
            traj_hf_us = np.insert(traj_hf_us, 0, traj_hf_us[0], axis=0)

    # copy last pose to last images
    after_last_pose = np.where(tss_imgs_us > tss_traj_us[-1])[0]
    if len(after_last_pose) > 0:
        assert len(after_last_pose) <= 6, f"data problem check! {len(after_last_pose)}"
        for i in after_last_pose:
            tss_traj_us = np.append(tss_traj_us, tss_imgs_us[i])
            traj_hf_us = np.append(traj_hf_us, traj_hf_us[-1:, :], axis=0)

    rot_interpolator = Slerp(tss_traj_us, R.from_quat(traj_hf_us[:, 3:])) 
    trans_interpolator = interp1d(x=tss_traj_us, y=traj_hf_us[:, :3], axis=0, kind="cubic", bounds_error=True)

    rots_ref = rot_interpolator(tss_imgs_us).as_quat()
    trans_ref = trans_interpolator(tss_imgs_us)
    traj_ref = np.concatenate((trans_ref, rots_ref), axis=1)

    return traj_ref