import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from io import StringIO
import sys
from PIL import Image
import io
import tqdm as tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R


def read_first_evs_from_rosbag(bag, evtopic):
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            t0_us = ev.ts.to_nsec()/1e3
            break
        break
    return t0_us


def read_evs_from_rosbag(bag, evtopic, H=180, W=240):
    print(f"Start reading evs from {evtopic}")

    evs = []
    progress_bar = tqdm.tqdm(total=bag.get_message_count(evtopic))
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            p = 1 if ev.polarity else 0
            evs.append([ev.x, ev.y, ev.ts.to_nsec()/1e3, p])
            # assert ev.x < W and ev.y < H # DEBUG
        progress_bar.update(1)

        # if len(evs) > 1000:
        #     break
    return np.array(evs) # (N, 4)


def read_H_W_from_bag(bag, imgtopic):
    for topic, msg, t in bag.read_messages(imgtopic):
        H, W = msg.height, msg.width
        print(f"Read H, W from bag: {H}, {W}")
        return H, W

def read_images_from_rosbag(bag, imgtopic, H=180, W=240):
    imgs = []
    
    progress_bar = tqdm.tqdm(total=bag.get_message_count(imgtopic))
    for topic, msg, t in bag.read_messages(imgtopic):
        img_str = str(msg)
        img_str = img_str[img_str.find("data")+6:]
        img_str = img_str[1:-1].split(',')
        pixel_values = [int(v) for v in img_str]
        image_array = np.array(pixel_values, dtype=np.uint8)
        image_array = image_array.reshape((msg.height, msg.width))
        imgs.append(image_array)
        progress_bar.update(1)

        if abs(H- msg.height) > 2 or abs(W-msg.width) > 2:
            print(f"WARNING: H, W mismatch: {msg.height}, {msg.width}, {H}, {W}")    

        # if len(imgs) > 50: # TODO: remove!
        #     break
    return imgs


def read_tss_us_from_rosbag(bag, imgtopic):
    tss_us = []
    for topic, msg, t in bag.read_messages(imgtopic):
        tss_us.append(msg.header.stamp.to_nsec() / 1e3)
    return tss_us


def read_poses_from_rosbag(bag, posestopic, T_marker_cam0, T_cam0_cam1):
    progress_bar = tqdm.tqdm(total=bag.get_message_count(posestopic))

    poses = []
    tss_us_gt = []
    for topic, msg, t in bag.read_messages(posestopic):
        if msg._type == "nav_msgs/Odometry":
            ps = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        else:
            ps = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        T_world_marker = np.eye(4)
        T_world_marker[:3, 3] = ps[:3]
        T_world_marker[:3, :3] = R.from_quat(ps[3:]).as_matrix()
        
        T_world_cam = T_world_marker @ T_marker_cam0
        T_world_cam = T_world_cam @ T_cam0_cam1

        T_world_cam = np.concatenate((T_world_cam[:3, 3], R.from_matrix(T_world_cam[:3, :3]).as_quat()))
        poses.append(T_world_cam)

        tss_us_gt.append(msg.header.stamp.to_nsec() / 1e3)
                     
        progress_bar.update(1)
    return np.array(poses), tss_us_gt

def read_calib_from_bag(bag, imtopic):
    for topic, msg, t in bag.read_messages(imtopic):
        K = msg.K
        break
    return K


def read_t0us_evs_from_rosbag(bag, evtopic):
    for topic, msg, t in bag.read_messages(evtopic):
        for ev in msg.events:
            t0_us = ev.ts.to_nsec()/1e3
            break
        break
    return t0_us