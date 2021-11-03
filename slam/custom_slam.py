import torch
import numpy as np
import gradslam as gs
from gradslam.structures import Pointclouds

def image_recover_slam(noisy_rgbd, slam, device):
    """
    This loop is used for the image recover experiments through gradslam pipeline

    Inputs:
    noisy_rgbd: contains a corrupted input
    slam: the slam setup
    device(torch.device): cuda or gpu

    outputs:
    noisy_pointcloud: the reconstructed pointcloud of the scene containing the noisy data.
    """



    noisy_pointcloud = Pointclouds(device=device)
    batch_size, seq_len = noisy_rgbd.shape[:2]

    initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
    prev_frame = None
    for s in range(seq_len):
        live_frame = noisy_rgbd[:, s].to(device)
        live_frame = live_frame.detach() if s < seq_len - 1 else live_frame

        if s == 0 and live_frame.poses is None:
            live_frame.poses = initial_poses

        noisy_pointcloud, live_frame.poses = slam.step(noisy_pointcloud, live_frame)
        live_frame.poses = live_frame.poses.detach()

    return noisy_pointcloud