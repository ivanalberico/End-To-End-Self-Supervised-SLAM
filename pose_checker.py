"""
Simple script to compute transformation between two specified poses
Used to validate the output transformations provided by gradslam dataset classes
"""

import os
import cv2
import yaml
import json
import torch # Sorry to the google supervisors
import shutil
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader

# GradSLAM Imports
import gradslam as gs
from gradslam.datasets import ICL, TUM
from gradslam import Pointclouds, RGBDImages

import numpy as np

# -------------SPECIFY YOUR DATASET PATH HERE!--------------
icl_path = 'data/ICL/'
# ----------------------------------------------------------

# load dataset
traj = "living_room_traj1_frei_png" # specifiy desired dataset trajectory
mylist = []
mylist.append(traj)
trajs_tuple = tuple(mylist) 

dataset = ICL(basedir=icl_path, trajectories=trajs_tuple, seqlen=3, height=240, width=320, dilation=100, start=700)
loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
colors, depths, intrinsics, poses, transforms, *_ = next(iter(loader))

print("RGB shape = {}".format(colors.shape))
print("Poses shape = {}".format(poses.shape))
print("Transforms shape = {}".format(transforms.shape))
print("Pose 0 =") 
print(poses[0,0])
print("Pose 1 =")
print(poses[0,1])
print("Pose 2 =")
print(poses[0,2])
print("Transform 0 =")
print(transforms[0,0])
print("Transform 1 =")
print(transforms[0,1])
print("Transform 2 =")
print(transforms[0,2])


# Translation and rotation for pose 1
t1 = np.array(poses[0,1,:3, 3])
R1 = np.array(poses[0,1,:3, :3])

# Translation and rotation for pose 2
t2 = np.array(poses[0,2,:3, 3])
R2 = np.array(poses[0,2,:3, :3])

# The transformation below are based on the good answer and explanation
# on how to compute relative transformations found in:
# https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes

# Compute transformation from pose 2 to pose 1
T_12 = np.c_[R1.T.dot(R2), R1.T.dot(t2-t1).reshape(3,1)]
T_12 = np.r_[T_12, np.array([0, 0, 0, 1]).reshape(1,4)]


# Compute transformation from pose 1 to pose 2
# T_21 = np.c_[R2.T.dot(R1), R2.T.dot(t1-t2).reshape(3,1)]
# T_21 = np.r_[T_21, np.array([0, 0, 0, 1]).reshape(1,4)]
# Compute inverse
# T_12= np.linalg.inv(T_21)

print("T12 = ")
print(T_12)
print("should match Transform 2 shown above")