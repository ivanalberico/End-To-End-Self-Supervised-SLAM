import os
import numpy as np
from path import Path

dataset_path = "C:\\Users\\akbar\\PycharmProjects\\SC-SfMLearner-Release-master\\rectified_nyu"
root = Path(dataset_path)
scene_list_path = root/'train.txt'
scenes = [root/folder[:-1] for folder in open(scene_list_path)]

pair_set_x = []
pair_set_y = []
for scene in scenes:

    imgs = sorted(scene.files('*.jpg'))
    intrinsics = sorted(scene.files('*.txt'))

    for i in range(0, len(imgs) - 1, 2):
        intrinsic = np.genfromtxt(intrinsics[int(i / 2)]).astype(np.float32).reshape((3, 3))
        store_fx = intrinsic[0][0]
        store_fy = intrinsic[1][1]
        pair_set_x.append(store_fx)
        pair_set_y.append(store_fy)

average_focal_x = sum(pair_set_x)/len(pair_set_x)
average_focal_y = sum(pair_set_y)/len(pair_set_y)

print("average_fx: ", average_focal_x, " average_fy: ", average_focal_y)
