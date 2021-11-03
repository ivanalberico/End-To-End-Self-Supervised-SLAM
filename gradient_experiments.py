import os
import cv2
import yaml
import json
import torch # Sorry to the google supervisors
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchviz import make_dot, make_dot_from_trace

# Imports from our files
from utils.arguments import arguments
from utils.modify_images import corrupt_rgbd
from slam.custom_slam import image_recover_slam
from loss.losses import knn_points_loss, color_points_loss
from utils.yaml_configs import load_yaml, save_yaml
from utils.training_utils import define_optim, define_schedular

# GradSLAM Imports
import gradslam as gs
from gradslam.datasets import ICL
from gradslam.slam import ICPSLAM
from gradslam.slam import PointFusion
from chamferdist import ChamferDistance
from gradslam import Pointclouds, RGBDImages



class Gradient_Flow:
    def __init__(self, arguments):
        self.args = arguments
        self.device = torch.device("cuda" if self.args.SETTINGS.device == "cuda" else "cpu")

        self.dataset_init()
        self.model_init()


    def dataset_init(self):
        """
        Initialize the dataset in this function

        Input:
        None
        Output:
        None
        """
        print("Loading Images of Size {} x {}".format(self.args.DATA.width,self.args.DATA.height))

        self.dataset = ICL(basedir=self.args.DATA.data_path,
                           seqlen=self.args.DATA.sequence_length,
                           height=self.args.DATA.height,
                           width=self.args.DATA.width)

        self.train_loader = DataLoader(dataset=self.dataset,
                                       batch_size=self.args.OPTIMIZATION.batch_size,
                                       shuffle=False,
                                       num_workers=self.args.SETTINGS.num_workers,
                                       pin_memory=True,
                                       drop_last=True)

        print("Data Loaded")

    def model_init(self):

        print("Initializing Models")

        if self.args.MODEL.slam == "ICPSLAM":
            self.slam = ICPSLAM(odom=self.args.MODEL.odom, device=self.device)
            self.gt_slam = ICPSLAM(odom="gt", device=self.device)

        elif self.args.MODEL.slam == "PointFusion":
            self.slam = PointFusion(odom=self.args.MODEL.odom, device=self.device)
            self.gt_slam = PointFusion(odom="gt", device=self.device)

        print("Using the {} based model for SLAM".format(self.args.MODEL.slam))


    def train(self):
        self.recover_image()
        return

    def recover_image(self):
        print("Recovering Image based on gradslam")


        colors, depths, intrinsics, poses, transform_seq, _ = next(iter(self.train_loader))
        colors /= 255

        colors, depths, intrinsics, poses = colors.to(self.device), \
                                            depths.to(self.device), \
                                            intrinsics.to(self.device), \
                                            poses.to(self.device)

        rgbd = RGBDImages(colors, depths, intrinsics, poses)

        # plot rgbd frames, check RGBDImages to see the plot function
        #rgbd.plotly(0).show()

        gt_reconstruction, _ = self.gt_slam(rgbd)
        gt_reconstruction = gt_reconstruction.detach()
        gt_pointcloud = gt_reconstruction.points_list[0].unsqueeze(0).contiguous()
        gt_color_pointcloud = gt_reconstruction.colors_list[0].unsqueeze(0).contiguous()
        #gt_reconstruction.plotly(0).show()


        # Modify Colors and Depths
        noisy_depths = depths.clone()
        noisy_colors = colors.clone()
        noisy_colors, noisy_depths = corrupt_rgbd(args = self.args,
                                                  device = self.device,
                                                  noisy_colors=noisy_colors,
                                                  noisy_depths=noisy_depths)
        noisy_rgbd = RGBDImages(noisy_colors, noisy_depths, intrinsics, poses)
        noisy_rgbd.plotly(0).show()

        self.slam(noisy_rgbd)[0].plotly(0).show()

        parameters = []
        if self.args.DEPTH_RECOVER.optimize_depth:
            parameters.append(noisy_rgbd.depth_image)
        if self.args.DEPTH_RECOVER.optimize_color:
            parameters.append(noisy_rgbd.rgb_image)

        self.optimizer = define_optim(args=self.args,
                                      parameters=parameters)

        self.schedular = define_schedular(args=self.args,
                                          optimizer=self.optimizer)

        for i in tqdm(range(self.args.OPTIMIZATION.epochs)):
            loss = 0
            noisy_reconstruction = image_recover_slam(noisy_rgbd=noisy_rgbd,
                                                      slam=self.slam,
                                                      device=self.device)

            noisy_pointcloud = noisy_reconstruction.points_list[0].unsqueeze(0).contiguous()
            noisy_color_pointcloud = noisy_reconstruction.colors_list[0].unsqueeze(0).contiguous()



            knn_loss, indexes =  knn_points_loss(gt_pointcloud=gt_pointcloud,
                                                 noisy_pointcloud=noisy_pointcloud)

            if self.args.DEPTH_RECOVER.optimize_depth:
                loss += knn_loss
                print("knn_loss: ", round(knn_loss.item(),6))

            if self.args.DEPTH_RECOVER.optimize_color:
                color_loss = color_points_loss(gt_pointcloud_color=gt_color_pointcloud,
                                               noisy_pointcloud_color=noisy_color_pointcloud,
                                               indexes=indexes)
                loss += color_loss
                print("color_loss: ", round(color_loss.item(),6))


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.schedular.step()

        print("Optimization Finished")
        noisy_rgbd.plotly(0).show()
        noisy_reconstruction.plotly(0).show()



if __name__ == "__main__":
    args = arguments()
    config_path = args['config_path']
    config_dict = load_yaml(config_path)
    config_dict.SETTINGS.name = args['name']
    SLAM = Gradient_Flow(config_dict)
    SLAM.train()