"""
This file does median scaling to precompute full median scaling of the entire dataset.
"""

import os
import cv2
import yaml
import json
import torch  # Sorry to the google supervisors
import shutil
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchviz import make_dot, make_dot_from_trace

# Imports from our files
from loss.losses import *
from utils.training_utils import *
from utils.arguments import arguments
from depth_estimation.networks import *
from utils.modify_images import corrupt_rgbd
from slam.custom_slam import image_recover_slam
from utils.yaml_configs import load_yaml, save_yaml
from utils.advanced_vis import plotly_map_update_visualization
from depth_estimation.view_synthesis import BackprojectDepth, Project3D

# GradSLAM Imports
import gradslam as gs
from gradslam.datasets import ICL, TUM
from gradslam.slam import ICPSLAM
from gradslam.slam import PointFusion
from chamferdist import ChamferDistance
from gradslam import Pointclouds, RGBDImages


class Depth_Estimation:
    def __init__(self, arguments):
        self.args = arguments
        self.device = torch.device("cuda" if self.args.SETTINGS.device == "cuda" else "cpu")
        self.sequence_length = len(self.args.DATA.frames)
        self.color_map = plt.cm.get_cmap("magma").reversed()
        # self.writer = SummaryWriter()

        self.dataset_init()
        self.model_init()

        if self.args.ABLATION.scale_intrinsics:
            print("Scaling Intrinsics")
        if self.args.ABLATION.scaled_depth:
            print("Scaling Depth Maps")

    def dataset_init(self):
        """
        Initialize  datasets in this function

        Input:
        None
        Output:
        None
        """

        print("Loading Images of Size {} x {}".format(self.args.DATA.width, self.args.DATA.height))

        self.data_path = os.path.join(self.args.DATA.data_path, self.args.DATA.name)
        if self.args.DATA.name == "ICL":
            self.dataset = ICL(basedir=self.data_path,
                               seqlen=self.sequence_length,
                               height=self.args.DATA.height,
                               width=self.args.DATA.width,
                               dilation=self.args.DATA.dilation,
                               stride=self.args.DATA.stride,
                               start=self.args.DATA.start)

        elif self.args.DATA.name == "TUM":
            self.dataset = TUM(basedir=self.data_path,
                               seqlen=self.sequence_length,
                               height=self.args.DATA.height,
                               width=self.args.DATA.width,
                               dilation=self.args.DATA.dilation,
                               stride=self.args.DATA.stride,
                               start=self.args.DATA.start)

        self.train_loader = DataLoader(dataset=self.dataset,
                                       batch_size=self.args.OPTIMIZATION.batch_size,
                                       shuffle=False,
                                       num_workers=self.args.SETTINGS.num_workers,
                                       pin_memory=True,
                                       drop_last=True)

        print("{} Dataset Loaded".format(self.args.DATA.name))

    def model_init(self):
        """
        Define all training models in here

        Variables To Consider:
        models: contains the list of all models
        train_params: contains the trainable params

        """
        self.models = {}
        self.train_params = []

        print("Initializing Models")

        "Resnet Encoder"
        if self.args.MODEL.depth_network == "monodepth2":
            self.models["depth_encoder"] = ResnetEncoder(self.args.MODEL.num_layers,
                                                         self.args.MODEL.weights_init_encoder == "imagenet")
            self.models["depth_encoder"].to(self.device)
            self.train_params += list(self.models["depth_encoder"].parameters())

            "Depth Decoder"
            self.models["depth_decoder"] = DepthDecoder(self.models["depth_encoder"].num_ch_enc,
                                                        self.args.DATA.scales)
            self.models["depth_decoder"].to(self.device)

            print("Loaded ResNet{} based depth network".format(self.args.MODEL.num_layers))
            self.train_params += list(self.models["depth_decoder"].parameters())

        elif self.args.MODEL.depth_network == "indoor":
            self.models["depth"] = DispResNet_Indoor(num_layers=self.args.MODEL.num_layers,
                                                     pretrained=self.args.MODEL.weights_init_encoder == "imagenet")
            self.train_params += list(self.models["depth"].parameters())
            self.models["depth"].to(self.device)
        else:
            raise ValueError("Given {} is not a valid depth network option".format(self.args.MODEL.depth_network))

        if self.args.MODEL.use_pretrained_models and self.args.MODEL.depth_network == "monodepth2":
            self.load_model()
        elif self.args.MODEL.use_pretrained_models and self.args.MODEL.depth_network == "indoor":
            self.load_model_indoor()

    def find_median_scale(self):
        """
        Main Training Loop
        """
        self.epoch = 0
        self.step = 0

        print("Computing Median Scale for {} dataset".format(self.args.DATA.name))

        self.median_scale = []

        "Initialize a small sequence, "
        for iter, batch in enumerate(self.train_loader):

            colors, gt_depths, intrinsics, poses = batch[0], batch[1], batch[2], batch[3]
            colors /= 255.0

            colors, gt_depths, intrinsics, poses = colors.to(self.device), \
                                                   gt_depths.to(self.device), \
                                                   intrinsics.to(self.device), \
                                                   poses.to(self.device)


            scale = 0  # Only using 1 scale for now.   kinda redundent for now (remove usage if not needed)
            self.initial_depths = {}


            inputs = OrderedDict()
            encoder_features = []
            depth_tensor = []

            for index in range(self.sequence_length):

                if self.args.MODEL.depth_network == "monodepth2":
                    encoder_features.append(self.models["depth_encoder"](colors[:, index, ...]))
                    inputs.update(self.models["depth_decoder"](encoder_features[index], index))

                    # Convert Disparity into Depth
                    inputs[("depth", index, scale)] = convert_disp_to_depth(inputs[("disp", index, scale)],
                                                                            self.args.DATA.min_depth,
                                                                            self.args.DATA.max_depth)

                    if self.args.ABLATION.scale_intrinsics:
                        focal_data = intrinsics[0, 0, 0, 0]
                        focal_pretrain = self.args.ABLATION.focal_pretrain
                        inputs[("depth", index, scale)] = scale_by_f(focal_data=focal_data,
                                                                     focal_pretrain=focal_pretrain,
                                                                     depth=inputs[("depth", index, scale)])

                    if self.args.ABLATION.scaled_depth:
                        inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

                    depth_tensor.append(
                        inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension

                elif self.args.MODEL.depth_network == "indoor":

                    inputs.update(self.models["depth"](colors[:, index, ...], index))

                    inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]

                    if self.args.ABLATION.scaled_depth:
                        inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

                    depth_tensor.append(
                        inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension

            del encoder_features  # Free Space

            depth_tensor = torch.cat(depth_tensor, dim=1)  # use this for SLAM!
            depth_tensor = depth_tensor.permute(0, 1, 3, 4, 2)  # Change to channel last representation

            ratio = torch.median(gt_depths) / torch.median(depth_tensor)
            self.median_scale.append(ratio.cpu().detach().numpy())
            print("Iter:", iter, "ratio:", ratio)

        self.median_scale = np.median(np.array(self.median_scale))
        print("Median Scale is:", self.median_scale)


    def load_model(self):
        """
        Load pretrained models from disk.

        Variables To Consider:
        self.models: Contains List of All Trainable Models.

        Flags To Consider:
        MODEL.load_depth_path: Path to pretrained models and their optimizers for resume training
        MODEL.pretrained_models_list: List of models to load

        """
        self.args.MODEL.load_depth_path = os.path.expanduser(self.args.MODEL.load_depth_path)

        assert os.path.isdir(self.args.MODEL.load_depth_path), "Cannot find folder {}".format(
            self.args.MODEL.load_depth_path)
        print("loading model from folder {}".format(self.args.MODEL.load_depth_path))

        for n in self.args.MODEL.pretrained_models_list:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.args.MODEL.load_depth_path, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_model_indoor(self):
        """
        Load pretrained models from disk.

        Variables To Consider:
        self.models: Contains List of All Trainable Models.

        Flags To Consider:
        MODEL.load_depth_path: Path to pretrained models and their optimizers for resume training
        MODEL.pretrained_models_list: List of models to load

        """
        self.args.MODEL.load_depth_path = os.path.expanduser(self.args.MODEL.load_depth_path)

        assert os.path.isdir(self.args.MODEL.load_depth_path), "Cannot find folder {}".format(
            self.args.MODEL.load_depth_path)
        print("loading model from folder {}".format(self.args.MODEL.load_depth_path))
        n = "depth"
        print("Loading {} weights...".format(n))
        path = os.path.join(self.args.MODEL.load_depth_path, "{}.pth.tar".format(n))
        pretrained_dict = torch.load(path)
        self.models[n].load_state_dict(pretrained_dict["state_dict"])
        print("Loaded Indoor Depth Model")




if __name__ == "__main__":
    args = arguments()
    config_path = args['config_path']
    config_dict = load_yaml(config_path)
    config_dict.SETTINGS.name = args['name']
    SLAM = Depth_Estimation(config_dict)
    SLAM.find_median_scale()

