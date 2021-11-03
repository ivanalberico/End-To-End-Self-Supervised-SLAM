"""
This file will train a scaling layer for depth maps, consisting of an affine transformation (scale + offset).
The motivation for this is to use the learned parameters to bring the network depth predictions
to a resonable estimate of the scale for the current scene, such that ground truth poses
can be used for the refinement.
This file DOES NOT perform the online refinement for indoor scenes.
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
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
        #self.writer = SummaryWriter()

        self.dataset_init()
        self.model_init()
        self.view_reconstruction_init()
        self.losses_init()

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
            traj = "living_room_traj0_frei_png"
            mylist = []
            mylist.append(traj)
            traj_tuple = tuple(mylist)

            self.dataset = ICL(basedir=self.data_path,
                               trajectories=traj_tuple,
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

        # ----- Set SLAM Models -----

        if self.args.MODEL.slam == "ICPSLAM":
            self.models["SLAM"] = ICPSLAM(odom=self.args.MODEL.odom,
                                          numiters=self.args.MODEL.numiters,
                                          device=self.device)
            self.models["GT_SLAM"] = ICPSLAM(odom="gt", device=self.device)

        elif self.args.MODEL.slam == "PointFusion":
            self.models["SLAM"] = PointFusion(odom=self.args.MODEL.odom,
                                              dist_th=self.args.MODEL.dist_th,
                                              angle_th= self.args.MODEL.angle_th,
                                              sigma= self.args.MODEL.sigma,
                                              numiters=self.args.MODEL.numiters,
                                              device=self.device)

            self.models["GT_SLAM"] = PointFusion(odom="gt", device=self.device)

        print("Using the {} for SLAM".format(self.args.MODEL.slam))

        # ----- Set Depth Prediction Networks -----

        "Resnet Encoder"
        if self.args.MODEL.depth_network == "monodepth2":
            self.models["depth_encoder"] = ResnetEncoder(self.args.MODEL.num_layers,
                                                         self.args.MODEL.weights_init_encoder == "imagenet")
            self.models["depth_encoder"].to(self.device)
            # self.train_params += list(self.models["depth_encoder"].parameters())

            "Depth Decoder"
            self.models["depth_decoder"] = DepthDecoder(self.models["depth_encoder"].num_ch_enc,
                                                        self.args.DATA.scales)
            self.models["depth_decoder"].to(self.device)

            print("Loaded ResNet{} based depth network".format(self.args.MODEL.num_layers))
            # self.train_params += list(self.models["depth_decoder"].parameters())

        elif self.args.MODEL.depth_network == "indoor":
            self.models["depth"] = DispResNet_Indoor(num_layers=self.args.MODEL.num_layers,
                                                     pretrained=self.args.MODEL.weights_init_encoder == "imagenet")
            #self.train_params += list(self.models["depth"].parameters())
            self.models["depth"].to(self.device)
        else:
            raise ValueError("Given {} is not a valid depth network option".format(self.args.MODEL.depth_network))

        # Load pretrained model weights

        if self.args.MODEL.use_pretrained_models and self.args.MODEL.depth_network == "monodepth2":
            self.load_model()
        elif self.args.MODEL.use_pretrained_models and self.args.MODEL.depth_network == "indoor":
            self.load_model_indoor()


        if self.args.OPTIMIZATION.load_optimizer and self.args.MODEL.load_depth_path:
            self.load_optimizer()
        elif self.args.OPTIMIZATION.load_optimizer and not self.args.MODEL.load_depth_path:
            raise ValueError("Load optimizer only if pretrained depth is used !! Set Flag off!")

        # Print losses in use
        losses = "photometric"
        for k,v in self.args.LOSS.items():
            if not("weight" in k or "type" in k) and (v == True):
                losses += " "
                losses += k

        print("Using losses:", losses)

    def view_reconstruction_init(self):
        self.backproject_depth = BackprojectDepth(self.args.OPTIMIZATION.batch_size,
                                                  self.args.DATA.height,
                                                  self.args.DATA.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.args.OPTIMIZATION.batch_size,
                                    self.args.DATA.height,
                                    self.args.DATA.width)
        self.project_3d.to(self.device)

    def losses_init(self):
        self.ssim = SSIM()
        self.ssim = self.ssim.to(self.device)

        self.chamfer = ChamferDistance()
        self.chamfer = self.chamfer.to(self.device)

    def scale_layer_init(self, init_value=0.5):
        """
        Initializes the scaling layer in the model.

        Two possible implementations are considered:
        1. Convolutional layer to learn scale + offset.
        2. Scalar scale layer to learn scale as single parameter.
        Default: 1

        Uncomment the corresponding sections below for choosing between implementations.
        """

        # Set training params for scale learning
        self.train_params = []

        # ------ UNCOMMENT BELOW FOR CONV LAYER (weight(scale) + bias(offset)) --------
        self.models["scale_layer"] = Conv1x1(1, 1, init_value=init_value, bias=True)
        self.train_params += list(self.models["scale_layer"].parameters())
        print("Number of trainable params = {}".format(len(self.train_params)))
        print("Type of trainable params = {}".format(type(self.train_params)))
        print("Trainable params = {}, {}".format(self.train_params[len(self.train_params)-2], self.train_params[len(self.train_params)-1]))
        self.models["scale_layer"].to(self.device)
        
        # ------ UNCOMMENT BELOW FOR SCALE LAYER (single scalar weight) --------
        # self.models["scale_layer"] = ScaleLayer(init_value=init_value)
        # self.train_params += list(self.models["scale_layer"].parameters())
        # print("Number of trainable params = {}".format(len(self.train_params)))
        # print("Type of trainable params = {}".format(type(self.train_params)))
        # print("Trainable params = {}".format(self.train_params[len(self.train_params)-1]))
        # self.models["scale_layer"].to(self.device)

        # Set optimizer settings
        self.optimizer = define_optim(self.args, self.train_params)
        self.schedular = define_schedular(self.args, self.optimizer)

    def train(self):
        """
        Main Training Loop
        """
        self.epoch = 0
        self.step = 0

        print("Scale Learning Started")

        "Initialize a small sequence, "
        for iter, batch in enumerate(self.train_loader):

            colors, gt_depths, intrinsics, poses, transform = batch[0], batch[1], batch[2], batch[3], batch[4]
            colors /= 255.0

            colors, gt_depths, intrinsics, poses, transform = colors.to(self.device), \
                                                   gt_depths.to(self.device), \
                                                   intrinsics.to(self.device), \
                                                   poses.to(self.device), \
                                                   transform.to(self.device)

            rgbd = RGBDImages(colors, gt_depths, intrinsics, poses)
            self.gt_reconstruction, _ = self.models["GT_SLAM"](rgbd)
            self.gt_reconstruction = self.gt_reconstruction.detach()

            # print("Type of search grid: {}".format(type(self.args.SCALE_GRID_SEARCH.grid)))
            for init_depth_scale in self.args.SCALE_GRID_SEARCH.grid:

                print("--------------------------------")
                print("Started learning scale with initial scale value = {}".format(init_depth_scale))

                self.scale_layer_init(init_depth_scale)

                """
                Depth Estimation
                """
                scale = 0       # Only using 1 scale feature for now.
                self.initial_depths = {}
                for refine_step in range(self.args.OPTIMIZATION.refinement_steps):
                    """
                    Notes: Outputs is a dict that contains the depth prediction for an entire sequence. 
                    Convert Depth Sequence stored in dictonary to depth sequence stored as a Tensor (to make it similar to gradslam)
                    """


                    inputs = OrderedDict()     # Used for the depth part
                    encoder_features = []
                    depth_tensor = []           # Used for the SLAM part

                    for index in range(self.sequence_length):

                        if self.args.MODEL.depth_network == "monodepth2":
                            encoder_features.append(self.models["depth_encoder"](colors[:, index, ...]))
                            inputs.update(self.models["depth_decoder"](encoder_features[index], index))

                            # Convert Disparity into Depth
                            inputs[("depth", index, scale)] = convert_disp_to_depth(inputs[("disp", index, scale)],
                                                                                    self.args.DATA.min_depth,
                                                                                    self.args.DATA.max_depth)

                            # Scaled depth maps
                            inputs[("scaled_depth", index, scale)] = self.models["scale_layer"](inputs[("depth", index, scale)])

                            if self.args.ABLATION.scale_intrinsics:
                                focal_data = intrinsics[0, 0, 0, 0]
                                focal_pretrain = self.args.ABLATION.focal_pretrain
                                inputs[("depth", index, scale)] = scale_by_f(focal_data=focal_data,
                                                                            focal_pretrain=focal_pretrain,
                                                                            depth= inputs[("depth", index, scale)])

                            if self.args.ABLATION.scaled_depth:
                                inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

                            if self.args.OPTIMIZATION.refinement == "PFT" and self.args.LOSS.depth_regularizer and refine_step == 0:
                                self.initial_depths[("initial_depth", index, scale)] = inputs[("depth", index, scale)].copy().detach()

                            inputs[("gt_depth", index, scale)] = gt_depths[:, index, ...]
                            
                            # depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension
                            depth_tensor.append(inputs[("scaled_depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension

                        elif self.args.MODEL.depth_network == "indoor":
                            inputs.update(self.models["depth"](colors[:, index, ...], index))

                            inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]
                            
                            # Generate scaled depth maps
                            inputs[("scaled_depth", index, scale)] = self.models["scale_layer"](inputs[("depth", index, scale)])

                            if self.args.ABLATION.scale_intrinsics:
                                focal_data = intrinsics[0, 0, 0, 0]
                                focal_pretrain = self.args.ABLATION.focal_pretrain
                                inputs[("depth", index, scale)] = scale_by_f(focal_data=focal_data,
                                                                            focal_pretrain=focal_pretrain,
                                                                            depth= inputs[("depth", index, scale)])

                            if self.args.ABLATION.scaled_depth:
                                inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

                            if self.args.OPTIMIZATION.refinement == "PFT" and self.args.LOSS.depth_regularizer and refine_step == 0:
                                self.initial_depths[("initial_depth", index, scale)] = inputs[("depth", index, scale)].clone().detach()

                            depth_tensor.append(inputs[("scaled_depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension

                            inputs[("gt_depth", index, scale)] = gt_depths[:, index, ...]

                    del encoder_features # Free Space

                    depth_tensor = torch.cat(depth_tensor, dim=1)           # use this for SLAM!
                    depth_tensor = depth_tensor.permute(0, 1, 3, 4, 2)      # Change to channel last representation

                    if self.args.DATA.use_gt_pose:
                        new_poses = poses
                    else:
                        if self.step == 0:      # Initially the pose is taken as identity
                            new_poses = torch.eye(4, device=self.device).view(1, 1, 4, 4).repeat(self.args.OPTIMIZATION.batch_size,
                                                                                                self.sequence_length, 1, 1)

                    noisy_rgbd = RGBDImages(rgb_image=colors,
                                            depth_image=depth_tensor,
                                            intrinsics=intrinsics,
                                            poses=new_poses)

                    if self.args.DATA.use_gt_pose:
                        """
                        We get rid of poses from SLAM if we using GT Pose, otherwise we will take the new pose from the SLAM 
                        and feed it back into the network.
                        """
                        noisy_reconstruction, _ = self.models["SLAM"](noisy_rgbd)
                    else:
                        noisy_reconstruction, new_poses = self.models["SLAM"](noisy_rgbd)

                    noisy_pointcloud = noisy_reconstruction.points_list[0].unsqueeze(0).contiguous()    # Only optimizing points not color!
                    inputs[("noisy_pointcloud")] = noisy_pointcloud

                    if self.step == 0 and self.args.VIZ.plot_first_step:
                        plt.imshow(inputs[("depth", 0, 0)][0].detach().cpu().squeeze(), cmap=self.color_map)
                        plt.axis('off')
                        plt.show()
                        #noisy_reconstruction.plotly(0).show()

                    # Note that we use the transformations between GT poses
                    total_loss = self.depth_refinement(colors, inputs, intrinsics, transform)

                    new_poses = new_poses.detach()
                    self.step += 1

                    if self.args.DEBUG.print_metrics:
                        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_metrics(dataset=self.args.DATA.name,
                                                                                    gt=gt_depths[0][1],
                                                                                    pred=inputs[("depth", 0, 0)][0])
                        print("Iter:", iter,
                            "Refine_Step:", refine_step,
                            "Total_Loss:", round(total_loss, 5),
                            "abs_rel: ", round(abs_rel.item(), 5),
                            "rmse: ", round(rmse.item(), 5),
                            "a1: ", round(a1.item(), 5))
                    else:
                        print("Iter:", iter,
                            "Refine_Step:", refine_step,
                            "Total_Loss:", round(total_loss, 5))
                    
                    if self.step % 10 == 0:
                        print("Current trainable param = {}, {}".format(self.train_params[len(self.train_params)-2], self.train_params[len(self.train_params)-1]))

                # Show final/trained (scale+offset) parameters for current experiment
                print("Trainable param after training = {}, {}".format(self.train_params[len(self.train_params)-2], self.train_params[len(self.train_params)-1]))

            self.schedular.step()

            if self.args.DEBUG.early_stop:
                if  iter == self.args.DEBUG.iter_stop:
                    break

        # Final plot after training. [Maybe we should perform validation step on other scenes]
        if self.args.VIZ.plot_final_step:
            plt.imshow(inputs[("depth", 0, 0)][0].detach().cpu().squeeze(), cmap=self.color_map)
            plt.axis('off')
            plt.show()
            noisy_reconstruction.plotly(0).show()
        if self.args.VIZ.plot_gt:
            self.gt_reconstruction.plotly(0).show()

        # # Show final/trained parameter
        # print("Trainable param after training = {}, {}".format(self.train_params[len(self.train_params)-1], self.train_params[len(self.train_params)-1]))

    def depth_refinement(self, colors, inputs, intrinsics, poses):
        """
        Essentially Unsupervised Depth Estimation in refinement steps.
        outputs only contain the outputs from the view synthesis module, NOT initial data that is available from other sources
        """
        outputs = {}
        inputs.update(self.process_inputs(colors, inputs, intrinsics, poses))
        outputs.update(self.novel_view_synthesis(inputs))
        loss = self.compute_losses(inputs, outputs)
        return loss

    def process_inputs(self, colors, inputs, intrinsics, poses):
        """
        Process the inputs suitable for view sythesis. Easier to understand for others then.

        NOTE: This function assumes sequence length of 2 or 3. For larger sequence lengths change appropriately.
        Making everything channel first representation too for PyTorch.

        """

        if self.sequence_length == 3:
            inputs["source_frame", -1] = colors[:, 0, ...].permute(0, 3, 1, 2)
            inputs["target_frame"] = colors[:, 1, ...].permute(0, 3, 1, 2)
            inputs["source_frame", 1] = colors[:, 2, ...].permute(0, 3, 1, 2)

            if self.args.MODEL.depth_network == "monodepth2" and self.args.DATA.normalize_intrinsics:
                # TODO: SLAM is giving errors when using this normalization.
                inputs["K"] = normalize_intrinsics(self.args, intrinsics[:, 0,...])  # In a particular sequence, the intrinsic should be constant.
                inputs["Inverse_K"] = torch.pinverse(inputs["K"])
            else:
                inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                inputs["Inverse_K"] = torch.pinverse(inputs["K"])

            inputs["source_depth", -1] = inputs["scaled_depth", 0, 0]
            inputs["target_depth"] = inputs["scaled_depth", 1, 0]
            inputs["source_depth", 1] = inputs["scaled_depth", 2, 0]

            inputs["source_disp", -1] = inputs["disp", 0, 0]
            inputs["target_disp"] = inputs["disp", 1, 0]
            inputs["source_disp", 1] = inputs["disp", 2, 0]

            # IF SEQUENCE LENGTH IS GREATER THAN 3, handle pose appropriately.

            inputs["T", -1] = poses[:, 1, ...]  # This represents the pose 0 -> -1
            inputs["T", 1] = inverse_T_matrix(poses[:, 2, ...])  # This represents the pose 0 -> 1

            # GT Depths
            scale = 0
            for index in range(len(self.args.DATA.frames)):
                inputs[("sparse_gt_depth", index, scale)], inputs[("sparse_mask", index, scale)] = sparse_sampling(self.args.LOSS.sampling_type,
                                                                     self.args.LOSS.sampling_prob,
                                                                     inputs[("gt_depth", index, scale)])

        elif self.sequence_length == 2:
            if self.args.DATA.frames[1] < 0:
                inputs["source_frame", -1] = colors[:, 0, ...].permute(0, 3, 1, 2)
                inputs["target_frame"] = colors[:, 1, ...].permute(0, 3, 1, 2)

                if self.args.MODEL.depth_network == "monodepth2" and self.args.DATA.normalize_intrinsics:
                    # TODO: SLAM is giving errors when using this normalization.
                    inputs["K"] = normalize_intrinsics(self.args, intrinsics[:, 0,
                                                                  ...])  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])
                else:
                    inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])

                inputs["source_depth", -1] = inputs["scaled_depth", 0, 0]
                inputs["target_depth"] = inputs["scaled_depth", 1, 0]

                inputs["source_disp", -1] = inputs["disp", 0, 0]
                inputs["target_disp"] = inputs["disp", 1, 0]

                inputs["T", -1] = poses[:, 1, ...]  # This represents the pose 0 -> -1

            elif self.args.DATA.frames[1] > 0:
                inputs["target_frame"] = colors[:, 0, ...].permute(0, 3, 1, 2)
                inputs["source_frame", 1] = colors[:, 1, ...].permute(0, 3, 1, 2)

                if self.args.MODEL.depth_network == "monodepth2" and self.args.DATA.normalize_intrinsics:
                    inputs["K"] = normalize_intrinsics(self.args, intrinsics[:, 0,
                                                                  ...])  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])
                else:
                    inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])

                inputs["target_depth"] = inputs["scaled_depth", 0, 0]
                inputs["source_depth", 1] = inputs["scaled_depth", 1, 0]

                inputs["target_disp"] = inputs["disp", 0, 0]
                inputs["source_disp", 1] = inputs["disp", 1, 0]

                inputs["T", 1] = inverse_T_matrix(poses[:, 1, ...])

        else:
            raise ValueError("Sequence Length of 2 and 3 is only supported")

        return inputs

    def novel_view_synthesis(self, inputs):
        outputs = {}
        for frame in self.args.DATA.frames[1:]:
            camera_points = self.backproject_depth(inputs["target_depth"], inputs["Inverse_K"])

            # if self.args.DEBUG.plot and self.step % 10 == 0:
            #     "Plots photometric error maps and saves them in the given location"
            #     plt.imshow(inputs["target_depth"].detach().cpu().squeeze(), cmap=self.color_map)
            #     plot_path = os.path.join(self.args.DEBUG.plot_path + "/depth_{}".format(self.step))
            #     plt.axis('off')
            #     plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)

            if self.args.LOSS.geometric:
                pixel_coordinates, warped_depth, valid_mask = self.project_3d(points=camera_points,
                                                                              K=inputs["K"],
                                                                              T=inputs["T", frame],
                                                                              geometric=True)
                outputs[("warped_depth", frame)] = warped_depth

                outputs[("valid_mask", frame)] = valid_mask

                outputs[("synthesized_frame", frame)] = F.grid_sample(inputs["source_frame", frame],
                                                                      pixel_coordinates,
                                                                      padding_mode=self.args.MODEL.padding_mode,
                                                                      align_corners=True)

                outputs[("interpolated_depth", frame)] = F.grid_sample(inputs["source_depth", frame],
                                                                      pixel_coordinates,
                                                                      padding_mode=self.args.MODEL.padding_mode,
                                                                      align_corners=False)

            else:
                pixel_coordinates, valid_mask = self.project_3d(points=camera_points,
                                                    K=inputs["K"],
                                                    T=inputs["T", frame],
                                                    geometric=False)

                outputs[("valid_mask", frame)] = valid_mask

                outputs[("synthesized_frame", frame)] = F.grid_sample(inputs["source_frame", frame],
                                                                      pixel_coordinates,
                                                                      padding_mode=self.args.MODEL.padding_mode,
                                                                      align_corners=False)

                # if self.args.DEBUG.plot and self.step == 0:
                #     "Plots photometric error maps and saves them in the given location"
                #     plt.imshow(inputs["target_frame"].detach().cpu().squeeze().permute(1,2,0))
                #     plot_path = os.path.join(self.args.DEBUG.plot_path + "/tF_{}".format(self.step))
                #     plt.axis('off')
                #     plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)
                #
                # if self.args.DEBUG.plot and self.step == 0:
                #     "Plots photometric error maps and saves them in the given location"
                #     plt.imshow(inputs["source_frame", frame].detach().cpu().squeeze().permute(1,2,0))
                #     plot_path = os.path.join(self.args.DEBUG.plot_path + "/sF_{}".format(self.step))
                #     plt.axis('off')
                #     plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)
                #
                # if self.args.DEBUG.plot and self.step % 10 == 0:
                #     "Plots photometric error maps and saves them in the given location"
                #     plt.imshow(outputs["synthesized_frame", frame].detach().cpu().squeeze().permute(1,2,0))
                #     plot_path = os.path.join(self.args.DEBUG.plot_path  + "/synthF_{}".format(self.step))
                #     plt.axis('off')
                #     plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)
                #     plt.close("all")
        return outputs

    def compute_losses(self, inputs, outputs):
        losses = {}
        loss = 0

        self.optimizer.zero_grad()

        photmetric = self.compute_photometric_loss(inputs=inputs,
                                                   outputs=outputs)

        # print("Photometric loss tensor size = {}".format(photmetric.shape))

        if self.args.LOSS.min_reprojection:
            photmetric = photmetric
        else:
            photmetric = photmetric.mean(1, keepdim=True)

        # if self.args.DEBUG.plot and self.args.LOSS.min_reprojection and self.step % 10 == 0:
        #     "Plots photometric error maps and saves them in the given location"
        #     plt.imshow(torch.min(photmetric, dim=1)[0].detach().cpu().squeeze(), cmap="binary")
        #     plot_path = os.path.join(self.args.DEBUG.plot_path + "/pE_{}".format(self.step))
        #     plt.axis('off')
        #     plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)
        # elif self.args.DEBUG.plot and self.step % 10 == 0:
        #     plt.imshow(photmetric[0].detach().cpu().squeeze(), cmap="binary")
        #     plot_path = os.path.join(self.args.DEBUG.plot_path + "/pE_{}".format(self.step))
        #     plt.axis('off')
        #     plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)

        if self.args.LOSS.auto_masking:
            auto_masking = self.compute_automasking_loss(inputs=inputs,
                                                         outputs=outputs)

            if self.args.LOSS.min_reprojection:
                auto_masking += torch.randn(auto_masking.shape).cuda() * 0.00001        # Break tie's
            else:
                auto_masking = auto_masking.mean(1, keepdim=True)

            photmetric = torch.cat((auto_masking, photmetric), dim=1)

        if photmetric.shape[1] == 1:
           optimize = photmetric

           # save photometric losses map
        #    save_image(optimize, '/cluster/scratch/semilk/photometric_error.png')

           optimize = optimize.mean()
        else:
            optimize, indexs = torch.min(photmetric, dim=1)

            # print("Executing path B...")
            # save photometric losses map
            # save_image(optimize, '/cluster/scratch/semilk/photometric_error.png')

            optimize = optimize.mean()

        loss += optimize

        losses["photometric_loss"] = optimize.item()

        if self.args.LOSS.geometric:
            geometric = self.compute_geometric_loss(outputs=outputs)
            geometric = geometric.mean()
            loss += geometric * self.args.LOSS.geometric_weight
            losses["geometric_loss"] = geometric.item()

        if self.args.LOSS.smoothness:
            smooth_loss = self.compute_smoothness_loss(inputs=inputs)
            loss += smooth_loss * self.args.LOSS.smoothness_weight
            losses["smoothn_loss"] = smooth_loss.item()

        if self.args.LOSS.depth_regularizer:
            depth_reg = self.compute_depth_regularizer(inputs=inputs)
            loss += depth_reg * self.args.LOSS.depth_regularizer_weight
            losses["depth regularizer"] = depth_reg.item()

        if self.args.LOSS.knn_points:
            knn_loss, indexs = knn_points_loss(gt_pointcloud=self.gt_reconstruction.points_list[0].unsqueeze(0).contiguous(),
                                               noisy_pointcloud=inputs["noisy_pointcloud"])

            loss += knn_loss * self.args.LOSS.knn_points_weight
            print("knn_loss", knn_loss.item())          # Turn off unless debug

        if self.args.LOSS.chamfer_distance:
            chamfer_dist = 0.5 * self.chamfer(inputs["noisy_pointcloud"],
                                              self.gt_reconstruction.points_list[0].unsqueeze(0).contiguous(),
                                              bidirectional=True)

            loss += chamfer_dist * self.args.LOSS.chamfer_weight
            print("chamfer_loss", chamfer_dist.item())  # Turn off unless debug

        if self.args.LOSS.supervise_depth:
            gt_loss = self.compute_gt_depth_loss(inputs=inputs)
            loss += gt_loss * self.args.LOSS.gt_depth_weight
            losses["gt_depth_loss"] = gt_loss.item()

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_photometric_loss(self, inputs, outputs):
        """
        Compute the photometric loss here
        """
        photometric_losses = []
        for frame in self.args.DATA.frames[1:]:
            if self.args.LOSS.photometric_mask:
                masked_prediction = outputs[("synthesized_frame", frame)] * outputs["valid_mask", frame]
                masked_target = inputs[("target_frame")] * outputs["valid_mask", frame]
                photometric_losses.append(photometric_loss(ssim=self.ssim,
                                                           prediction=masked_prediction,
                                                           target=masked_target))
            else:
                prediction = outputs[("synthesized_frame", frame)]
                target = inputs[("target_frame")]
                photometric_losses.append(photometric_loss(ssim=self.ssim,
                                                           prediction=prediction,
                                                           target=target))

        photometric_losses = torch.cat(photometric_losses, 1)
        return photometric_losses

    def compute_automasking_loss(self, inputs, outputs):
        """
        Compute identity losses for auto-masking technique by monodepth2
        """
        auto_masking_losses = []

        for frame in self.args.DATA.frames[1:]:
            if self.args.LOSS.photometric_mask:
                masked_prediction = inputs[("source_frame", frame)] * outputs["valid_mask", frame]
                masked_target = inputs[("target_frame")] * outputs["valid_mask", frame]
                auto_masking_losses.append(photometric_loss(ssim=self.ssim,
                                                            prediction=masked_prediction,
                                                            target=masked_target))
            else:
                prediction = inputs[("source_frame", frame)]
                target = inputs[("target_frame")]
                auto_masking_losses.append(photometric_loss(ssim=self.ssim,
                                                           prediction=prediction,
                                                           target=target))

        auto_masking_losses = torch.cat(auto_masking_losses, 1)
        return auto_masking_losses

    def compute_geometric_loss(self, outputs):
        """
        Compute the geometric loss here
        """
        geometric_losses = []
        for frame in self.args.DATA.frames[1:]:
            geometric_losses.append(geometric_consistency_loss(outputs, frame, self.device))

        geometric_losses = torch.stack(geometric_losses, dim=0)
        return geometric_losses

    def compute_smoothness_loss(self, inputs):
        """
        compute the smoothness loss here on the disparities! Not depths.
        """
        disparity = inputs[("disp", 0, 0)]
        mean_disparity = disparity.mean(2, True).mean(3, True)
        norm_disparity = disparity / (mean_disparity + 1e-7)
        smooth_loss = disparity_smoothness_loss(disp=norm_disparity,
                                                img=inputs[("target_frame")])

        return smooth_loss

    def compute_depth_regularizer(self, inputs):
        """
        This function computes a regularizer such that the refined depth does not sway too far from the initial estimate of depth.
        """
        scale = 0
        reg = 0
        for frame in range(len(self.args.DATA.frames)): 
            reg += depth_reguralizer(initial_depth=self.initial_depths[("initial_depth", frame, scale)],
                                    refined_depth=inputs[("depth", frame, scale)],
                                    loss_func=self.args.LOSS.depth_regularizer_type)

        return reg

    def compute_gt_depth_loss(self, inputs):
        gt_loss = 0
        scale = 0
        for frame in range(len(self.args.DATA.frames)):
            gt_loss += depth_gt_loss(prediction=inputs[("depth", frame, scale)],
                                     sparse_groundtruth=inputs[("sparse_gt_depth", frame, scale)],
                                     sparse_mask=inputs[("sparse_mask", frame, scale)])

        return gt_loss
        
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

        assert os.path.isdir(self.args.MODEL.load_depth_path), "Cannot find folder {}".format(self.args.MODEL.load_depth_path)
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

        assert os.path.isdir(self.args.MODEL.load_depth_path), "Cannot find folder {}".format(self.args.MODEL.load_depth_path)
        print("loading model from folder {}".format(self.args.MODEL.load_depth_path))
        n = "depth"
        print("Loading {} weights...".format(n))
        path = os.path.join(self.args.MODEL.load_depth_path, "{}.pth.tar".format(n))
        pretrained_dict = torch.load(path)
        self.models[n].load_state_dict(pretrained_dict["state_dict"])
        print("Loaded Indoor Depth Model")

    #TODO: Add Save_Model

    def load_optimizer(self):
        """
        Load Optimizer state dict if resuming training.

        Flags To Consider:
        MODEL.load_depth_path: Path to pretrained models and their optimizers for resume training
        MODEL.pretrained_models_list: List of models to load
        """
        optimizer_load_path = os.path.join(self.args.MODEL.load_depth_path, "{}.pth".format(self.args.OPTIMIZATION.optimizer))
        if os.path.isfile(optimizer_load_path):
            print("Loading Optimizer Weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Optimizer Not Found. Randomly Initialized")


if __name__ == "__main__":
    args = arguments()
    config_path = args['config_path']
    config_dict = load_yaml(config_path)
    config_dict.SETTINGS.name = args['name']
    SLAM = Depth_Estimation(config_dict)
    SLAM.train()
