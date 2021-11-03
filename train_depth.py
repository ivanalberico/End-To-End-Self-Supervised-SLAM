"""
This script is used to train the online refinement model outside the full SLAM pipeline.
We used this to develop the online refinement module.
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

        # Initialization of required functions
        self.args = arguments
        self.device = torch.device("cuda" if self.args.SETTINGS.device == "cuda" else "cpu")
        self.sequence_length = len(self.args.DATA.frames)
        self.color_map = plt.cm.get_cmap("magma").reversed()
        self.writer = SummaryWriter("tensorboard_outputs")

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
        """
        self.models = {}
        self.train_params = []

        print("Initializing Models")

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
            
            if self.args.VIZ.tensorboard:
                print("Requested tensorboard output.")

                for module in self.models["depth_decoder"].decoder:

                    if type(module.conv) == Conv3x3:
                        print("Submodule: ", module.conv.conv)
                        module.conv.conv.register_backward_hook(self.hook_fn)
                        print("Hook registered.")
                    elif type(module.conv) == torch.nn.modules.conv.Conv2d:
                        print("Submodule: ", module.conv)
                        module.conv.register_backward_hook(self.hook_fn)
                        print("Hook registered.")

        elif self.args.MODEL.depth_network == "indoor":
            self.models["depth"] = DispResNet_Indoor(num_layers=self.args.MODEL.num_layers,
                                                     pretrained=self.args.MODEL.weights_init_encoder == "imagenet")
            self.train_params += list(self.models["depth"].parameters())
            self.models["depth"].to(self.device)
            
            if self.args.VIZ.tensorboard:
                print("Requested tensorboard output.")

                for module in self.models["depth"].decoder.decoder:
                    if type(module.conv) == Conv3x3:
                        print("Submodule: ", module.conv.conv)
                        module.conv.conv.register_backward_hook(self.hook_fn)
                        print("Hook registered.")
                    elif type(module.conv) == torch.nn.modules.conv.Conv2d:
                        print("Submodule: ", module.conv)
                        module.conv.register_backward_hook(self.hook_fn)
                        print("Hook registered.")
                        
            print("Using indoor pretrained model.")
            
        else:
            raise ValueError("Given {} is not a valid depth network option".format(self.args.MODEL.depth_network))

        if self.args.MODEL.use_pretrained_models and self.args.MODEL.depth_network == "monodepth2":
            self.load_model()
        elif self.args.MODEL.use_pretrained_models and self.args.MODEL.depth_network == "indoor":
            self.load_model_indoor()

        self.optimizer = define_optim(self.args, self.train_params)
        self.schedular = define_schedular(self.args, self.optimizer)

        if self.args.OPTIMIZATION.load_optimizer and self.args.MODEL.load_depth_path:
            self.load_optimizer()
        elif self.args.OPTIMIZATION.load_optimizer and not self.args.MODEL.load_depth_path:
            raise ValueError("Load optimizer only if pretrained depth is used !! Set Flag off!")

    def view_reconstruction_init(self):
        """
        Initialize the novel view synthesis functions
        """
        self.backproject_depth = BackprojectDepth(self.args.OPTIMIZATION.batch_size,
                                                  self.args.DATA.height,
                                                  self.args.DATA.width)
        self.backproject_depth.to(self.device)

        self.project_3d = Project3D(self.args.OPTIMIZATION.batch_size,
                                    self.args.DATA.height,
                                    self.args.DATA.width)
        self.project_3d.to(self.device)

    def losses_init(self):
        """
        Initialize loss functions
        """
        self.ssim = SSIM()
        self.ssim = self.ssim.to(self.device)

        self.chamfer = ChamferDistance()
        self.chamfer = self.chamfer.to(self.device)

    def set_refinement_mode(self):
        """Convert depth model to refinement mode:
        batch norm is in eval mode + frozen params
        """
        # try using frozen batch norm?
        for m in self.models.values():
            m.eval()
            for name, param in m.named_parameters():
                if name.find("bn") != -1:
                    param.requires_grad = False

    def process_disparity(self, inputs, index):
        """
        This function does an additional processing step on the disparity maps by aggregating the original and a flipped version
        """
        left = inputs[("disp", index, 0)][:1]
        right = torch.flip(inputs[("disp", index, 0)][1:], [3])
        middle = 0.5 * (left + right)
        h, w = left.shape[2], left.shape[3]
        l_mesh, _ = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
        l_mesh = l_mesh.to(self.device)
        l_mask  = (1.0 - torch.clip(20 * (l_mesh - 0.05), 0, 1)).unsqueeze(0).unsqueeze(0)
        r_mask = torch.flip(l_mask, [3])
        inputs[("disp", index , 0)] = r_mask * left + l_mask * right + (1.0 - l_mask - r_mask) * middle
        return inputs

    def train(self):
        """
        Main Training Loop
        """
        self.epoch = 0
        self.step = 0

        if self.args.MODEL.refinement_mode:
            self.set_refinement_mode()

        print("SLAM Reconstruction Started")

        "Initialize a small sequence, "
        for iter, batch in enumerate(self.train_loader):

            colors, gt_depths, intrinsics, poses, transform = batch[0], batch[1], batch[2], batch[3], batch[4]
            colors /= 255.0

            colors, gt_depths, intrinsics, poses, transform = colors.to(self.device), \
                                                   gt_depths.to(self.device), \
                                                   intrinsics.to(self.device), \
                                                   poses.to(self.device), \
                                                   transform.to(self.device)
            # Convert to an RGB-D object for SLAM
            rgbd = RGBDImages(colors, gt_depths, intrinsics, poses)

            # Obtain GT Reconstruction from SLAM
            self.gt_reconstruction, _ = self.models["GT_SLAM"](rgbd)
            self.gt_reconstruction = self.gt_reconstruction.detach()

            scale = 0       # Only using 1 scale for now.
            self.initial_depths = {}

            # This is the main refinement loop
            for refine_step in range(self.args.OPTIMIZATION.refinement_steps):
                inputs = OrderedDict()     # Used for the depth part
                encoder_features = []
                depth_tensor = []
                
                # counters for naming of tensorboard outputs (reset for every refinement step)
                if self.args.VIZ.tensorboard:
                    if self.args.MODEL.depth_network == "indoor":
                        self.hook_counter = 33

                    elif self.args.MODEL.depth_network == "monodepth2":
                        self.hook_counter = 10

                    self.refinement_step = refine_step

                # Predict depth for a sequence of frames.
                for index in range(self.sequence_length):

                    if self.args.MODEL.depth_network == "monodepth2":
                        # We no longer use Monodepth2 but it's still here incase we decide to do outdoor
                        encoder_features.append(self.models["depth_encoder"](colors[:, index, ...]))
                        inputs.update(self.models["depth_decoder"](encoder_features[index], index))

                        # Convert Disparity into Depth
                        inputs[("depth", index, scale)] = convert_disp_to_depth(inputs[("disp", index, scale)],
                                                                                 self.args.DATA.min_depth,
                                                                                 self.args.DATA.max_depth)


                        if self.args.ABLATION.scale_intrinsics:
                            # Intrinsics scaling was proposed by CNN-SLAM [Keisuke et al] but we no longer use it since
                            # we do median scaling.
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

                        depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension

                    elif self.args.MODEL.depth_network == "indoor":
                        if self.args.ABLATION.dual_disparity:
                            # Additional processing step that aggregrates two disparity maps into a final one.
                            inputs.update(self.models["depth"](torch.cat([colors[:, index, ...],
                                                                          torch.flip(colors[:, index, ...], [2])], 0),
                                                               index))
                            inputs.update(self.process_disparity(inputs, index))
                        else:
                            inputs.update(self.models["depth"](colors[:, index, ...], index))

                        # depth is the inverse of disparity
                        inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]

                        if self.args.ABLATION.scale_intrinsics:
                            # Intrinsics scaling was proposed by CNN-SLAM [Keisuke et al] but we no longer use it since
                            # we do median scaling.
                            focal_data = intrinsics[0, 0, 0, 0]
                            focal_pretrain = self.args.ABLATION.focal_pretrain
                            inputs[("depth", index, scale)] = scale_by_f(focal_data=focal_data,
                                                                          focal_pretrain=focal_pretrain,
                                                                          depth= inputs[("depth", index, scale)])

                        if self.args.ABLATION.scaled_depth:
                            # Median Scaling
                            inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

                        if self.args.OPTIMIZATION.refinement == "PFT" and self.args.LOSS.depth_regularizer and refine_step == 0:
                            # Store initial depth predictions for depth regularization
                            self.initial_depths[("initial_depth", index, scale)] = inputs[("depth", index, scale)].clone().detach()

                        depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension

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
                # Create a noisy RGB-D item (contains depth predictions)
                noisy_rgbd = RGBDImages(rgb_image=colors,
                                        depth_image=depth_tensor,
                                        intrinsics=intrinsics,
                                        poses=new_poses)

                # Pass noisy RGB-D through SLAM to obtain a 3D reconstruction and corresponding poses.
                if self.args.DATA.use_gt_pose:
                    """
                    We get rid of poses from SLAM if we using GT Pose, otherwise we will take the new pose from the SLAM 
                    and feed it back into the network.
                    """
                    noisy_reconstruction, _ = self.models["SLAM"](noisy_rgbd)
                    new_transform = transform
                else:
                    noisy_reconstruction, new_poses = self.models["SLAM"](noisy_rgbd)
                    new_transform = torch_poses_to_transforms(new_poses)

                noisy_pointcloud = noisy_reconstruction.points_list[0].unsqueeze(0).contiguous()    # Only optimizing points not color!
                inputs[("noisy_pointcloud")] = noisy_pointcloud

                # Viz Funcs
                if self.step == 0 and self.args.VIZ.plot_first_step:
                    plt.imshow(inputs[("depth", 1, 0)][0].detach().cpu().squeeze(), cmap=self.color_map)
                    plt.xlabel("Initial Predicted Depth")
                    plt.show()
                    noisy_reconstruction.plotly(0).show()

                "--- Refinement of Depth Maps ---"
                total_loss = self.depth_refinement(colors, inputs, intrinsics, new_transform)

                new_poses = new_poses.detach()
                self.step += 1

                # Refinement Metrics and print progress
                if self.args.DEBUG.print_metrics:
                    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_metrics(dataset=self.args.DATA.name,
                                                                                gt=gt_depths[0][1],
                                                                                pred=inputs[("depth", 1, 0)][0])
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

            self.schedular.step()
            # When we are only training on one sequence, we break off early.
            if self.args.DEBUG.early_stop and iter == self.args.DEBUG.iter_stop:
                break

        # Final plot after training. [Maybe we should perform validation step on other scenes]
        if self.args.VIZ.plot_final_step:
            plt.imshow(inputs[("depth", 1, 0)][0].detach().cpu().squeeze(), cmap=self.color_map)
            plt.xlabel("Final Refined Depth")
            plt.show()
            noisy_reconstruction.plotly(0).show()
        if self.args.VIZ.plot_gt:
            self.gt_reconstruction.plotly(0).show()

    def depth_refinement(self, colors, inputs, intrinsics, poses):
        """
        Essentially self-supervised depth estimation pipeline in a few refinement steps.
        outputs only contain the outputs from the view synthesis module, NOT initial data.
        """
        outputs = {}
        # Process to make inputs compatiable with our self-supervised depth estimation pipeline
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
                # TODO: SLAM is giving errors if i do this normalization.
                inputs["K"] = normalize_intrinsics(self.args, intrinsics[:, 0,...])  # In a particular sequence, the intrinsic should be constant.
                inputs["Inverse_K"] = torch.pinverse(inputs["K"])
            else:
                inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                inputs["Inverse_K"] = torch.pinverse(inputs["K"])

            inputs["source_depth", -1] = inputs["depth", 0, 0]
            inputs["target_depth"] = inputs["depth", 1, 0]
            inputs["source_depth", 1] = inputs["depth", 2, 0]

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
                    # TODO: SLAM is giving errors if i do this normalization.
                    inputs["K"] = normalize_intrinsics(self.args, intrinsics[:, 0,
                                                                  ...])  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])
                else:
                    inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])

                inputs["source_depth", -1] = inputs["depth", 0, 0]
                inputs["target_depth"] = inputs["depth", 1, 0]

                inputs["source_disp", -1] = inputs["disp", 0, 0]
                inputs["target_disp"] = inputs["disp", 1, 0]

                inputs["T", -1] = poses[:, 1, ...]  # This represents the pose 0 -> -1

                scale = 0
                for index in range(len(self.args.DATA.frames)):
                    inputs[("sparse_gt_depth", index, scale)], inputs[("sparse_mask", index, scale)] = sparse_sampling(
                                                                         self.args.LOSS.sampling_type,
                                                                         self.args.LOSS.sampling_prob,
                                                                         inputs[("gt_depth", index, scale)])

            elif self.args.DATA.frames[1] > 0:
                inputs["target_frame"] = colors[:, 0, ...].permute(0, 3, 1, 2)
                inputs["source_frame", 1] = colors[:, 1, ...].permute(0, 3, 1, 2)

                if self.args.MODEL.depth_network == "monodepth2" and self.args.DATA.normalize_intrinsics:
                    # TODO: SLAM is giving errors if i do this normalization.
                    inputs["K"] = normalize_intrinsics(self.args, intrinsics[:, 0,
                                                                  ...])  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])
                else:
                    inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                    inputs["Inverse_K"] = torch.pinverse(inputs["K"])

                inputs["target_depth"] = inputs["depth", 0, 0]
                inputs["source_depth", 1] = inputs["depth", 1, 0]

                inputs["target_disp"] = inputs["disp", 0, 0]
                inputs["source_disp", 1] = inputs["disp", 1, 0]

                inputs["T", 1] = inverse_T_matrix(poses[:, 1, ...])

                scale = 0
                for index in range(len(self.args.DATA.frames)):
                    inputs[("sparse_gt_depth", index, scale)], inputs[("sparse_mask", index, scale)] = sparse_sampling(
                        self.args.LOSS.sampling_type,
                        self.args.LOSS.sampling_prob,
                        inputs[("gt_depth", index, scale)])

        else:
            raise ValueError("Sequence Length of 2 and 3 is only supported")

        return inputs

    def novel_view_synthesis(self, inputs):
        outputs = {}
        for frame in self.args.DATA.frames[1:]:
            # project the points into 3D
            camera_points = self.backproject_depth(inputs["target_depth"], inputs["Inverse_K"])

            if self.args.DEBUG.plot and self.step % 10 == 0:
                "Plots photometric error maps and saves them in the given location"
                plt.imshow(inputs["target_depth"].detach().cpu().squeeze(), cmap=self.color_map)
                plot_path = os.path.join(self.args.DEBUG.plot_path + "/depth_{}".format(self.step))
                plt.axis('off')
                plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)

            if self.args.LOSS.geometric:
                # we decided to not used geometric loss in the end so proceed to else:
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
                # Create projection flow that we will use to sample on top off from the source frame
                pixel_coordinates, valid_mask = self.project_3d(points=camera_points,
                                                    K=inputs["K"],
                                                    T=inputs["T", frame],
                                                    geometric=False)
                # Photometric mask that contains only valid points
                outputs[("valid_mask", frame)] = valid_mask
                # Sample from source frame onto the projection flow
                outputs[("synthesized_frame", frame)] = F.grid_sample(inputs["source_frame", frame],
                                                                      pixel_coordinates,
                                                                      padding_mode=self.args.MODEL.padding_mode,
                                                                      align_corners=False)

                if self.args.DEBUG.plot and self.step == 0:
                    "Plots photometric error maps and saves them in the given location"
                    plt.imshow(inputs["target_frame"].detach().cpu().squeeze().permute(1,2,0))
                    plot_path = os.path.join(self.args.DEBUG.plot_path + "/tF_{}".format(self.step))
                    plt.axis('off')
                    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)

                if self.args.DEBUG.plot and self.step == 0:
                    "Plots photometric error maps and saves them in the given location"
                    plt.imshow(inputs["source_frame", frame].detach().cpu().squeeze().permute(1,2,0))
                    plot_path = os.path.join(self.args.DEBUG.plot_path + "/sF_{}".format(self.step))
                    plt.axis('off')
                    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)

                if self.args.DEBUG.plot and self.step % 10 == 0:
                    "Plots photometric error maps and saves them in the given location"
                    plt.imshow(outputs["synthesized_frame", frame].detach().cpu().squeeze().permute(1,2,0))
                    plot_path = os.path.join(self.args.DEBUG.plot_path  + "/synthF_{}".format(self.step))
                    plt.axis('off')
                    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)
                    plt.close("all")
        return outputs

    def compute_losses(self, inputs, outputs):
        losses = {}
        loss = 0

        self.optimizer.zero_grad()

        photmetric = self.compute_photometric_loss(inputs=inputs,
                                                   outputs=outputs)

        if self.args.LOSS.min_reprojection:
            # we decided to not use min reprojection because it was designed to tackle occlusion issues and in
            # indoor we don't face alot of those.
            photmetric = photmetric
        else:
            photmetric = photmetric.mean(1, keepdim=True)

        if self.args.DEBUG.plot and self.args.LOSS.min_reprojection and self.step % 10 == 0:
            "Plots photometric error maps and saves them in the given location"
            plt.imshow(torch.min(photmetric, dim=1)[0].detach().cpu().squeeze(), cmap="binary")
            plot_path = os.path.join(self.args.DEBUG.plot_path + "/pE_{}".format(self.step))
            plt.axis('off')
            plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)
        elif self.args.DEBUG.plot and self.step % 10 == 0:
            plt.imshow(photmetric[0].detach().cpu().squeeze(), cmap="binary")
            plot_path = os.path.join(self.args.DEBUG.plot_path + "/pE_{}".format(self.step))
            plt.axis('off')
            plt.savefig(plot_path, bbox_inches="tight", pad_inches=0, dpi=1200)

        if self.args.LOSS.auto_masking:
            # We decided to not use auto_masking technique because it was defined for outdoor scenes.
            auto_masking = self.compute_automasking_loss(inputs=inputs,
                                                         outputs=outputs)

            if self.args.LOSS.min_reprojection:
                auto_masking += torch.randn(auto_masking.shape).cuda() * 0.00001        # Break tie's
            else:
                auto_masking = auto_masking.mean(1, keepdim=True)

            photmetric = torch.cat((auto_masking, photmetric), dim=1)

        if photmetric.shape[1] == 1:
           optimize = photmetric
           optimize = optimize.mean()
        else:
            optimize, indexs = torch.min(photmetric, dim=1)
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
        for frame in range(len(self.args.DATA.frames)):    # TODO: Improve convention of how initial_depth and predicted depth are stored maybe.
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
            
    def hook_fn(self, layer, inputs, outputs):
        """INFO: Outputs of gradients w.r.t. to inputs and outputs are basically the same, therefore only one metric is
        used."""
        # tensorboard needs numpy array as input
        o = outputs[0].clone().detach().cpu().numpy()

        if self.args.VIZ.tensorboard_scaled:
            # scale the convolution layers
            scale_o = np.min(o) + np.max(o)

            if scale_o == 0.0:
                scale_o = 1.0

            o /= scale_o

        if self.args.MODEL.depth_network == "indoor":
            decoder_idx = 20
            image_idx = 33

            if self.hook_counter > decoder_idx:
                self.writer.add_histogram("Decoder_Layer_{}".format(self.hook_counter), o, self.refinement_step)
                print("Added histogram for convolution layer {}.".format(self.hook_counter))

                if self.hook_counter == image_idx:
                    self.writer.add_images("Image_Layer_{}_Step_{}_Outputs".format(self.hook_counter, self.refinement_step),
                                           o, dataformats='NCHW')
                    print("Added images for convolution layer {}.".format(self.hook_counter))

                self.hook_counter -= 1

            elif self.hook_counter > 0:
                self.writer.add_histogram("Encoder_Layer_{}".format(self.hook_counter), o, self.refinement_step)
                print("Added histogram for convolution layer {}.".format(self.hook_counter))
                self.hook_counter -= 1

            else:
                self.hook_counter = image_idx

        elif self.args.MODEL.depth_network == "monodepth2":
            image_idx = 10

            self.writer.add_histogram("Decoder_Layer_{}".format(self.hook_counter), o, self.refinement_step)
            print("Added histogram for convolution layer {}.".format(self.hook_counter))

            if self.hook_counter == image_idx:
                self.writer.add_images("Image_Layer_{}_Step_{}_Outputs".format(self.hook_counter, self.refinement_step), o, dataformats='NCHW')
                print("Added images for convolution layer {}.".format(self.hook_counter))

            if self.hook_counter > 0:
                self.hook_counter -= 1

            else:
                self.hook_counter = image_idx 


if __name__ == "__main__":
    args = arguments()
    config_path = args['config_path']
    config_dict = load_yaml(config_path)
    config_dict.SETTINGS.name = args['name']
    SLAM = Depth_Estimation(config_dict)
    SLAM.train()

    
