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
from kornia.geometry.linalg import inverse_transformation

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
from gradslam.slam.fusionutils import find_active_map_points
from gradslam.geometry.geometryutils import transform_pointcloud


class SLAM:
    def __init__(self, arguments):
        self.args = arguments
        self.device = torch.device("cuda" if self.args.SETTINGS.device == "cuda" else "cpu")
        self.sequence_length = self.args.DEMO.sequence_length
        self.color_map = plt.cm.get_cmap("magma").reversed()
        # self.writer = SummaryWriter()

        self.dataset_init()
        self.model_init()
        self.view_reconstruction_init()
        self.losses_init()

        if self.args.ABLATION.scale_intrinsics:
            print("Scaling Intrinsics")
        if self.args.ABLATION.scaled_depth:
            print("Scaling Depth Maps with median scaling")

        self.mean_abs = []

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

        if self.args.MODEL.slam == "ICPSLAM":
            self.models["SLAM"] = ICPSLAM(odom=self.args.MODEL.odom,
                                          numiters=self.args.MODEL.numiters,
                                          device=self.device)
            self.models["GT_SLAM"] = ICPSLAM(odom="gt", device=self.device)

        elif self.args.MODEL.slam == "PointFusion":
            self.models["SLAM"] = PointFusion(odom=self.args.MODEL.odom,
                                              dist_th=self.args.MODEL.dist_th,
                                              angle_th=self.args.MODEL.angle_th,
                                              sigma=self.args.MODEL.sigma,
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

        elif self.args.MODEL.depth_network == "indoor":
            self.models["depth"] = DispResNet_Indoor(num_layers=self.args.MODEL.num_layers,
                                                     pretrained=self.args.MODEL.weights_init_encoder == "imagenet")
            self.train_params += list(self.models["depth"].parameters())
            self.models["depth"].to(self.device)
        else:
            raise ValueError("Given {} is not a valid depth network option".format(self.args.MODEL.depth_network))

        if self.args.MODEL.use_pretrained_models:
            self.load_model_indoor()

        self.optimizer = define_optim(self.args, self.train_params)
        self.schedular = define_schedular(self.args, self.optimizer)

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

    def compute_frame_distance(self, prev, cur):
        """
        Compute frame_distance to filter the key-frames
        :param prev:  camera extrinsics of previous key-frame
        :param cur: camera extrinsics of current key-frame
        :return: distance between previous key-frame and current key-frame
        """

        # Extract Rotation and Translation
        prev_rot, prev_trans = prev[0, :3, :3], prev[0, :3, -1]
        cur_rot, cur_trans = cur[0, :3, :3], cur[0, :3, -1]

        # C=-R^(t)T  where t = transpose operation and T = Translation Vector
        prev_cam = -1 * torch.matmul(torch.transpose(prev_rot, 0, 1), prev_trans)
        cur_cam = -1 * torch.matmul(torch.transpose(cur_rot, 0, 1), cur_trans)

        # Compute Euclidean Norm
        distance_cam = torch.linalg.norm(prev_cam - cur_cam)

        return distance_cam

    def main(self):
        if self.args.MODEL.refinement_mode:
            self.set_refinement_mode()

        # Load a single batch from data loader, this will contain an entire sequence of images from TUM or ICL
        batch = next(iter(self.train_loader))

        colors, gt_depths, intrinsics, poses, = batch[0], batch[1], batch[2], batch[3]
        colors /= 255.0

        colors, gt_depths, intrinsics, poses = colors.to(self.device), \
                                               gt_depths.to(self.device), \
                                               intrinsics.to(self.device), \
                                               poses.to(self.device)

        # Initialize from 0th frame in the sequence
        prev_colors, prev_gt_depths, prev_poses = colors[:, 0, ...], gt_depths[:, 0, ...], poses[:, 0, ...]
        global_pointcloud = Pointclouds(device=self.device)
        self.first_iter = True

        # Here we iterate through the loaded sequence. Note that the sequence is predefined to keep in check GPU memory consumption
        for frame in range(1, self.sequence_length):
            cur_colors, cur_gt_depths, cur_poses = colors[:, frame, ...], gt_depths[:, frame, ...], poses[:, frame, ...]

            frame_distance = self.compute_frame_distance(prev_poses, cur_poses)

            # Check distance b/w two frames.
            if frame_distance > self.args.DEMO.frame_threshold:
                # If distance b/w two frame is higher than a set threshold then we do depth refinement !
                global_pointcloud = self.refinement(prev_colors, prev_gt_depths, prev_poses,
                                                   cur_colors, cur_gt_depths, cur_poses,
                                                   intrinsics, global_pointcloud)

                if self.first_iter:
                    self.first_iter = False

                print("Saved Sequence Frame {}".format(frame))

            else:
                continue

            # Make current_keyframe the previous one and continue.
            prev_colors, prev_gt_depths, prev_poses = cur_colors, cur_gt_depths, cur_poses

        # Plot the final global pointcloud
        global_pointcloud.plotly(0, point_size=2, max_num_points=50000).show()

        # Report the mean_abs
        if self.args.DEBUG.print_metrics:
            self.mean_abs = torch.Tensor(self.mean_abs)
            print(self.mean_abs.mean().item())

    def refinement(self,
                   prev_colors, prev_gt_depths, prev_poses,
                   cur_colors, cur_gt_depths, cur_poses,
                   intrinsics, global_pointcloud):

        # Combine prev and current into a sequence of 2 frames.
        colors, gt_depths, poses = torch.stack([prev_colors, cur_colors], 1), \
                                   torch.stack([prev_gt_depths, cur_gt_depths], 1), \
                                   torch.stack([prev_poses, cur_poses], 1)

        # Convert camera extrinsics to transformations
        transform = torch_poses_to_transforms(poses)

        self.initial_depths = {}

        for refine_step in range(self.args.OPTIMIZATION.refinement_steps):

            inputs = OrderedDict()
            depth_tensor = []
            scale = 0

            for index in range(len(self.args.DATA.frames)):
                inputs.update(self.models["depth"](colors[:, index, ...], index))
                inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]

                if self.args.OPTIMIZATION.refinement == "PFT" and self.args.LOSS.depth_regularizer and refine_step == 0:
                    self.initial_depths[("initial_depth", index, scale)] = inputs[("depth", index, scale)].clone().detach()

                depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))

                inputs[("gt_depth", index, scale)] = gt_depths[:, index, ...]

            # Median Scaling
            depth_tensor = torch.cat(depth_tensor, dim=1)
            depth_tensor = depth_tensor.permute(0, 1, 3, 4, 2)  # Change to channel last representation

            ratio = torch.median(gt_depths) / torch.median(depth_tensor)

            inputs[("depth", 0, scale)] *= ratio
            inputs[("depth", 1, scale)] *= ratio

            # Online Adaption
            total_loss = self.depth_refinement(colors, inputs, intrinsics, poses, transform, global_pointcloud)

            # Print some metrics for progress and debugging
            if self.args.DEBUG.print_metrics:
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_metrics(dataset=self.args.DATA.name,
                                                                            gt=gt_depths[0][1],
                                                                            pred=inputs[("depth", 1, 0)][0])
                print("Refine_Step:", refine_step,
                      "Total_Loss:", round(total_loss, 5),
                      "abs_rel: ", round(abs_rel.item(), 5),
                      "rmse: ", round(rmse.item(), 5),
                      "a1: ", round(a1.item(), 5))
                if refine_step == self.args.OPTIMIZATION.refinement_steps -1:
                    self.mean_abs.append(abs_rel.item())
            else:
                print("Iter:", iter,
                      "Refine_Step:", refine_step,
                      "Total_Loss:", round(total_loss, 5))

        # update the global pointcloud by adding the current key-frame after online adaption to it
        global_pointcloud = self.create_refined_pointcloud(colors,
                                                           gt_depths,
                                                           poses,
                                                           intrinsics,
                                                           global_pointcloud)

        return global_pointcloud

    def create_refined_pointcloud(self, colors, gt_depths, poses, intrinsics, global_pointcloud):
        inputs = OrderedDict()
        depth_tensor = []
        scale = 0

        # Depth estimation after refinement
        for index in range(len(self.args.DATA.frames)):
            inputs.update(self.models["depth"](colors[:, index, ...], index))
            inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]
            depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))
            inputs[("gt_depth", index, scale)] = gt_depths[:, index, ...]

        depth_tensor = torch.cat(depth_tensor, dim=1)  # use this for SLAM!
        depth_tensor = depth_tensor.permute(0, 1, 3, 4, 2)  # Change to channel last representation
        ratio = torch.median(gt_depths) / torch.median(depth_tensor)
        depth_tensor = depth_tensor * ratio

        # Store old frame
        prev_rgbd = RGBDImages(rgb_image=colors[:, 0, ...].unsqueeze(1),
                               depth_image=depth_tensor[:, 0, ...].unsqueeze(1),
                               poses=poses[:, 0,...].unsqueeze(1),
                               intrinsics=intrinsics)

        if self.first_iter:
            # For first iteration, we make pointcloud from previous frame
            global_pointcloud, _, = self.models["SLAM"].step(pointclouds=global_pointcloud,
                                                             live_frame=prev_rgbd, prev_frame=None)
        # Make final global pointcloud from the new key frame.
        live_rgbd = RGBDImages(rgb_image=colors[:, 1, ...].unsqueeze(1),
                               depth_image=depth_tensor[:, 1, ...].unsqueeze(1),
                               poses=poses[:, 1,...].unsqueeze(1),
                               intrinsics=intrinsics)
        # Take a SLAM step
        global_pointcloud, _, = self.models["SLAM"].step(pointclouds=global_pointcloud,     # This global pointcloud contains the old global pointcloud
                                                         live_frame=live_rgbd, prev_frame=prev_rgbd)

        
        return global_pointcloud    # This return is the updated global pointcloud


    def depth_refinement(self, colors, inputs, intrinsics, poses, transform, global_pointcloud):
        """
        Essentially Unsupervised Depth Estimation in refinement steps.
        outputs only contain the outputs from the view synthesis module, NOT initial data that is available from other sources
        """
        outputs = {}
        # Process inputs to make them PyTorch compatible in channel first representation
        inputs.update(self.process_inputs(colors, inputs, intrinsics, poses, transform, global_pointcloud))

        if not self.first_iter:
            # If it's not the first iteration then we compute the End-2-End Point Supervision
            outputs.update(self.pointcloud_computation(inputs))
        # Novel View Synthesis
        outputs.update(self.novel_view_synthesis(inputs))
        # Compute Losses
        loss = self.compute_losses(inputs, outputs)

        return loss

    def process_inputs(self, colors, inputs, intrinsics, poses, transform, global_pointcloud):
        """
        Create an inputs dictionary that contains all the relevant things needed for novel view synthesis and loss computations
        """

        inputs["source_frame", -1] = colors[:, 0, ...].permute(0, 3, 1, 2)
        inputs["target_frame"] = colors[:, 1, ...].permute(0, 3, 1, 2)
        inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
        inputs["Inverse_K"] = torch.pinverse(inputs["K"])

        inputs["source_depth", -1] = inputs["depth", 0, 0]
        inputs["target_depth"] = inputs["depth", 1, 0]

        inputs["source_disp", -1] = inputs["disp", 0, 0]
        inputs["target_disp"] = inputs["disp", 1, 0]

        inputs["T", -1] = transform[:, 1, ...]  # This represents the pose 0 -> -1
        inputs["global_pointcloud"] = global_pointcloud

        inputs["poses", -1] = poses[:, 0, ...]
        inputs["poses", 0] = poses[:, 1, ...]

        return inputs

    def novel_view_synthesis(self, inputs):
        """
        This function performs novel view synthesis as per monodepth2/SFM-Learner paper.
        """
        outputs = {}
        for frame in self.args.DATA.frames[1:]:
            # project the points into 3D
            camera_points = self.backproject_depth(inputs["target_depth"], inputs["Inverse_K"])

            if self.args.LOSS.geometric:
                # In the end we opt'd against using geometric loss
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
                # Get the projection flow and valid masking of view synthesis procedure
                pixel_coordinates, valid_mask = self.project_3d(points=camera_points,
                                                    K=inputs["K"],
                                                    T=inputs["T", frame].squeeze(1),
                                                    geometric=False)

                outputs[("valid_mask", frame)] = valid_mask
                # Synthesize the new frame by sampling from the source frame on top of the projection flow
                outputs[("synthesized_frame", frame)] = F.grid_sample(inputs["source_frame", frame],
                                                                      pixel_coordinates,
                                                                      padding_mode=self.args.MODEL.padding_mode,
                                                                      align_corners=False)

        return outputs

    def pointcloud_computation(self, inputs):
        """
        Compute a local pointcloud for the current keyframe
        """
        target_rgbd = RGBDImages(rgb_image=inputs["target_frame"].unsqueeze(1).permute(0, 1, 3, 4, 2),
                                 depth_image=inputs["target_depth"].unsqueeze(1).permute(0, 1, 3, 4, 2),
                                 poses=inputs["poses", 0].unsqueeze(1),
                                 intrinsics=inputs["K"].unsqueeze(1))

        pointcloud = Pointclouds(device=self.device)
        inputs["target_pc"], _, = self.models["SLAM"].step(pointclouds=pointcloud,
                                                             live_frame=target_rgbd,
                                                             prev_frame=None)

        return inputs

    def compute_losses(self, inputs, outputs):
        """
        Computer losses and back prop !
        """

        loss = 0

        self.optimizer.zero_grad()

        # Compute photometric loss
        photmetric = self.compute_photometric_loss(inputs=inputs,
                                                   outputs=outputs)
        # Akbar: we no longer use min reprojection as it mainly handles occlusions and has only been shown to be effective in outdoor scenes
        # it slowed down computation and is unnecessary if we only have one source frame.
        if self.args.LOSS.min_reprojection:
            photmetric = photmetric
        else:
            photmetric = photmetric.mean(1, keepdim=True)

        # The auto-masking technique was good for outdoors but we did not see any good effects indoor
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
           optimize = optimize.mean()
        else:
            optimize, indexs = torch.min(photmetric, dim=1)
            optimize = optimize.mean()

        loss += optimize

        # Geometric Consistency Loss
        if self.args.LOSS.geometric:
            geometric = self.compute_geometric_loss(outputs=outputs)
            geometric = geometric.mean()
            loss += geometric * self.args.LOSS.geometric_weight

        # Smoothness Regularizer
        if self.args.LOSS.smoothness:
            smooth_loss = self.compute_smoothness_loss(inputs=inputs)
            loss += smooth_loss * self.args.LOSS.smoothness_weight

        # Depth Regularizer
        if self.args.LOSS.depth_regularizer:
            depth_reg = self.compute_depth_regularizer(inputs=inputs)
            loss += depth_reg * self.args.LOSS.depth_regularizer_weight

        # Weak supervision
        if self.args.LOSS.supervise_depth:
            gt_loss = self.compute_gt_depth_loss(inputs=inputs)
            loss += gt_loss * self.args.LOSS.gt_depth_weight

        # End-2-End-Point Supervision Loss
        if self.args.LOSS.three3d_loss and not self.first_iter:
            threeD_loss = self.compute_3d_loss(inputs=inputs)
            loss += threeD_loss * self.args.LOSS.three3d_loss_weight

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
        """
        This function computes a loss function by comparing with a sparse GT depth (Weak supervision)
        """
        gt_loss = 0
        scale = 0
        for frame in range(len(self.args.DATA.frames)):
            gt_loss += depth_gt_loss(prediction=inputs[("depth", frame, scale)],
                                     sparse_groundtruth=inputs[("sparse_gt_depth", frame, scale)],
                                     sparse_mask=inputs[("sparse_mask", frame, scale)])

        return gt_loss

    def compute_3d_loss(self, inputs):
        """
        This function computes the End-2-End Point Supervision Loss
        """
        transformed_target_pc = transform_pointcloud(inputs["target_pc"].points_list[0], inputs["T", -1][0])
        loss, _ = knn_points_loss(inputs["global_pointcloud"].points_list[0].unsqueeze(0).detach(), transformed_target_pc.unsqueeze(0))

        return loss


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

if __name__ == "__main__":
    args = arguments()
    config_path = args['config_path']
    config_dict = load_yaml(config_path)
    config_dict.SETTINGS.name = args['name']
    slam = SLAM(config_dict)
    slam.main()