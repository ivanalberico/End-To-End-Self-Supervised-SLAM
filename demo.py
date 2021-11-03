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
from gradslam import Pointclouds, RGBDImages


class Demo:
    def __init__(self, arguments):
        self.args = arguments
        self.device = torch.device("cuda" if self.args.SETTINGS.device == "cuda" else "cpu")
        self.sequence_length = self.args.DEMO.sequence_length
        self.sequence_length_refinement = self.args.DEMO.sequence_length_refinement
        self.color_map = plt.cm.get_cmap("magma").reversed()
        # self.writer = SummaryWriter()

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
        if self.args.MODEL.depth_network == "indoor":
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

    def demo(self):
        if self.args.MODEL.refinement_mode:
            self.set_refinement_mode()

        # Load a single batch from data loader, this will contain an entire sequence of images from TUM or ICL
        batch = next(iter(self.train_loader))
        colors, gt_depths, intrinsics, poses, transforms = batch[0], batch[1], batch[2], batch[3], batch[4]
        colors /= 255.0

        self.epoch = 0
        self.step = 0

        first_iter = True
        self.iter_seq = 0

        index_frame = 0

        while index_frame < self.sequence_length:

            prev_colors, prev_gt_depths, prev_poses = colors[:, index_frame, ...], gt_depths[:, index_frame, ...], poses[:, index_frame, ...]

            for frame in range((index_frame + 1), self.sequence_length):

                cur_colors, cur_gt_depths, cur_poses = colors[:, frame, ...], gt_depths[:, frame, ...], poses[:, frame, ...]
                frame_distance = self.compute_frame_distance(prev_poses, cur_poses)

                # If distance b/w two frame is higher than a set threshold then we do depth refinement !
                if frame_distance > self.args.DEMO.frame_threshold:
                    noisy_rgbd, inputs = self.depth_refinement(prev_colors,
                                                               prev_gt_depths,
                                                               prev_poses,
                                                               cur_colors,
                                                               cur_gt_depths,
                                                               cur_poses,
                                                               intrinsics)
                    self.iter_seq += 1

                    plt.imshow(inputs[("depth", 1, 0)][0].detach().cpu().squeeze(), cmap=self.color_map)
                    plt.axis('off')
                    plt.show()

                    ###########################   DEMO VISUALIZATION   #############################

                    pointclouds = Pointclouds(device=self.device)
                    batch_size, seq_len = noisy_rgbd.shape[:2]
                    initial_poses = torch.eye(4, device=self.device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
                    prev_frame = None

                    if first_iter:
                        intermediate_pcs = []
                        intermediate_poses = torch.Tensor([]).cuda()


                    poses_2frames = torch.cat((prev_poses, cur_poses), dim=0).cuda()
                    poses_2frames = torch.reshape(poses_2frames, (self.args.OPTIMIZATION.batch_size,
                                                                  poses_2frames.size()[0], poses_2frames.size()[1],
                                                                  poses_2frames.size()[2]))

                    for s in range(self.sequence_length_refinement):
                        live_frame = noisy_rgbd[:, s].to(self.device)
                        if s == 0 and live_frame.poses is None:
                            live_frame.poses = initial_poses

                        pointclouds, live_frame.poses = self.models["SLAM"].step(pointclouds, live_frame, prev_frame)
                        prev_frame = live_frame if self.models["SLAM"].odom != 'gt' else None

                        intermediate_pcs.append(pointclouds[0])


                    if first_iter:
                        pointclouds.plotly(0).show()

                    first_iter = False

                    intermediate_poses = torch.cat((intermediate_poses, poses_2frames[0]), dim=0)

                    ##########################################################################################

                    index_frame += 1
                    break       # if refinement takes place, we exit the for loop and we consider a new target frame

                index_frame += 1     # even though frame_distance is not higher than the threshold we still need to increase the index

            index_frame += 1   # this increment is performed only at the last step, to exit the while loop

        fig = plotly_map_update_visualization(intermediate_pcs, intermediate_poses, intrinsics[0, 0], 10000)



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



    def depth_refinement(self, prev_colors, prev_gt_depths, prev_poses, cur_colors, cur_gt_depths, cur_poses, intrinsics):

        scale = 0  # Only using 1 scale for now.   kinda redundent for now (remove usage if not needed)
        self.initial_depths = {}

        # The poses are with respect to the first frame of the sequence loaded from the dataset
        cur_poses = torch.matmul(inverse_T_matrix(prev_poses), cur_poses)   # computing the relative pose of the current frame wrt the previous frame
        prev_poses = torch.matmul(inverse_T_matrix(prev_poses), prev_poses)  # identity matrix

        for refine_step in range(self.args.OPTIMIZATION.refinement_steps):

            inputs = OrderedDict()  # Used for the depth part
            encoder_features = []
            depth_tensor = []  # Used for the SLAM part

            ##############  DEPTH ESTIMATION OF PREVIOUS FRAME ###############
            index = 0

            inputs.update(self.models["depth"](prev_colors.cuda(), index))
            inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]

            if self.args.ABLATION.scaled_depth:
                inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

            if self.args.OPTIMIZATION.refinement == "PFT" and self.args.LOSS.depth_regularizer and refine_step == 0:
                self.initial_depths[("initial_depth", index, scale)] = inputs[("depth", index, scale)].clone().detach()

            depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension
            inputs[("gt_depth", index, scale)] = prev_gt_depths

            ##############  DEPTH ESTIMATION OF CURRENT FRAME ###############
            index = 1

            inputs.update(self.models["depth"](cur_colors.cuda(), index))
            inputs[("depth", index, scale)] = 1 / inputs[("disp", index, scale)]

            if self.args.ABLATION.scaled_depth:
                inputs[("depth", index, scale)] *= self.args.ABLATION.scaling_depth

            if self.args.OPTIMIZATION.refinement == "PFT" and self.args.LOSS.depth_regularizer and refine_step == 0:
                self.initial_depths[("initial_depth", index, scale)] = inputs[("depth", index, scale)].clone().detach()

            depth_tensor.append(inputs[("depth", index, scale)].unsqueeze(1))  # Unsqueeze to create Sequence Dimension
            inputs[("gt_depth", index, scale)] = cur_gt_depths



            ########################################################################################

            del encoder_features  # Free Space

            depth_tensor = torch.cat(depth_tensor, dim=1).cuda()  # use this for SLAM!
            depth_tensor = depth_tensor.permute(0, 1, 3, 4, 2)  # Change to channel last representation

            poses = torch.cat((prev_poses, cur_poses), dim=0).cuda()
            poses = torch.reshape(poses, (self.args.OPTIMIZATION.batch_size, poses.size()[0], poses.size()[1], poses.size()[2]))

            colors = torch.cat((prev_colors, cur_colors), dim=0).cuda()
            colors = torch.reshape(colors, (self.args.OPTIMIZATION.batch_size, colors.size()[0], colors.size()[1], colors.size()[2], colors.size()[3]))

            gt_depths = torch.cat((prev_gt_depths, cur_gt_depths), dim=0).cuda()
            gt_depths = torch.reshape(gt_depths, (self.args.OPTIMIZATION.batch_size, gt_depths.size()[0], gt_depths.size()[1], gt_depths.size()[2], gt_depths.size()[3]))

            intrinsics = intrinsics.cuda()

            transforms = poses.cuda()     # when taking into account only two frames, poses and transforms are the same

            
            if self.args.DATA.use_gt_pose:
                new_poses = poses
            else:
                if self.step == 0:  # Initially the pose is taken as identity
                    new_poses = torch.eye(4, device=self.device).view(1, 1, 4, 4).repeat(self.args.OPTIMIZATION.batch_size,
                                                                                         poses.size()[1], 1, 1)

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
                new_transform = transforms
            else:
                noisy_reconstruction, new_poses = self.models["SLAM"](noisy_rgbd)
                new_transform = torch_poses_to_transforms(new_poses)


            noisy_pointcloud = noisy_reconstruction.points_list[0].unsqueeze(0).contiguous()  # Only optimizing points not color!
            inputs[("noisy_pointcloud")] = noisy_pointcloud


            "--- Refinement of Depth Maps ---"
            outputs = {}

            # total_loss = self.depth_refinement(colors, inputs, intrinsics, new_transform)
            inputs.update(self.process_inputs(colors, inputs, intrinsics, new_transform))
            outputs.update(self.novel_view_synthesis(inputs))
            total_loss = self.compute_losses(inputs, outputs)

            new_poses = new_poses.detach()  # TODO: Careful :: But it worked in GradSLAM??
            self.step += 1

            if self.args.DEBUG.print_metrics:
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = depth_metrics(dataset=self.args.DATA.name,
                                                                            gt=gt_depths[0][1],
                                                                            pred=inputs[("depth", 1, 0)][0])
                print("Iter:", self.iter_seq,
                      "Refine_Step:", refine_step,
                      "Total_Loss:", round(total_loss, 5),
                      "abs_rel: ", round(abs_rel.item(), 5),
                      "rmse: ", round(rmse.item(), 5),
                      "a1: ", round(a1.item(), 5))
            else:
                print("Iter:", self.iter_seq,
                      "Refine_Step:", refine_step,
                      "Total_Loss:", round(total_loss, 5))

            self.schedular.step()

        return noisy_rgbd, inputs

    def process_inputs(self, colors, inputs, intrinsics, poses):
        """
        Process the inputs suitable for view sythesis. Easier to understand for others then.
        NOTE: This function assumes sequence length of 2 or 3. For larger sequence lengths change appropriately.
        Making everything channel first representation too for PyTorch.
        """

        if self.sequence_length_refinement == 3:
            inputs["source_frame", -1] = colors[:, 0, ...].permute(0, 3, 1, 2)
            inputs["target_frame"] = colors[:, 1, ...].permute(0, 3, 1, 2)
            inputs["source_frame", 1] = colors[:, 2, ...].permute(0, 3, 1, 2)

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
                inputs[("sparse_gt_depth", index, scale)], inputs[("sparse_mask", index, scale)] = sparse_sampling(self.args.LOSS.sampling_type, self.args.LOSS.sampling_prob, inputs[("gt_depth", index, scale)])

        elif self.sequence_length_refinement == 2:
            if self.args.DATA.frames[1] < 0:
                inputs["source_frame", -1] = colors[:, 0, ...].permute(0, 3, 1, 2)
                inputs["target_frame"] = colors[:, 1, ...].permute(0, 3, 1, 2)

                inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                inputs["Inverse_K"] = torch.pinverse(inputs["K"])

                inputs["source_depth", -1] = inputs["depth", 0, 0]
                inputs["target_depth"] = inputs["depth", 1, 0]

                inputs["source_disp", -1] = inputs["disp", 0, 0]
                inputs["target_disp"] = inputs["disp", 1, 0]

                inputs["T", -1] = poses[:, 1, ...]  # This represents the pose 0 -> -1

                # GT Depths
                scale = 0
                for index in range(len(self.args.DATA.frames)):
                    inputs[("sparse_gt_depth", index, scale)], inputs[("sparse_mask", index, scale)] = sparse_sampling(self.args.LOSS.sampling_type, self.args.LOSS.sampling_prob, inputs[("gt_depth", index, scale)])


            elif self.args.DATA.frames[1] > 0:
                inputs["target_frame"] = colors[:, 0, ...].permute(0, 3, 1, 2)
                inputs["source_frame", 1] = colors[:, 1, ...].permute(0, 3, 1, 2)

                inputs["K"] = intrinsics[:, 0, ...]  # In a particular sequence, the intrinsic should be constant.
                inputs["Inverse_K"] = torch.pinverse(inputs["K"])

                inputs["target_depth"] = inputs["depth", 0, 0]
                inputs["source_depth", 1] = inputs["depth", 1, 0]

                inputs["target_disp"] = inputs["disp", 0, 0]
                inputs["source_disp", 1] = inputs["disp", 1, 0]

                inputs["T", 1] = inverse_T_matrix(poses[:, 1, ...])

                # GT Depths
                scale = 0
                for index in range(len(self.args.DATA.frames)):
                    inputs[("sparse_gt_depth", index, scale)], inputs[("sparse_mask", index, scale)] = sparse_sampling(self.args.LOSS.sampling_type, self.args.LOSS.sampling_prob, inputs[("gt_depth", index, scale)])

        else:
            raise ValueError("Sequence Length of 2 and 3 is only supported")

        return inputs



    def novel_view_synthesis(self, inputs):

        outputs = {}
        for frame in self.args.DATA.frames[1:]:
            camera_points = self.backproject_depth(inputs["target_depth"], inputs["Inverse_K"])

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


        return outputs



    def compute_losses(self, inputs, outputs):

        losses = {}
        loss = 0

        self.optimizer.zero_grad()

        photmetric = self.compute_photometric_loss(inputs=inputs,
                                                   outputs=outputs)

        if self.args.LOSS.min_reprojection:
            photmetric = photmetric
        else:
            photmetric = photmetric.mean(1, keepdim=True)


        if self.args.LOSS.auto_masking:
            auto_masking = self.compute_automasking_loss(inputs=inputs,
                                                         outputs=outputs)

            if self.args.LOSS.min_reprojection:
                auto_masking += torch.randn(auto_masking.shape).cuda() * 0.00001  # Break tie's
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
        for frame in range(len(self.args.DATA.frames)):  # TODO: Improve convention of how initial_depth and predicted depth are stored maybe.
            reg += depth_reguralizer(initial_depth=self.initial_depths[("initial_depth", frame, scale)],
                                     refined_depth=inputs[("depth", frame, scale)],
                                     loss_func=self.args.LOSS.depth_regularizer_type)

        return reg

    def compute_gt_depth_loss(self, inputs):
        gt_loss = 0
        scale = 0
        for frame in range(len(self.args.DATA.frames)):
            gt_loss += depth_gt_loss(prediction=inputs[("depth", frame, scale)].cuda(),
                                     sparse_groundtruth=inputs[("sparse_gt_depth", frame, scale)].cuda(),
                                     sparse_mask=inputs[("sparse_mask", frame, scale)].cuda())

        return gt_loss





if __name__ == "__main__":
    args = arguments()
    config_path = args['config_path']
    config_dict = load_yaml(config_path)
    config_dict.SETTINGS.name = args['name']
    SLAM = Demo(config_dict)
    SLAM.demo()
