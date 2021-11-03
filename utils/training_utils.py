import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
from collections import OrderedDict
from typing import List, Union


def define_optim(args, parameters):
    """
    Defines the optimizers for training

    Inputs:
    args: (EasyDict) contains the given arguments
    parameter: (List) contains the parameters we want to optimize over

    Outputs:
    optimizer: returns an optimizer for training pipeline
    """

    if args.OPTIMIZATION.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=parameters,
                                     lr=args.OPTIMIZATION.learning_rate)

    elif args.OPTIMIZATION.optimizer == "SparseAdam":
        # In this variant, only moments that show up in the gradient get updated,
        # and only those portions of the gradient get applied to the parameters.
        optimizer = torch.optim.SparseAdam(params=parameters,
                                           lr=args.OPTIMIZATION.learning_rate)

    elif args.OPTIMIZATION.optimizer == "SGD":
        optimizer = torch.optim.SGD(params=parameters,
                                    lr=args.OPTIMIZATION.learning_rate,
                                    momentum=0.9,
                                    weight_decay=1e-3)

    elif args.OPTIMIZATION.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(params=parameters,
                                        lr=args.OPTIMIZATION.learning_rate)

    elif args.OPTIMIZATION.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(params=parameters,
                                        lr=args.OPTIMIZATION.learning_rate)

    else:
        raise ValueError("Define an optimizer")

    print("{} Optimizer Defined with initial LR = {}".format(args.OPTIMIZATION.optimizer,
                                                             args.OPTIMIZATION.learning_rate))

    return optimizer

def define_schedular(args, optimizer):

    """
    Defines the schedular to decay learning rate for training

    Inputs:
    args: (EasyDict) contains the given arguments
    optimizer: contains the optimizer for which we decay learning rate

    Outputs:
    schedular: returns an schedular for training pipeline
    """

    if args.OPTIMIZATION.schedular == "StepLR":
        schedular = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=args.OPTIMIZATION.schedular_step_size,
                                                    gamma=args.OPTIMIZATION.schedular_gamma)
        print("Learning Rate Decayed by StepLR")

    elif args.OPTIMIZATION.schedular == "MultiStepLR":
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=args.OPTIMIZATION.schedular_milestones,
                                                         gamma=args.OPTIMIZATION.schedular_gamma)
    elif args.OPTIMIZATION.schedular == "ExponentialLR":
        schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                           gamma=args.OPTIMIZATION.schedular_gamma)

    else:
        raise ValueError("decay_lr in config set to True but no schedular given")

    print("{} Schedular Defined with gamma = {} ".format(args.OPTIMIZATION.schedular,
                                                         args.OPTIMIZATION.schedular_gamma))

    return schedular

def set_train(models):
    """Convert all models to training mode
    """
    for m in models.values():
        m.train()

    return models

def set_eval(models):
    """Convert all models to testing/evaluation mode
    """
    for m in models.values():
        m.eval()

    return models

def convert_disp_to_depth(disp, min_depth, max_depth):
    """
    Convert depth model's sigmoid output into depth prediction


    """
    min_disp, max_disp = 1 / max_depth, 1 / min_depth
    "Scaled Disparity"
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    "Depth is just the inverse of disparity"
    depth = 1 / scaled_disp

    return depth

def scale_disp(disp, min_depth, max_depth):
    """
    Scale disparity
    """
    min_disp, max_disp = 1 / max_depth, 1 / min_depth
    "Scaled Disparity"
    scaled_disp = min_disp + (max_disp - min_disp) * disp

    return scaled_disp

def inverse_T_matrix(T):
    """
    Inverse a given transformation matrix.
    Akbar: I compared this inversion to numpy's np.linalg.inv and the error between them is very very small.
    I think its due to precision of the algorithm they use to compute the inverse, this should be fine.
    """
    # T[:, :3, -1] = -1 * T[:, :3, -1]         # This stores the reversed translation
    # T[:, :3, :3] = T[:, :3, :3].transpose(1, 2)
    T = torch.pinverse(T)

    return T

def scale_by_f(focal_data, focal_pretrain ,depth):
    """
    Scales depth map by the focal length
    Input: Depth image
    """

    scaling_factor = focal_data / focal_pretrain

    depth = depth * scaling_factor

    return depth

def normalize_intrinsics(args, K):
    """
    Monodepth2 normalizes the intrinsics of their pretrained depth model
    So we are doing the same for our intrinsic matrix.

    This function is called only for depth refinement section!
    """

    if args.DATA.name == "ICL":
        x_size = 640.0
        y_size = 480.0
    elif args.DATA.name == "TUM":
        x_size = 640.0
        y_size = 480.0
    else:
        raise ValueError("normalize intrinsics not supported for this dataset")

    K[:, 0, :] /= x_size
    K[:, 1, :] /= y_size

    return K

def sparse_sampling(sampling_type, prob, depth):

    if sampling_type == "random":
        # Create sparse random mask
        mask = torch.rand_like(depth)
        mask[mask >= prob] = 0.0
        mask[mask > 0.0] = 1.0
        mask[depth == 0.0] = 0.0
    else:
        raise ValueError("Sampling type not implemented")
    # Multiple mask by depth to create sparse GT depth
    masked_depth = depth * mask

    return masked_depth, mask

def torch_poses_to_transforms(poses: Union[torch.Tensor, List[torch.Tensor]]):
    r"""Converts poses to transformations w.r.t. the first frame in the sequence having identity pose

    Args:
        poses (torch.Tensor): Sequence of poses in `torch.Tensor` format.

    Returns:
        torch.Tensor or list of torch.Tensor: Sequence of frame to frame transformations where initial
            frame is transformed to have identity pose.

    Shape:
        - poses: Could be `torch.Tensor` of shape :math:`(N, 4, 4)`, or list of `torch.Tensor`s of shape
          :math:`(4, 4)`
        - Output: Of same shape as input `poses`
    """
    transformations = poses.detach().clone()
    batch_size = poses.shape[0]
    seq_len = poses.shape[1]
    for bs in range(batch_size):
        for sq in range(seq_len):
            if sq == 0:
                transformations[bs][sq] = torch.eye(4)
            else:
                transformations[bs][sq] = torch.pinverse(poses[bs][sq - 1]).matmul(poses[bs][sq])

    return transformations