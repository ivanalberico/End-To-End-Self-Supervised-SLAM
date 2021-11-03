import torch
import torch.nn as nn
from chamferdist.chamfer import knn_points
import matplotlib.pyplot as plt

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
       source: monodepth2 style.
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def knn_points_loss(gt_pointcloud, noisy_pointcloud):
    """
    Measures the distance between the two point clouds in terms of points and colors!
    knn_points: https://github.com/krrish94/chamferdist/blob/255b7108a70f96d22d6a7e2258d03888fb3c8b89/chamferdist/chamfer.py#L200

    Inputs:
    gt_pointcloud: Points of the groundtruth pointcloud
    noisy_pointcloud: Points of the noisy pointcloud

    Outputs:
    knn_loss:

    """
    if gt_pointcloud.shape[0] != noisy_pointcloud.shape[0]:
        raise ValueError("Pointclouds must have the same batch dimension")
    if gt_pointcloud.shape[2] != noisy_pointcloud.shape[2]:
        raise ValueError("Number of axes is not the same in both pointclouds")

    KNN = knn_points(noisy_pointcloud, gt_pointcloud)
    distances = KNN.dists.squeeze(
        -1)  # Squaed Distances between the closest points in the noisy and groundtruth pointclouds
    indexes = KNN.idx.squeeze(-1).detach()  # Indices of noisy_pointclouds closest points to the groundtruth pointcloud
    knn_loss = torch.mean(distances)

    return knn_loss, indexes

def color_points_loss(gt_pointcloud_color, noisy_pointcloud_color, indexes):
    """
    Minimizes the L1 distance between the two color pointclouds.

     Inputs:
    gt_pointcloud_color: Color values of the groundtruth pointcloud
    noisy_pointcloud_color: Color values of the noisy pointcloud
    indexes

    Outputs:
    color_loss: L1 difference between the colored pointclouds

    """
    if gt_pointcloud_color.shape[2] != noisy_pointcloud_color.shape[2]:
        raise ValueError("Number of axes is not the same in both pointclouds")

    color_loss = torch.mean(torch.abs(noisy_pointcloud_color[0] - gt_pointcloud_color[0, indexes[0].long()]))
    return color_loss

def geometric_consistency_loss(outputs, frame, device):
    "This function returns the L1 geometric consistency loss between two depth maps"
    abs_diff = torch.div((outputs[("warped_depth", frame)] - outputs[("interpolated_depth", frame)]).abs(),
                         (outputs[("warped_depth", frame)] + outputs[("interpolated_depth", frame)])).clamp(0, 1)
    mask = outputs[("valid_mask", frame)].expand_as(abs_diff)

    if mask.sum() > 10000:
        mean_value = (abs_diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)

    return mean_value

def photometric_loss(ssim, prediction, target):
    """
    Computes reconstruction loss between the synthesized target and the original target frame

    Input:
    ssim: SSIM loss function initialized
    prediction: Synthesized Frame   (B, 3, H, W)
    target: Target frame  (B, 3, H, W)

    Output:
    rpj_loss: Scalar containing the photometric_loss

    """

    ssim_loss = ssim(x=prediction, y=target).mean(1, True)
    abs_diff = torch.abs(target - prediction)
    l1_loss = abs_diff.mean(1, True)

    rpj_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return rpj_loss

def disparity_smoothness_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()

def depth_reguralizer(initial_depth, refined_depth, loss_func):
    """
    Computes the depth regualizer terms, we hardcode for only the 0th scale right now!
    """

    if loss_func == "l1":
        loss = nn.L1Loss()
        reg = loss(initial_depth, refined_depth)
    elif loss_func == "l2":
        loss = nn.MSELoss()
        reg = loss(initial_depth, refined_depth)
    else:
        raise ValueError("please specify a correct norm")

    return reg


def depth_gt_loss(prediction, sparse_groundtruth, sparse_mask):
    """
    Computes the difference between the prediction from the self-supervised part with the corresponding groundtruth
    """
    loss_func = nn.L1Loss()
    # squeeze to remove depth channel...
    masked_prediction = prediction.squeeze() * sparse_mask.squeeze()
    loss = loss_func(masked_prediction, sparse_groundtruth.squeeze())

    return loss

@torch.no_grad()
def depth_metrics(dataset, gt, pred):
    """Evaluate errors for masked ground truth and predicted depth maps."""
    pred = pred.squeeze().detach()
    gt = gt.squeeze().detach()
    if dataset == "TUM":
        valid_mask = torch.ones_like(gt)
        valid_mask[gt == 0.0] = 0.0
    elif dataset == "ICL":
        valid_mask = torch.ones_like(gt)
    else:
        raise ValueError("Dataset Not Found")

    # only keep valid depth values in flattened tensor and compute errors
    valid_gt = gt[valid_mask.bool()]
    valid_pred = pred[valid_mask.bool()]

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(valid_gt, valid_pred)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

@torch.no_grad()
def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3