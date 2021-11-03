import torch
import torch.nn as nn
import numpy as np



class BackprojectDepth(nn.Module):
    """Transform a single depth map into pointcloud for view-synthesis"""
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # Create a rectangular grid out of an array of x values and an array of y values.
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')

        # Stack the meshgrid's together
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # Convert into tensor
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        # Create ones
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)
        # Stack along dimension 0 and then unsqueeze along dimension 0.
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)  # .repeat -> repeats this tensor long the specified dimenson, this function copies the tensor's data.

        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        # Picture to world coordinate
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        # Multiply by depth to get the corresponding point cloud.
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps


    def forward(self, points, K, T, geometric):
        # T transforms the point cloud
        # K moves
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)

        pix_coords = pix_coords.permute(0, 2, 3, 1)

        # This is for grid sample to make it [-1, 1]
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        valid_points = pix_coords.abs().max(dim=-1)[0] <= 1
        valid_mask = valid_points.unsqueeze(1).float()

        if geometric:
            depth = cam_points[:, 2].clamp(min=1e-3)
            depth = depth.reshape(self.batch_size, 1, self.height, self.width)
            return pix_coords, depth, valid_mask
        else:
            return pix_coords, valid_mask