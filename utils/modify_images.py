import torch

def noise_depth(image: torch.Tensor, device: torch.device, std: torch.Tensor, mean:torch.Tensor ) -> torch.Tensor:
    """
    Returns an images with gaussian noise

    Inputs:
    image (torch.Tensor): contains RGB image or Depth image of size [B,L,H,W,3] or [B,L,H,W,1]
    device (torch.device): cuda or cpu
    std: std of the depth image
    mean: mean of the depth image

    Outputs:
    noisy_image (torch.Tensor): Depth image perturbed with noise based on its mean and std.. return size [B,L,H,W,3] or [B,L,H,W,1]
    """

    """
    mu = self.args.DEPTH_RECOVER.gx_sigma
        std = self.args.DEPTH_RECOVER.gx_mean

        noisy_depths[:, -1, ...] = (torch.rand(1, self.args.DATA.height, self.args.DATA.width, 1).to(self.device) * std) + mu
        noisy_colors[:, -1, ...] = torch.rand(1, self.args.DATA.height, self.args.DATA.width, 3).to(self.device)

        #Turn requires_grad on for colors/depths
        noisy_depths.requires_grad_().retain_grad()
        noisy_colors.requires_grad_().retain_grad()

    """

    if not torch.is_tensor(image):
        raise TypeError(
            "Expected input rgb/depth to be of type torch.Tensor. Got {0} instead.".format(type(image)))

    if image.shape[4] != 1:
        raise ValueError("Expected image.shape[4] to have 3 or 1 channels. Got {0}.".format(image.shape[4]))

    if not torch.is_tensor(std) or not torch.is_tensor(mean):
        raise ValueError("Expected sigma and mean to be type 0D tensor. Got {0} and {0}".format(type(std),type(mean)))

    height = image.shape[2]
    width = image.shape[3]

    image[:, -1, ...] = (torch.rand(1, height, width, 1).to(device) * std) + mean
    return image

def noise_color(image: torch.Tensor, device) -> torch.Tensor:
    """
    Replace a color image by white noise in the last position of the given sequence

    Inputs:
    image (torch.Tensor)
    device(torch.device)

    Outputs:
    image (torch.Tensor)
    """
    if not torch.is_tensor(image):
        raise TypeError(
            "Expected input to be of type torch.Tensor. Got {0} instead.".format(type(input)))
    if image.shape[4] != 3:
        raise ValueError("Expected image.shape[4] to have 3.. Got {0}.".format(image.shape[4]))

    height = image.shape[2]
    width = image.shape[3]

    image[:, -1, ...] = torch.rand(1, height, width, 3).to(device)

    return image


def remove_pixels(image: torch.Tensor, device, mask_height: int, mask_width: int) -> torch.Tensor:
    """ returns the input image with pixels removed at specificed location ( center by default )

    Input:
    input (torch.Tensor): an RGB or depth image
    device(torch.device): "cuda" or "cpu"
    mask_height (int): specifies height of mask
    mask_width (int): specifies width of mask

    Output:
    input (torch.Tensor) an RGB or depth image with pixels removed from specificed location (center by default)
    """
    if not torch.is_tensor(image):
        raise TypeError("Expected input to be of type torch.Tensor. Got {0} instead.".format(type(image)))

    if image.shape[4] != 3 and image.shape[4] != 1:
        raise ValueError("Expected image.shape[4] to have 3 or 1 channels. Got {0}.".format(image.shape[4]))

    if not isinstance(mask_height, int) or not isinstance(mask_width, int):
        raise ValueError("Expected mask height and mask width to be type int. Got {0} and {0}". format(type(mask_width),
                                                                                                       type(mask_height)))

    # Creates a mask of the specified width and height
    # Check if mask is valid height and width
    if not (0 <= mask_height < image.shape[2] and
            0 <= mask_width < image.shape[3]):
        raise ValueError(
            " mask height {} and mask width {} should be smaller than input height {} and input width {}".format(
                mask_height,
                mask_width,
                image.shape[2],
                image.shape[3]))

    mask = torch.ones([1,
                       1,
                       mask_height,
                       mask_width,
                       image.shape[4]]).to(device)

    # Select the center of the input image and mask it with ones! #TODO: Find cleaner way to do this
    # These indices determine where you want to place the mask.
    start_height = image.shape[2] // 2 - mask_height // 2
    end_height = image.shape[2] // 2 + mask_height // 2

    start_width = image.shape[3] // 2 - mask_width // 2
    end_width = image.shape[3] // 2 + mask_width // 2

    # Check if mask is within the given input image.
    if not (0 <= start_height < image.shape[2] - mask_height // 2 and
            0 + mask_height // 2 <= end_height < image.shape[2] and
            0 <= start_width < image.shape[3] - mask_width // 2 and
            0 + mask_height // 2 <= end_height < image.shape[2]):
        raise ValueError(
            "Mask out of bounds start_height {}, start_width {}, end_height {}, end_width {}".format(start_height,
                                                                                                     start_width,
                                                                                                     end_height,
                                                                                                     end_width))
    # Replace pixels with mask
    image[:, -1, start_height:end_height, start_width:end_width, :] = mask

    return image

def replace_image(image: torch.Tensor, device) -> torch.Tensor:
    """
    Replaces the entire input by a constant value of 1.

    Inputs:
    input (torch.Tensor)
    device: (torch.device) "cuda" or "cpu"

    Outputs:
    constant (torch.Tensor)
    """
    if not torch.is_tensor(image):
        raise TypeError(
            "Expected input to be of type torch.Tensor. Got {0} instead.".format(type(input)))
    if image.shape[4] != 3 and image.shape[4] != 1:
        raise ValueError("Expected image.shape[4] to have 3 or 1 channels. Got {0}.".format(image.shape[4]))

    image[:, -1, ...] = 1.0

    return image

def corrupt_rgbd(args, device,noisy_colors, noisy_depths):
    """
    Modify the 4th RGB/Depth pair according to given flags in the config

    Inputs:
    colors (torch.Tensor): contains RGB image of size [B,L,H,W,3]
    depths (torch.Tensor): contains depth image of size [B,L,H,W,1]

    Outputs:
    noisy_image (torch.Tensor): Modified color image [B,L,H,W,3]
    noisy_depths (torch.Tensor): Modified depth image [B,L,H,W,1]

    """

    # Add Gaussian Noise
    if args.DEPTH_RECOVER.noise_color:
        if args.DEPTH_RECOVER.optimize_color:
            print("Adding White Noise to color image")
            noisy_colors = noise_color(image=noisy_colors,
                                       device=device)
        else:
            raise ValueError("Set the optimize_color flag in config to optimize noisy color image")


    if args.DEPTH_RECOVER.noise_depth:
        if args.DEPTH_RECOVER.optimize_depth:
            print("Adding Gaussian Noise to depth image")
            mean = torch.mean(noisy_depths)
            std = torch.std(noisy_depths)
            noisy_depths = noise_depth(image=noisy_depths,
                                          device=device,
                                          std=std,
                                          mean=mean)
        else:
            raise ValueError("Set the optimize_depth flag in config to optimize noisy depth image")

    # Remove Pixels
    if args.DEPTH_RECOVER.remove_pixels_color:
        if args.DEPTH_RECOVER.optimize_color:
            print("Masking pixels from middle of the color frame")
            noisy_colors = remove_pixels(image=noisy_colors,
                                         device=device,
                                         mask_height=args.DEPTH_RECOVER.mask_height,
                                         mask_width=args.DEPTH_RECOVER.mask_width)
        else:
            raise ValueError("Set the optimize_color flag in config to optimize noisy color image")

    if args.DEPTH_RECOVER.remove_pixels_depth:
        if args.DEPTH_RECOVER.optimize_depth:
            print("Masking pixels from middle of the depth frame")
            noisy_depths = remove_pixels(image=noisy_depths,
                                         device=device,
                                         mask_height=args.DEPTH_RECOVER.mask_height,
                                         mask_width=args.DEPTH_RECOVER.mask_width)

        else:
            raise ValueError("Set the optimize_depth flag in config to optimize noisy depth image")

    if args.DEPTH_RECOVER.replace_color:
        if args.DEPTH_RECOVER.optimize_color:
            print("Replacing color by Constant")
            noisy_colors = replace_image(image=noisy_colors,
                                         device=device)
        else:
            raise ValueError("Set optimize_rgb in args to optimize the constant else set replace_rgb off")

    if args.DEPTH_RECOVER.replace_depth:
        if args.DEPTH_RECOVER.optimize_depth:
            print("Replacing depth by Constant")
            noisy_depths = replace_image(image=noisy_depths,
                                         device=device)
        else:
            raise ValueError("Set the optimize_depth flag in config to optimize noisy depth image")

    if args.DEPTH_RECOVER.optimize_color:
        noisy_colors.requires_grad_().retain_grad()
    if args.DEPTH_RECOVER.optimize_depth:
        noisy_depths.requires_grad_().retain_grad()

    return noisy_colors, noisy_depths


