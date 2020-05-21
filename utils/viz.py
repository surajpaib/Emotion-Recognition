import torch
import wandb
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def visualize_filters(modules):
    
    conv_filters = []
    count = 1
    for m in modules:
        if isinstance(m, nn.Conv2d):
            filters = m.weight.detach().clone()
            filters = (filters - filters.min())/ filters.max()
            filter_grid = make_grid(filters)
            filter_grid = filter_grid.permute(1, 2, 0)

            first_filter_grid = filter_grid[:, :, 0]
         
            wandb.log({"Conv{} Filters".format(count): wandb.Image(first_filter_grid)})
            count += 1


def visualize_kernels(modules, cmap='jet'):
    
    count = 0
    fig = plt.figure(figsize=(100, 100))

    subplots = [m for m in modules if isinstance(m, nn.Conv2d)]
    for idx, m in enumerate(subplots):
        ax = fig.add_subplot(len(subplots), 1, idx + 1)

        filters = m.weight.detach().clone()
        filters = (filters - filters.min())/ filters.max()
        filter_grid = make_grid(filters)
        filter_grid = filter_grid.permute(1, 2, 0)

        first_filter_grid = filter_grid[:, :, 0]
        ax.set_axis_off()
        ax.set_title("Conv{} Filters".format(idx+1))
        ax.imshow(first_filter_grid, cmap=cmap)

    plt.show()


def visualize_feature_maps(modules):
    activation = {}
    
    def get_activation(name):
        print(name)
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    subplots = [m for m in modules if isinstance(m, nn.Conv2d)]
    for idx, m in enumerate(subplots):
        print(m)
        m.register_forward_hook(get_activation('conv{}'.format(idx+1)))

    return activation


def standardize_and_clip(tensor, min_value=0.0, max_value=1.0,
                         saturation=0.1, brightness=0.5):

    """Standardizes and clips input tensor.
    Standardizes the input tensor (mean = 0.0, std = 1.0). The color saturation
    and brightness are adjusted, before tensor values are clipped to min/max
    (default: 0.0/1.0).
    Args:
        tensor (torch.Tensor):
        min_value (float, optional, default=0.0)
        max_value (float, optional, default=1.0)
        saturation (float, optional, default=0.1)
        brightness (float, optional, default=0.5)
    Shape:
        Input: :math:`(C, H, W)`
        Output: Same as the input
    Return:
        torch.Tensor (torch.float32): Normalised tensor with values between
            [min_value, max_value]
    """

    tensor = tensor.detach().cpu()

    mean = tensor.mean()
    std = tensor.std()

    if std == 0:
        std += 1e-7

    standardized = tensor.sub(mean).div(std).mul(saturation)
    clipped = standardized.add(brightness).clamp(min_value, max_value)

    return clipped


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()

