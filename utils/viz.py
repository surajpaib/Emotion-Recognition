import torch
import wandb
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


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

            plt.imshow(first_filter_grid)
            plt.show() 
            wandb.log({"Conv{} Filters".format(count): wandb.Image(first_filter_grid)})
            count += 1



            