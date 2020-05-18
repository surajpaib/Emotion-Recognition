import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from utils.metrics import Metrics
from utils.utils import get_loss, save_checkpoint
from utils.viz import visualize_filters

from models.model import Model
from fer2013_dataset import FER2013Dataset

# Reproducibility Settings
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############################################################################
'''
Transforms on PIL image --

torchvision.transforms.RandomChoice(transforms)
torchvision.transforms.RandomApply(transforms, p=0.5)
torchvision.transforms.RandomOrder(transforms)


torchvision.transforms.RandomAffine(...)
torchvision.transforms.RandomHorizontalFlip(p=0.5)
torchvision.transforms.RandomRotation(...)
'''

image_data = FER2013Dataset("data/fer2013/fer2013/fer2013.csv", "Training").data.iloc[:,1].values

samples = []
for i in range(10):
    img = image_data[i].split(" ")
    img = np.array(img).reshape(48,48).astype(np.float32)
    samples.append(PIL.Image.fromarray(img))
#samples = np.array(samples)

# img_1 = PIL.Image.fromarray(samples[0])
# plt.imshow(img_1, cmap='gray')
# plt.show()

fig = plt.figure(figsize=(2., 5.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, samples):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, cmap='gray')
    ax.axis('off')

#plt.show()

# Apply augmentation ----------------------------------------------------------
transform = torchvision.transforms.RandomChoice([torchvision.transforms.RandomHorizontalFlip(p=1),
                                                 torchvision.transforms.RandomRotation(degrees=60)])

transformed_samples = []
for img in samples:
    tf_img = transform(img)
    transformed_samples.append(tf_img)

fig = plt.figure(figsize=(2., 5.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, transformed_samples):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, cmap='gray')
    ax.axis('off')

#plt.show()
