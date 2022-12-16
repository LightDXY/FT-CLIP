from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets
from pathlib import Path

from .labeled_memcached_dataset import McDataset

def build_imagenet_dataset(args, is_train, transform):
    if is_train:
        dataset = McDataset(args.data_path, 
            './data/ILSVRC2012_name_train.txt', transform=transform)
    else:
        dataset = McDataset(args.data_path, 
            './data/ILSVRC2012_name_val.txt', 'val', transform=transform)

    return dataset


