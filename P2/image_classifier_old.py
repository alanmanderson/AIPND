import numpy as np
import torch

from torch import nn
from torch import optim

from collections import OrderedDict

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

import json

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    "training" : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(100),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    "validation" : transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()]),
    "testing" : transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
}


# TODO: Load the datasets with ImageFolder
image_datasets = {
    "training" : datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
    "validation" : datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
    "testing" : datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    "training" : torch.utils.data.DataLoader(image_datasets["training"], batch_size=32, shuffle=True),
    "validation" : torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32),
    "testing" : torch.utils.data.DataLoader(image_datasets["testing"], batch_size=32)
}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

