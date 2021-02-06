from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
from PIL import Image
import glob
import random
import torchvision.transforms.functional as TF


class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)


def load_data(data_dir, split_list, batch_size, num_workers):
    my_datasets = datasets.ImageFolder(data_dir)

    data_transforms_dict = {
        'train': transforms.Compose([
            transforms.Resize((401, 501)),
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((401, 501)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets_dict = {x: MyLazyDataset(my_datasets, data_transforms_dict[x])
                           for x in split_list}

    # Create the index splits for training, validation
    train_size = 0.75
    num_train = len(my_datasets)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))
    np.random.shuffle(indices)

    idx_dict = {'train': indices[:split], 'val': indices[split:]}

    data_dict = {x: Subset(image_datasets_dict[x], indices=idx_dict[x]) for x in split_list}

    dataloaders = {x: torch.utils.data.DataLoader(data_dict[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
                   for x in split_list}

    dataset_sizes = {x: len(data_dict[x]) for x in split_list}
    class_names = my_datasets.classes

    return dataloaders, dataset_sizes, class_names
