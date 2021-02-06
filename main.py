from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import time
import os
import copy
from torchsummary import summary


from dataset_utils import *
from model_utils import *
from model import *


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

results_dir = model_name+'_results'
if not (os.path.exists(results_dir)):
    os.mkdir(results_dir)


# Number of classes in the dataset
num_classes = 339

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Learning rate for optimizers
lr = 0.01
target_lr = 0.001

num_workers = 4

# Number of epochs to train for
num_epochs = 50

ngpu = 2

split_list = ['train','val']

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

dataloaders, dataset_sizes, class_names = load_data(data_dir, split_list, batch_size, num_workers)


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

model = classification_model(model_name,num_classes=len(class_names),use_pretrained=True)
if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
model.to(device)
# summary(model, (3, 401, 501))

# BCE loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Print the model
print(model)

# Training Loop
model,train_metrics,val_metrics = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, lr=lr,target_lr=target_lr,num_epochs=num_epochs,device=device, results_dir=results_dir,is_inception=False)

visualize_model(model, device,dataloaders,class_names,num_images=6)
print(train_metrics)
print(val_metrics)