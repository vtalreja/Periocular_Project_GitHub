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
import matplotlib.pyplot as plt



from dataset_utils import *
from model_utils import *
from model import *


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"


# Number of classes in the dataset
num_classes = 339

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Learning rate for optimizers
lr = 0.01
target_lr = 0.001

num_workers = 4

# Number of epochs to train for
num_epochs = 80

ngpu = 2

split_list = ['train','val']

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

results_dir = os.path.join('Classification_Results',model_name+'_results_' + str(num_epochs)+ '_epochs')
if not (os.path.exists(results_dir)):
    os.mkdir(results_dir)

dataloaders, dataset_sizes, class_names = load_data(data_dir, split_list, batch_size, num_workers)


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

if model_name == "vgg":
    """ VGG16_bn
    """
    model = classification_model(model_name,num_classes=len(class_names),use_pretrained=True)
elif model_name in ["resnet","squeezenet"]:
    """ Resnet50 or squeezenet1_0
    """
    model = initialize_model(model_name,num_classes=len(class_names),use_pretrained=True)

if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
model.to(device)
# summary(model, (3, 401, 501))

# BCE loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Print the model
print(model)

# Training Loop
model,train_metrics,val_metrics = train_model(model=model, dataloaders=dataloaders, criterion=criterion, optimizer=optimizer, lr=lr,target_lr=target_lr,num_epochs=num_epochs,device=device, results_dir=results_dir,is_inception=False)

visualize_model(model, device,dataloaders,class_names,num_images=6)
print(train_metrics)
print(val_metrics)

for metric in ['loss','acc']:
    plot_metrics(train_metrics[metric],val_metrics[metric],results_dir=results_dir,metric=metric)