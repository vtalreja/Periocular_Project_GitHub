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

# # Learning rate for optimizers
# lr = 0.01
# target_lr = 0.001
lr =0.001

num_workers = 4

# Number of epochs to train for
num_epochs = 180

ngpu = 2

split_list = ['train', 'val']

restart_training = False

ckpt_fname = None

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

results_dir = os.path.join('Classification_Results', model_name + '_results_' + str(150) + '_epochs')
if not (os.path.exists(results_dir)):
    os.mkdir(results_dir)

train_results_txt = os.path.join(results_dir, 'results_train_metrics.txt')
val_results_txt = os.path.join(results_dir, 'results_val_metrics.txt')


data_loader = data_utils(batch_size=batch_size, train_size=0.75, num_workers=num_workers, num_classes_sampler=2,
                         num_samples=16, balanced_batches=False)

dataloaders, dataset_sizes, class_names = data_loader.load_classification_data(data_dir, split_list)

# dataloaders, dataset_sizes, class_names = load_data(data_dir, split_list, batch_size, num_workers,False)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
if model_name == "vgg":
    """ VGG16_bn
    """
    model = classification_model(model_name, num_classes=len(class_names), use_pretrained=True)
elif model_name in ["resnet", "squeezenet"]:
    """ Resnet50 or squeezenet1_0
    """
    model = initialize_model(model_name, num_classes=len(class_names), use_pretrained=True)



# BCE loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

start_epoch = 1
train_metrics = {'loss': [], 'acc': []}
val_metrics = {'loss': [], 'acc': []}

if (device.type == 'cuda') and (ngpu > 1):
    model = nn.DataParallel(model, list(range(ngpu)))
model.to(device)

if (restart_training):
    if (ckpt_fname == None):
        list_of_ckpts = glob.glob(os.path.join(results_dir, '*.pt'))
        latest_ckpt = None if list_of_ckpts == [] else max(list_of_ckpts, key=os.path.getctime)
    else:
        latest_ckpt = os.path.join(results_dir, ckpt_fname)
    assert latest_ckpt != None, ' latest_ckpt cannot be None as None cannot be loaded in the model'
    model, optimizer, start_epoch, train_metrics, val_metrics,exp_lr_scheduler = load_ckp(latest_ckpt, model, optimizer,exp_lr_scheduler)
    print("Last epoch on saved checkpoint is {}. Training will start from {} for more {} iterations".format(
        start_epoch - 1, start_epoch, num_epochs - start_epoch - 1))

# summary(model, (3, 401, 501))


# Print the model
print(model)

# Training Loop
model, train_metrics, val_metrics = train_model(start_epoch=start_epoch, model=model, dataloaders=dataloaders,
                                                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                                                num_epochs=num_epochs, device=device, results_dir=results_dir,
                                                train_metrics=train_metrics, val_metrics=val_metrics,
                                                is_inception=False)

visualize_model(model, device, dataloaders, class_names, num_images=6)
print(train_metrics)
print(val_metrics)

for metric in ['loss', 'acc']:
    plot_metrics(train_metrics[metric], val_metrics[metric], results_dir=results_dir, metric=metric)

# write_results_csv(train_results_txt,train_metrics)
# write_results_csv(val_results_txt,val_metrics)
