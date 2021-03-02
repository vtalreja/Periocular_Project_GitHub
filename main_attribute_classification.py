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
from utils import *

# Classification folder
data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split"

# Attribute classification folders
image_folder='/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split'
output_folder='/home/n-lab/Documents/Periocular_Project_GitHub'
attribute_data_csv_loc='attribute_data.csv'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 339

# Batch size for training (change depending on how much memory you have)
batch_size = 32

classification = False

# # Learning rate for optimizers
# lr = 0.01
# target_lr = 0.001
lr = 0.0005

num_workers = 4

# Number of epochs to train for
num_epochs = 90

ngpu = 2

split_list = ['train', 'val']

restart_training = True

ckpt_fname = None

balanced_batches = True

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

results_dir = os.path.join('Attribute_Classification_Results', model_name + '_results_' + str(80) + '_epochs_' + str(lr)+'_updated_model')
if not (os.path.exists(results_dir)):
    os.mkdir(results_dir)

# returns JSON object as a dictionary
data = json.load(open('runtime.json', 'r'))

train_results_csv = os.path.join(results_dir, 'results_train_metrics.csv')
val_results_csv = os.path.join(results_dir, 'results_val_metrics.csv')

data_loader = data_utils(batch_size=batch_size, train_size=0.75, num_workers=num_workers, num_classes_sampler=2,
                         num_samples=16, balanced_batches=balanced_batches)
if classification:
    dataloaders, dataset_sizes, class_names = data_loader.load_classification_data(data_dir, split_list)
    data_dict = next(iter(dataloaders['train']))
    # new_data_dict = shuffle_dict(data_dict)
    classes = data_dict[1]
    # Make a grid from batch
    out = torchvision.utils.make_grid(data_dict[0])
    # title_list=[attribute_mapper.gender_id_to_name[x] for x in classes]
    imshow(out, title=[class_names[x] for x in classes])
    if model_name == "vgg":
        """ VGG16_bn
        """
        model = classification_model(model_name, num_classes=len(class_names), use_pretrained=True)
    elif model_name in ["resnet", "squeezenet"]:
        """ Resnet18 or squeezenet1_0
        """
        model = initialize_model(model_name, num_classes=len(class_names), use_pretrained=True)
else:
    dataloaders, dataset_sizes, attribute_mapper = data_loader.load_attribute_classification_data(
        image_folder=image_folder,output_folder=output_folder, attribute_data_csv_loc=attribute_data_csv_loc)

    # Get a batch of training data
    data_dict = next(iter(dataloaders['train']))
    new_data_dict = shuffle_dict(data_dict)
    classes = new_data_dict['gender_label']
    # Make a grid from batch
    out = torchvision.utils.make_grid(new_data_dict['img'])
    # title_list=[attribute_mapper.gender_id_to_name[x] for x in classes]
    imshow(out, title=[attribute_mapper.gender_labels[x] for x in classes])

    model = attribute_model(
        load_model_fname='/home/n-lab/Documents/Periocular_Project_GitHub/vgg_results_70epochs/epoch_68.pth',
        model_name=model_name, num_classes_classification=attribute_mapper.num_class_names, use_pretrained=True,
        head_config=data['head_configs'][0])


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
    model, optimizer, start_epoch, train_metrics, val_metrics, exp_lr_scheduler = load_ckp(latest_ckpt, model,
                                                                                           optimizer, exp_lr_scheduler)
    print("Last epoch on saved checkpoint is {}. Training will start from {} for more {} iterations".format(
        start_epoch - 1, start_epoch, num_epochs - (start_epoch - 1)))

# summary(model, (3, 401, 501))


# Print the model
print(model)

# Training Loop

if classification:
    model, train_metrics, val_metrics = train_model(start_epoch=start_epoch, model=model,
                                                    dataloaders=dataloaders,
                                                    criterion=criterion, optimizer=optimizer,
                                                    scheduler=exp_lr_scheduler,
                                                    num_epochs=num_epochs, device=device,
                                                    results_dir=results_dir,
                                                    train_metrics=train_metrics, val_metrics=val_metrics,
                                                    is_inception=False)
    visualize_model(model, device, dataloaders, class_names,
                    num_images=6)
else:
    model, train_metrics, val_metrics = train_model_attribute(start_epoch=start_epoch, model=model,
                                                              dataloaders=dataloaders,
                                                              criterion=criterion, optimizer=optimizer,
                                                              scheduler=exp_lr_scheduler,
                                                              num_epochs=num_epochs, device=device,
                                                              results_dir=results_dir,
                                                              train_metrics=train_metrics, val_metrics=val_metrics,
                                                              head_config=data['head_configs'][0],
                                                              is_inception=False)

    visualize_model_attribute(model, device, dataloaders, attribute_mapper.gender_labels, data['head_configs'][0],
                              num_images=6)

print(train_metrics)
print(val_metrics)

for metric in ['loss', 'acc']:
    plot_metrics(train_metrics[metric], val_metrics[metric], results_dir=results_dir, metric=metric)

write_results_csv(train_results_csv,train_metrics)
write_results_csv(val_results_csv,val_metrics)
