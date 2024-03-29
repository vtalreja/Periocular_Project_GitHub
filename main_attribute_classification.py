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

data_set = 'FRGC_S2004'

# Classification folder
# data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Class_Folders_No_Attributes" # For UBIRIS_V2 dataset
# data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split" # For UBIPr dataset
# data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Class_Folders"  # For FRGC dataset
data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images_MTCNN/Spring2004_cropped_Class_Folders" # FRGC Spring 2004

# Attribute classification folders
if data_set == 'FRGC':
    image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Class_Folders'
    output_folder = '/home/n-lab/Documents/Periocular_Project_GitHub'
    attribute_data_csv_loc = 'attribute_data_FRGC.csv'
elif data_set == 'UBIRIS_V2':
    image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/UBIris_v2/Class_Folders_No_Attributes'
    output_folder = '/home/n-lab/Documents/Periocular_Project_GitHub'
    attribute_data_csv_loc = 'attribute_data_UBIRIS_V2.csv'
elif data_set == 'FRGC_S2004':
    image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images_MTCNN/Spring2004_cropped_Class_Folders'
    output_folder = '/home/n-lab/Documents/Periocular_Project_GitHub'
    attribute_data_csv_loc = 'attribute_data_FRGC_Spring2004.csv'
else:
    image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split'
    output_folder = '/home/n-lab/Documents/Periocular_Project_GitHub'
    attribute_data_csv_loc = 'attribute_data.csv'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset

if data_set == 'FRGC':
    num_classes = 110
elif data_set == "UBIRIS_V2":
    num_classes = 514
elif data_set == 'FRGC_S2004':
    num_classes = 690
else:
    num_classes = 339  # UBiPr

classification = False

if not classification and 'FRGC' in data_set:
    batch_size = 36
else:
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

# # Learning rate for optimizers
lr = 0.0007

num_workers = 4

# Number of epochs to train for
num_epochs = 101

ngpu = 2

split_list = ['train', 'val']

restart_training = False

ckpt_fname = None

balanced_batches = True

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# results_dir = os.path.join('Attribute_Classification_Results', model_name + '_results_' + str(80) + '_epochs_' + str(lr)+'_updated_model') # UbiPr
results_dir = os.path.join('FRGC_S2004_Attribute_Classification_Results',
                           model_name + '_results_' + str(num_epochs) + '_epochs_' + str(lr))

if not (os.path.exists(results_dir)):
    os.mkdir(results_dir)

# returns JSON object as a dictionary
if 'FRGC' in data_set:
    data = json.load(open('runtime_FRGC.json', 'r'))
else:
    data = json.load(open('runtime.json', 'r'))

train_results_csv = os.path.join(results_dir, 'results_train_metrics.csv')
val_results_csv = os.path.join(results_dir, 'results_val_metrics.csv')

if 'FRGC' in data_set and classification:
    data_loader = data_utils(batch_size=batch_size, train_size=0.80, num_workers=num_workers, num_classes_sampler=8,
                             num_samples=4, balanced_batches=balanced_batches, data_set=data_set)
elif 'FRGC' in data_set and not classification:
    data_loader = data_utils(batch_size=batch_size, train_size=0.80, num_workers=num_workers, num_classes_sampler=3,
                             num_samples=12, balanced_batches=balanced_batches, data_set=data_set)
elif data_set == 'UBIRIS_V2' and classification:
    data_loader = data_utils(batch_size=batch_size, train_size=0.75, num_workers=num_workers, num_classes_sampler=8,
                             num_samples=4, balanced_batches=balanced_batches, data_set=data_set)
else:
    data_loader = data_utils(batch_size=batch_size, train_size=0.75, num_workers=num_workers, num_classes_sampler=2,
                             num_samples=16, balanced_batches=balanced_batches, data_set=data_set)
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
        model = classification_model(model_name, num_classes=len(class_names), use_pretrained=True, data_set=data_set)
    elif model_name in ["resnet", "squeezenet"]:
        """ Resnet18 or squeezenet1_0
        """
        model = initialize_model(model_name, num_classes=len(class_names), use_pretrained=True)
else:
    dataloaders, dataset_sizes, attribute_mapper = data_loader.load_attribute_classification_data(
        image_folder=image_folder, output_folder=output_folder, attribute_data_csv_loc=attribute_data_csv_loc)

    # Get a batch of training data
    data_dict = next(iter(dataloaders['train']))
    new_data_dict = shuffle_dict(data_dict,data_set = data_set)
    # Make a grid from batch
    out = torchvision.utils.make_grid(new_data_dict['img'])
    # title_list=[attribute_mapper.gender_id_to_name[x] for x in classes]
    if 'FRGC' in data_set:
        classes = new_data_dict['ethnicity_label']
        imshow(out, title=[attribute_mapper.ethnicity_labels[x] for x in classes])
    else:
        classes = new_data_dict['gender_label']
        imshow(out, title=[attribute_mapper.gender_labels[x] for x in classes])

    if data_set == 'FRGC':
        model = attribute_model(
            load_model_fname='/home/n-lab/Documents/Periocular_Project_GitHub/FRGC_Classification_Results/vgg_results_300_epochs_0.001/epoch_291.pt',
            model_name=model_name, num_classes_classification=attribute_mapper.num_class_names, use_pretrained=True,
            head_config=data['head_configs'][0],data_set = data_set)
    elif data_set == 'FRGC_S2004':
        model = attribute_model(
            load_model_fname='/home/n-lab/Documents/Periocular_Project_GitHub/FRGC_S2004_Classification_Results/vgg_results_350_epochs_0.001/epoch_346.pt',
            model_name=model_name, num_classes_classification=attribute_mapper.num_class_names, use_pretrained=True,
            head_config=data['head_configs'][0],data_set = data_set)
    elif data_set == 'UBIRIS_V2':
        model = attribute_model(
            load_model_fname='/home/n-lab/Documents/Periocular_Project_GitHub/UBRIS_V2_Classification_Results/vgg_results_350_epochs_0.001/epoch_350.pt',
            model_name=model_name, num_classes_classification=attribute_mapper.num_class_names, use_pretrained=True,
            head_config=data['head_configs'][0],data_set = data_set)
    else:
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
if 'FRGC' in data_set:
    train_metrics = {'loss': [], 'acc': [],'gender_acc' : [], 'ethnicity_acc':[]}
    val_metrics = {'loss': [], 'acc': [], 'gender_acc' : [], 'ethnicity_acc':[]}
else:
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
                                                              is_inception=False,data_set=data_set)

    visualize_model_attribute(model, device, dataloaders, attribute_mapper.gender_labels, data['head_configs'][0],
                              num_images=6,data_set=data_set)

print(train_metrics)
print(val_metrics)

if 'FRGC' in data_set and not classification:
    for metric in ['loss', 'acc','gender_acc','ethnicity_acc']:
        plot_metrics(train_metrics[metric], val_metrics[metric], results_dir=results_dir, metric=metric)
else:
    for metric in ['loss', 'acc']:
        plot_metrics(train_metrics[metric], val_metrics[metric], results_dir=results_dir, metric=metric)

# write_results_csv(train_results_csv,train_metrics)
# write_results_csv(val_results_csv,val_metrics)
