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
import cv2
from PIL import Image

from dataset_utils import *
from model_utils import *
from model import *
from utils import *


data_set = 'FRGC'

# data_dir =  '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_periocular_MTCNN_cropped_Left_Eye'
# dest_dir = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_periocular_MTCNN_cropped_Left_Eye_Features_UBIPr'


data_dir = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned'
dest_dir = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Fall2002_Images_copped_periocular_cleaned_Features_FRGC'
if not (os.path.exists(dest_dir)):
    os.mkdir(dest_dir)

# # Classification folder
# if data_set == 'FRGC':
#     data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Class_Folders"  # For FRGC dataset
# else:
#     data_dir = "/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split"
#
#
#
# # Attribute classification folders
# if data_set == 'FRGC':
#     image_folder = '/home/n-lab/Documents/Periocular_project/Datasets/FRGC_dataset_extracted_periocular_images/Class_Folders'
#     output_folder = '/home/n-lab/Documents/Periocular_Project_GitHub'
#     attribute_data_csv_loc = 'attribute_data_FRGC.csv'
# else:
#     image_folder='/home/n-lab/Documents/Periocular_project/Datasets/ubipr/Class_Folders_Left_Images_Split'
#     output_folder='/home/n-lab/Documents/Periocular_Project_GitHub'
#     attribute_data_csv_loc='attribute_data.csv'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
if data_set == 'FRGC':
    num_classes = 110
else:
    num_classes = 339  # UBiPr

# Batch size for training (change depending on how much memory you have)
classification = False

# Batch size for training (change depending on how much memory you have)
if not classification and data_set == 'FRGC':
    batch_size = 36
else:
    batch_size = 32

joint_optimization = True

# # Learning rate for optimizers
# lr = 0.01
# target_lr = 0.001
lr = 0.0007

num_workers = 4

# Number of epochs to train for
num_epochs = 100

ngpu = 1

split_list = ['train', 'val']

restart_training = True

ckpt_fname = 'epoch_90.pt'

balanced_batches = True

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


if data_set == 'FRGC':
    results_dir = '/home/n-lab/Documents/Periocular_Project_GitHub/FRGC_Joint_Optim_Results/vgg_results_100_epochs_0.0007_joint_model'
else:
    results_dir = '/home/n-lab/Documents/Periocular_Project_GitHub/Joint_Optim_Results/vgg_results_100_epochs_0.001_joint_model'


# returns JSON object as a dictionary
if data_set == 'FRGC':
    data = json.load(open('runtime_FRGC.json', 'r'))
else:
    data = json.load(open('runtime.json', 'r'))


if data_set == 'FRGC':
    model = joint_optimization_model(
        load_model_fname='/home/n-lab/Documents/Periocular_Project_GitHub/FRGC_Classification_Results/vgg_results_300_epochs_0.001/epoch_291.pt',
        load_model_attribute_name='/home/n-lab/Documents/Periocular_Project_GitHub/FRGC_Attribute_Classification_Results/vgg_results_100_epochs_0.0007/epoch_90.pt',
        model_name=model_name, num_classes_classification=num_classes, use_pretrained=True,
        head_config=data['head_configs'][0],data_set = data_set)
else:
    model = joint_optimization_model(
    load_model_fname='/home/n-lab/Documents/Periocular_Project_GitHub/vgg_results_70epochs/epoch_68.pth',
    load_model_attribute_name ='/home/n-lab/Documents/Periocular_Project_GitHub/Attribute_Classification_Results/vgg_results_80_epochs_0.0005_updated_model/epoch_77.pt',
    model_name=model_name, num_classes_classification=num_classes, use_pretrained=True,
    head_config=data['head_configs'][0])


# BCE loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

start_epoch = 1
if data_set == 'FRGC':
    train_metrics = {'loss': [], 'class_acc': [], 'class_acc_top_5': [], 'gender_acc': [],'ethnicity_acc' : []}
    val_metrics = {'loss': [], 'class_acc': [], 'class_acc_top_5': [], 'gender_acc': [],'ethnicity_acc' : []}
else:
    train_metrics = {'loss': [], 'class_acc': [], 'class_acc_top_5': [],'gender_acc':[]}
    val_metrics = {'loss': [], 'class_acc': [], 'class_acc_top_5': [],'gender_acc':[]}

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
    model, optimizer, start_epoch, train_metrics, val_metrics, exp_lr_scheduler = load_ckp_feature_extract(latest_ckpt, model,
                                                                                           optimizer, exp_lr_scheduler)

print(model)

image_files = os.listdir(data_dir)
model.eval()


if data_set == 'FRGC':
    image_transform = transforms.Compose([
        transforms.Resize((200, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    image_transform = transforms.Compose([
                    transforms.Resize((401, 501)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


with torch.no_grad():
    for image_file in image_files:
        image_path = os.path.join(data_dir,image_file)
        image_name = str(image_file).rstrip(".jpg")

        # Read the file
        img = Image.open(image_path)
        # Transform the image
        img = image_transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
        if data_set == 'FRGC':
            img = img.reshape(1, 3, 200, 150)
        else:
            img = img.reshape(1, 3, 401, 501)

        img = img.to(device)

        outputs, class_outputs, features = model(img)
        numpy_features = features.cpu().detach().numpy()
        np.save(os.path.join(dest_dir,image_name + '.npy'), numpy_features)



