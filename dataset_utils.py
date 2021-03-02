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
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms, utils
from PIL import Image
import glob
import random
import torchvision.transforms.functional as TF
import csv
from tqdm import tqdm

from utils import *


class Attribute_Mapper():
    def __init__(self, attribute_csv):
        self.attrbute_csv = attribute_csv
        class_names_list = []
        gender_list = []
        with open(self.attrbute_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names_list.append(row['class_name_label'])
                gender_list.append(row['gender_label'])

        self.class_name_labels = np.unique(class_names_list)
        self.gender_labels = np.unique(gender_list)

        self.num_class_names = len(self.class_name_labels)
        self.num_genders = len(self.gender_labels)

        self.class_id_to_name = dict(zip(range(self.num_class_names), self.class_name_labels))
        self.class_name_to_id = dict(zip(self.class_name_labels, range(self.num_class_names)))

        # Assigning numeric labels to the categories. For example : Male is given a label of 0, Female is given a label of 1
        self.gender_id_to_name = dict(zip(range(self.num_genders), self.gender_labels))
        self.gender_name_to_id = dict(zip(self.gender_labels, range(self.num_genders)))


class Attribute_Dataset(Dataset):
    def __init__(self, data_csv, mapper, transform=None):
        self.transform = transform
        self.mapper = mapper

        self.data = []
        self.class_names = []
        self.genders = []

        with open(data_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.class_names.append(self.mapper.class_name_to_id[row['class_name_label']])
                self.genders.append(self.mapper.gender_name_to_id[row['gender_label']])

    def __getitem__(self, index):
        image_path = self.data[index]

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        data_dict = {'img': img,
                     'gender_label': self.genders[index],
                     'class_name_label': self.class_names[index]}

        return data_dict

    def __len__(self):
        return len(self.data)


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


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, attribute):
        loader = DataLoader(dataset)
        self.labels_list = []
        for data_dict in loader:
            label = data_dict[attribute]
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


class data_utils():
    def __init__(self, batch_size, train_size, num_workers, num_classes_sampler=None, num_samples=None,
                 balanced_batches=False):
        if balanced_batches:
            assert num_classes_sampler != None, " if balanced batches then num_classes_sampler cannot be None"
            assert num_samples != None, " if balanced batches then num_samples cannot be None"
        self.batch_size = batch_size
        self.balanced = balanced_batches
        self.train_size = train_size
        self.data_transforms_dict = {
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
        self.num_classes_sampler = num_classes_sampler
        self.num_samples = num_samples
        self.num_workers = num_workers

    def load_classification_data(self, image_dir, split_list):
        my_datasets = datasets.ImageFolder(image_dir)

        image_datasets_dict = {x: MyLazyDataset(my_datasets, self.data_transforms_dict[x])
                               for x in split_list}

        # Create the index splits for training, validation
        num_train = len(my_datasets)

        indices, split = self.split_size(num_train)
        # indices = list(range(num_train))
        # split = int(np.floor(self.train_size * num_train))
        # np.random.shuffle(indices)

        idx_dict = {'train': indices[:split], 'val': indices[split:]}

        data_dict = {x: Subset(image_datasets_dict[x], indices=idx_dict[x]) for x in split_list}

        dataset_sizes = {x: len(data_dict[x]) for x in split_list}
        class_names = my_datasets.classes

        if self.balanced:
            balanced_batch_sampler = BalancedBatchSampler(data_dict['train'], self.num_classes_sampler,
                                                          self.num_samples,
                                                          'gender_label')

            dataloaders = {
                'train': torch.utils.data.DataLoader(data_dict['train'], batch_sampler=balanced_batch_sampler,
                                                     num_workers=self.num_workers),
                'val': torch.utils.data.DataLoader(data_dict['val'], batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers)}
        else:

            dataloaders = {x: torch.utils.data.DataLoader(data_dict[x], batch_size=self.batch_size,
                                                          shuffle=True, num_workers=self.num_workers)
                           for x in split_list}

        return dataloaders, dataset_sizes, class_names

    def load_attribute_classification_data(self, image_folder, output_folder, attribute_data_csv_loc):
        train_csv_loc, val_csv_loc = self.split_data(image_folder=image_folder, output_folder=output_folder,
                                                     attribute_data_csv=attribute_data_csv_loc)

        attribute_mapper = Attribute_Mapper(attribute_csv=attribute_data_csv_loc)

        train_dataset = Attribute_Dataset(train_csv_loc, attribute_mapper, transform=self.data_transforms_dict['train'])
        val_dataset = Attribute_Dataset(val_csv_loc, attribute_mapper, transform=self.data_transforms_dict['val'])

        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        if self.balanced:
            balanced_batch_sampler = BalancedBatchSampler(train_dataset, self.num_classes_sampler, self.num_samples,
                                                          'gender_label')

            dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_sampler=balanced_batch_sampler,
                                                                num_workers=self.num_workers),
                           'val': torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                                                              num_workers=self.num_workers)}
        else:

            dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                                shuffle=True, num_workers=self.num_workers),
                           'val': torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                                                              num_workers=self.num_workers)
                           }
        return dataloaders,dataset_sizes,attribute_mapper
    def split_data(self, image_folder, output_folder, attribute_data_csv):
        # input_folder = "/home/n-lab/Documents/Multi_Label_Classification/Fashion_product_Images"
        # output_folder = "/home/n-lab/Documents/Multi_Label_Classification/Fashion_product_Images"
        annotation = attribute_data_csv

        # open annotation file
        all_data = []
        with open(annotation) as csv_file:
            # parse it as CSV
            reader = csv.DictReader(csv_file)
            # tqdm shows pretty progress bar
            # each row in the CSV file corresponds to the image
            for row in tqdm(reader, total=reader.line_num):
                # we need image ID to build the path to the image file
                img_id = row['id']
                # we're going to use only 3 attributes
                gender = row['gender_label']
                class_name = row['class_name_label']
                img_name = os.path.join(image_folder, class_name, str(img_id))
                # check if file is in place
                # if os.path.exists(img_name):
                    # # check if the image has 80*60 pixels with 3 channels
                    # img = Image.open(img_name)
                    # if img.size == (60, 80) and img.mode == "RGB":
                all_data.append([img_name, class_name, gender])

        # set the seed of the random numbers generator, so we can reproduce the results later
        # np.random.seed(42)

        # construct a Numpy array from the list
        all_data = np.asarray(all_data)
        data_set_size = all_data.shape[0]

        # split the data into train/val and save them as csv files
        indices, split = self.split_size(data_set_size)
        train_csv = os.path.join(output_folder, 'train.csv')
        val_csv = os.path.join(output_folder, 'val.csv')
        save_csv(all_data[indices][:split], train_csv)
        save_csv(all_data[indices][split:], val_csv)
        return train_csv, val_csv

    def split_size(self, num_train):
        indices = list(range(num_train))
        split = int(np.floor(self.train_size * num_train))
        np.random.shuffle(indices)
        return indices, split

# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision import datasets
# import numpy as np
# import matplotlib.pyplot as plt
#
# n_classes = 5
# n_samples = 8
#
# mnist_train = torchvision.datasets.MNIST(root="mnist/mnist_train", train=True, download=True,
#                                          transform=transforms.Compose([transforms.ToTensor(), ]))
#
# balanced_batch_sampler = BalancedBatchSampler(mnist_train, n_classes, n_samples)
#
# dataloader = torch.utils.data.DataLoader(mnist_train, batch_sampler=balanced_batch_sampler)
# my_testiter = iter(dataloader)
# images, target = my_testiter.next()
#
#
# imshow(torchvision.utils.make_grid(images))

# class BalancedBatchSampler(BatchSampler):
#     """
#     BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
#     Returns batches of size n_classes * n_samples
#     """
#
#     def __init__(self, dataset, n_classes, n_samples):
#         loader = DataLoader(dataset)
#         self.labels_list = []
#         for _, label in loader:
#             self.labels_list.append(label)
#         self.labels = torch.LongTensor(self.labels_list)
#         self.labels_set = list(set(self.labels.numpy()))
#         self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
#                                  for label in self.labels_set}
#         for l in self.labels_set:
#             np.random.shuffle(self.label_to_indices[l])
#         self.used_label_indices_count = {label: 0 for label in self.labels_set}
#         self.count = 0
#         self.n_classes = n_classes
#         self.n_samples = n_samples
#         self.dataset = dataset
#         self.batch_size = self.n_samples * self.n_classes
#
#     def __iter__(self):
#         self.count = 0
#         while self.count + self.batch_size < len(self.dataset):
#             classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
#             indices = []
#             for class_ in classes:
#                 indices.extend(self.label_to_indices[class_][
#                                self.used_label_indices_count[class_]:self.used_label_indices_count[
#                                                                          class_] + self.n_samples])
#                 self.used_label_indices_count[class_] += self.n_samples
#                 if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
#                     np.random.shuffle(self.label_to_indices[class_])
#                     self.used_label_indices_count[class_] = 0
#             yield indices
#             self.count += self.n_classes * self.n_samples
#
#     def __len__(self):
#         return len(self.dataset) // self.batch_size
