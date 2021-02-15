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

from dataset_utils import *
from utils import *


def train_model(start_epoch, model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, results_dir,
                train_metrics, val_metrics,
                is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    updated_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # best_train_metrics = {'loss': [], 'acc': []}
    # best_val_metrics = {'loss': [], 'acc': []}
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_metrics['loss'].append(epoch_loss)
                train_metrics['acc'].append(epoch_acc)
            else:
                val_metrics['loss'].append(epoch_loss)
                val_metrics['acc'].append(epoch_acc)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_dict = copy.deepcopy(optimizer.state_dict())
                best_scheduler = copy.deepcopy(scheduler.state_dict())
                best_train_metrics = {x: train_metrics[x][:best_epoch] for x in ['loss', 'acc']}
                best_val_metrics = {x: val_metrics[x][:best_epoch] for x in ['loss', 'acc']}

        if (epoch % 5 == 0):
            updated_model_wts = copy.deepcopy(model.state_dict())
            updated_optimizer_dict = copy.deepcopy(optimizer.state_dict())
            updated_scheduler = copy.deepcopy(scheduler.state_dict())
            save_ckp(save_dir=results_dir, epoch=epoch, state_dict=updated_model_wts, optimizer=updated_optimizer_dict,
                     train_metrics=train_metrics, val_metrics=val_metrics, scheduler = updated_scheduler)
            # torch.save(updated_model_wts, os.path.join(results_dir, 'epoch_' + str(epoch) + '.pth'))

        # adjust_learning_rate(optimizer, lr, target_lr, epoch, None, num_epochs)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} , which is for epoch: {}'.format(best_acc, best_epoch))

    if not os.path.exists(os.path.join(results_dir, 'epoch_' + str(
            best_epoch) + '.pt')):  # Save the best model only if it has not been saved previously in the regular updates
        save_ckp(save_dir=results_dir, epoch=best_epoch, state_dict=best_model_wts, optimizer=best_optimizer_dict,
                 train_metrics=best_train_metrics, val_metrics=best_val_metrics, scheduler = best_scheduler)
        # torch.save(best_model_wts, os.path.join(results_dir, 'epoch_' + str(best_epoch) + '.pth'))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_metrics, val_metrics


def visualize_model(model, device, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
