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
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)



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


class calculations():
    def __init__(self, head_config, device, outputs, gts):
        self.head_config= head_config
        self.device = device
        self.outputs = outputs
        self.gts = gts


    def calculate_loss(self,criterion,weights=None):
        '''
        This function is use to calculate the loss for each head
        and combine them together using the weights parameter (if provided) as loss coefficients.
        If weights parameter is None, then we just use weights (equal to 1) for all the losses

        Args:
        preds (dict) : Predictions of the model for the given inputs.
        gts (dict) : Ground Truths for the given inputs.
        weights (dict): If provided, these weights are loss coefficients for each head.

        Returns:
        loss (tensor) : Total loss for all the heads combined. The loss for each head is weighted by the weights parameter.
        '''
        loss = torch.tensor(0.0, requires_grad=True)

        for head_name, num_labels in self.head_config.items():
            self.gts[head_name] = self.gts[head_name].to(self.device) # required as the preds and gts have to be on the same device (cuda or cpu)

            if weights == None:  # If weights are not provided, we just use all weights equal to 1
                loss = loss + criterion(self.outputs[head_name], self.gts[head_name])

            else:  # If weights are provided, then we use the corresponding weights for the respective heads

                loss = loss + weights[head_name] * criterion(self.outputs[head_name], self.gts[head_name])

        return loss

    def calculate_predictions(self):
        '''
        This function is use to calculate the label prediction for each head

        preds (dict) : Predictions of the model for the given inputs.


        Returns:
        loss (tensor) : Total loss for all the heads combined. The loss for each head is weighted by the weights parameter.
        '''
        preds = {}
        for head_name, num_labels in self.head_config.items():
            _,preds[head_name] = torch.max(self.outputs[head_name], 1)

        return preds
    def calculate_running_corrects (self,preds,running_corrects):
        for head_name, num_labels in self.head_config.items():
            x = torch.sum(preds[head_name] == self.gts[head_name].data)
            running_corrects[head_name] += torch.sum(preds[head_name] == self.gts[head_name].data)
        return running_corrects

    def calculate_acc (self,running_corrects,dataloaders):
        acc = 0.0
        for head_name, _ in self.head_config.items():
            acc += running_corrects[head_name].double() / len(dataloaders.dataset)
        return acc/len(self.head_config)


def train_model_attribute(start_epoch, model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, results_dir,
                train_metrics, val_metrics,head_config,
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
            running_corrects = {head_name: 0 for head_name, _ in head_config.items()}

            # Iterate over data.
            for data_dict in dataloaders[phase]:
                shuffled_data_dict = shuffle_dict(data_dict)
                inputs = shuffled_data_dict['img']
                labels = {name: shuffled_data_dict[name] for name,_ in head_config.items()}
                inputs = inputs.to(device)
                for name, _ in head_config.items():
                    labels[name] = labels[name].to(device)

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
                        calc = calculations(head_config=head_config, device=device, outputs=outputs, gts=labels)
                        loss = calc.calculate_loss(criterion=criterion)

                        # loss = criterion(outputs, labels)
                    preds = calc.calculate_predictions()
                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects = calc.calculate_running_corrects(preds = preds, running_corrects=running_corrects)
                # running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = calc.calculate_acc(running_corrects=running_corrects,dataloaders=dataloaders[phase])
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

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


def visualize_model_attribute(model, device, dataloaders, class_names, head_config,num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, data_dict in enumerate(dataloaders['val']):
            shuffled_data_dict = shuffle_dict(data_dict)
            inputs = shuffled_data_dict['img']
            labels = {name: shuffled_data_dict[name] for name, _ in head_config.items()}
            inputs = inputs.to(device)
            for name, _ in head_config.items():
                labels[name] = labels[name].to(device)


            outputs = model(inputs)
            calc = calculations(head_config=head_config, device=device, outputs=outputs, gts=labels)
            preds = calc.calculate_predictions()
            # _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds['gender_label'][j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def train_model_joint_optim(start_epoch, model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, results_dir,
                train_metrics, val_metrics,head_config,
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
            gender_running_corrects = {head_name: 0 for head_name, _ in head_config.items()}
            class_running_corrects = 0

            # Iterate over data.
            for data_dict in dataloaders[phase]:
                shuffled_data_dict = shuffle_dict(data_dict)
                inputs = shuffled_data_dict['img']
                gender_labels = {name: shuffled_data_dict[name] for name,_ in head_config.items()}
                class_labels = shuffled_data_dict['class_name_label']
                inputs = inputs.to(device)
                for name, _ in head_config.items():
                    gender_labels[name] = gender_labels[name].to(device)
                class_labels = class_labels.to(device)

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
                        outputs,class_outputs = model(inputs)
                        calc = calculations(head_config=head_config, device=device, outputs=outputs, gts=gender_labels)
                        gender_loss = calc.calculate_loss(criterion=criterion)
                        class_loss = criterion(class_outputs,class_labels)
                        loss = class_loss + 1.0 * gender_loss

                        # loss = criterion(outputs, labels)
                    gender_preds = calc.calculate_predictions()
                    _, class_preds = torch.max(class_outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                gender_running_corrects = calc.calculate_running_corrects(preds = gender_preds, running_corrects=gender_running_corrects)
                class_running_corrects += torch.sum(class_preds == class_labels.data)

                # running_corrects += torch.sum(preds == labels.data)
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            gender_epoch_acc = calc.calculate_acc(running_corrects=gender_running_corrects,dataloaders=dataloaders[phase])
            class_epoch_acc = class_running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc = (gender_epoch_acc+class_epoch_acc)/2

            print('{} Loss: {:.4f} Avg_Acc: {:.4f} Gender_Acc: {:.4f} Class_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,gender_epoch_acc,class_epoch_acc))
            if phase == 'train':
                train_metrics['loss'].append(epoch_loss)
                train_metrics['class_acc'].append(class_epoch_acc)
                train_metrics['gender_acc'].append(gender_epoch_acc)

            else:
                val_metrics['loss'].append(epoch_loss)
                val_metrics['class_acc'].append(class_epoch_acc)
                val_metrics['gender_acc'].append(gender_epoch_acc)
            # deep copy the model
            if phase == 'val' and class_epoch_acc > best_acc:
                best_acc = class_epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_dict = copy.deepcopy(optimizer.state_dict())
                best_scheduler = copy.deepcopy(scheduler.state_dict())
                best_train_metrics = {x: train_metrics[x][:best_epoch] for x in ['loss', 'class_acc','gender_acc']}
                best_val_metrics = {x: val_metrics[x][:best_epoch] for x in ['loss', 'class_acc','gender_acc']}

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
    print('Best Class Acc: {:4f} , which is for epoch: {}'.format(best_acc, best_epoch))

    if not os.path.exists(os.path.join(results_dir, 'epoch_' + str(
            best_epoch) + '.pt')):  # Save the best model only if it has not been saved previously in the regular updates
        save_ckp(save_dir=results_dir, epoch=best_epoch, state_dict=best_model_wts, optimizer=best_optimizer_dict,
                 train_metrics=best_train_metrics, val_metrics=best_val_metrics, scheduler = best_scheduler)
        # torch.save(best_model_wts, os.path.join(results_dir, 'epoch_' + str(best_epoch) + '.pth'))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_metrics, val_metrics


def visualize_model_joint_optim(model, device, dataloaders, attribute_mapper, head_config,results_dir,num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    predicted_gender_all=[]
    gt_gender_all=[]
    predicted_class_all =[]
    gt_class_all =[]

    with torch.no_grad():
        for i, data_dict in enumerate(dataloaders['val']):
            shuffled_data_dict = shuffle_dict(data_dict)
            inputs = shuffled_data_dict['img']
            gender_labels = {name: shuffled_data_dict[name] for name, _ in head_config.items()}
            class_labels = shuffled_data_dict['class_name_label']
            inputs = inputs.to(device)
            for name, _ in head_config.items():
                gender_labels[name] = gender_labels[name].to(device)
            class_labels = class_labels.to(device)


            outputs,class_outputs = model(inputs)
            calc = calculations(head_config=head_config, device=device, outputs=outputs, gts=gender_labels)
            gender_preds = calc.calculate_predictions()
            _, class_preds = torch.max(class_outputs, 1)
            for i in range(inputs.shape[0]):
                # image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_gender = attribute_mapper.gender_id_to_name[gender_preds['gender_label'][i].item()]
                predicted_class = attribute_mapper.class_id_to_name[class_preds[i].item()]

                gt_gender = attribute_mapper.gender_id_to_name[gender_labels['gender_label'][i].item()]
                gt_class = attribute_mapper.class_id_to_name[class_labels[i].item()]

                gt_gender_all.append(gt_gender)
                gt_class_all.append(gt_class)

                predicted_gender_all.append(predicted_gender)
                predicted_class_all.append(predicted_class)
            # predicted_gender_all.extend(
            #     prediction.item() for prediction in gender_preds
            # )
            # gt_gender_all.extend(
            #     gt_color.item() for gt_color in gender_labels
            # )
            # predicted_class_all.extend(
            #     prediction.item() for prediction in class_preds
            # )
            # gt_class_all.extend(
            #     gt_color.item() for gt_color in class_labels
            # )
        plt.figure(figsize=(10,5))
        cn_matrix = confusion_matrix(
            y_true=gt_gender_all,
            y_pred=predicted_gender_all,
            labels=attribute_mapper.gender_labels,
            normalize="true")
        ConfusionMatrixDisplay(cn_matrix, attribute_mapper.gender_labels).plot(
            include_values=False, xticks_rotation="vertical")
        plt.title("Gender")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'Gender_Confusion.png'))
        # plt.show()

        plt.figure(figsize=(30,20))
        cn_matrix = confusion_matrix(
            y_true=gt_class_all,
            y_pred=predicted_class_all,
            labels=attribute_mapper.class_name_labels,
            normalize="true")
        ConfusionMatrixDisplay(cn_matrix, attribute_mapper.class_name_labels).plot(
            include_values=False, xticks_rotation="vertical")
        plt.title("Class")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'Class_Confusion.png'))
        # plt.show()

            # for j in range(inputs.size()[0]):
            #     images_so_far += 1
            #     ax = plt.subplot(num_images // 2, 2, images_so_far)
            #     ax.axis('off')
            #     ax.set_title('predicted: {}'.format(class_names[gender_preds['gender_label'][j]]))
            #     imshow(inputs.cpu().data[j])
            #
            #     if images_so_far == num_images:
            #         model.train(mode=was_training)
            #         return
    model.train(mode=was_training)