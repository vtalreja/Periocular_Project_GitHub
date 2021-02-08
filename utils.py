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



def adjust_learning_rate(optimizer, init_lr, target_lr, epoch, factor, every):
    lrd = (init_lr - target_lr) / every
    old_lr = optimizer.param_groups[0]['lr']
    # linearly decaying lr
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def plot_metrics (train_metric,val_metric,results_dir,metric):
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation " + metric)
    plt.plot(train_metric, label="Training")
    plt.plot(val_metric, label="Val")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join (results_dir,metric +'.png'))

def write_results_csv(text_file, data):
    if text_file != None:
        with open(text_file, 'a') as fp:
            np.savetxt(fp, np.asarray(data), delimiter=',')


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    train_metrics = checkpoint['train_metrics']
    val_metrics = checkpoint['val_metrics']
    return model, optimizer, start_epoch,train_metrics,val_metrics

def save_ckp(save_dir,epoch,state_dict,optimizer,train_metrics,val_metrics):
    checkpoint = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer,
        'val_metrics':val_metrics,
        'train_metrics':train_metrics
    }
    torch.save(checkpoint,os.path.join(save_dir,'epoch_' + str(epoch) + '.pt'))