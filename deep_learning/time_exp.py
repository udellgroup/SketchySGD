import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch_optimizer

from functools import partial
import random
import numpy as np
import pandas as pd
import timeit
import os
import argparse
import json

from ray import tune
from ray.tune import CLIReporter

from models.resnet import resnet20, resnet32
from models.mlp import MultiClassification
from optimizers.sketchy_sgd import SketchySGD
from optimizers.kfac import KFACOptimizer
from optimizers.seng import SENG
from utils import *
from average_meter import AverageMeter

def init_model(dataset, activation, device):
    model = None
    if dataset == "cifar10":
        model = resnet20(activation = activation)
        filename = dataset+"_resnet.pth"
    elif dataset == "svhn":
        model = resnet32(activation = activation)
        filename = dataset+"_resnet.pth"
    elif dataset == "volkert":
        model = MultiClassification([180, 256, 256, 256, 10], activation = activation)
        filename = dataset+"_multiclass.pth"
    elif dataset == "adult":
        model = MultiClassification([123, 128, 128, 128, 2], activation = activation)
        filename = dataset+"_multiclass.pth"
    elif dataset == "miniboone":
        model = MultiClassification([50, 128, 128, 128, 2], activation = activation)
        filename = dataset+"_multiclass.pth"
    elif dataset == "higgs":
        model = MultiClassification([28, 64, 64, 64, 2], activation = activation)
        filename = dataset+"_multiclass.pth"
    else:
        raise RuntimeError("This dataset is not supported at this time")
    model.to(device)
    torch.save(model.state_dict(), filename)

    return os.path.abspath(filename)

def load_model(activation, dataset, path):
    model = None
    if dataset == "cifar10":
        model = resnet20(activation = activation)
    elif dataset == "svhn":
        model = resnet32(activation = activation)
    elif dataset == "volkert":
        model = MultiClassification([180, 256, 256, 256, 10], activation = activation)
    elif dataset == "adult":
        model = MultiClassification([123, 128, 128, 128, 2], activation = activation)
    elif dataset == "miniboone":
        model = MultiClassification([50, 128, 128, 128, 2], activation = activation)
    elif dataset == "higgs":
        model = MultiClassification([28, 64, 64, 64, 2], activation = activation)
    else:
        raise RuntimeError("Model could not be loaded because dataset is not supported!")
    model.load_state_dict(torch.load(path))

    return model

def set_seeds(r_seed, np_seed, torch_seed, torch_cuda_seed):
    random.seed(r_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_cuda_seed)

def train_one_epoch(train_loader, test_loader, model, criterion, optimizer, create_graph, device, scheduler = None):
    # create_graph is a necessary argument, since SketchySGD requires create_graph = True
    epoch_start_time = timeit.default_timer()
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)
        acc = multi_acc(y_pred, y_batch)

        loss.backward(create_graph = create_graph)

        optimizer.step()

        epoch_loss.update(loss.item(), X_batch.size(0))
        epoch_acc.update(acc.item(), X_batch.size(0))

    # Adjust learning rate
    if scheduler is not None:
        scheduler.step() 

    # Get loss, accuracy for train, validation, and test sets
    train_epoch_loss = epoch_loss.avg
    train_epoch_acc = epoch_acc.avg
    test_epoch_loss, test_epoch_acc = dataset_loss_acc(model, criterion, test_loader, device)

    epoch_end_time = timeit.default_timer()

    return epoch_end_time - epoch_start_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc


def train_one_epoch_kfac(train_loader, test_loader, model, criterion, optimizer, create_graph, device, scheduler = None):
    # create_graph is a necessary argument, since SketchySGD requires create_graph = True
    epoch_start_time = timeit.default_timer()
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)
        acc = multi_acc(y_pred, y_batch)

        if optimizer.steps % optimizer.TCov == 0:
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(y_pred.cpu().data, dim = 1), 1).squeeze().cuda()
            loss_sample = criterion(y_pred, sampled_y)
            loss_sample.backward(retain_graph = True)
            optimizer.acc_stats = False
            optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item(), X_batch.size(0))
        epoch_acc.update(acc.item(), X_batch.size(0))

    # Adjust learning rate
    if scheduler is not None:
        scheduler.step() 

    # Get loss, accuracy for train, validation, and test sets
    train_epoch_loss = epoch_loss.avg
    train_epoch_acc = epoch_acc.avg
    test_epoch_loss, test_epoch_acc = dataset_loss_acc(model, criterion, test_loader, device)

    epoch_end_time = timeit.default_timer()

    return epoch_end_time - epoch_start_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc


def train_one_epoch_seng(train_loader, test_loader, model, criterion, optimizer, preconditioner, create_graph, device, scheduler = None):
    # create_graph is a necessary argument, since SketchySGD requires create_graph = True
    epoch_start_time = timeit.default_timer()
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch)
        acc = multi_acc(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        preconditioner.step()
        optimizer.step()

        epoch_loss.update(loss.item(), X_batch.size(0))
        epoch_acc.update(acc.item(), X_batch.size(0))

    # Adjust learning rate
    if scheduler is not None:
        scheduler.step() 

    # Get loss, accuracy for train, validation, and test sets
    train_epoch_loss = epoch_loss.avg
    train_epoch_acc = epoch_acc.avg
    test_epoch_loss, test_epoch_acc = dataset_loss_acc(model, criterion, test_loader, device)

    epoch_end_time = timeit.default_timer()

    return epoch_end_time - epoch_start_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc

def train_sketchysgd(data_dir = None, model_path = None, dataset = "cifar10", rank = 100, wd = 0.0, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu"):

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    hes_interval = 2 * len(train_loader) # Hessian update frequency
    lr = 0.01
    wd /= lr

    optimizer = SketchySGD(model.parameters(), rank = rank, rho = 0.1, lr = lr, weight_decay = wd, hes_update_freq = hes_interval, proportional = False, device = device)
    print("For SketchySGD, we use the decoupled weight decay as AdamW. Here we automatically correct this for you! If this is not what you want, please modify the code!")

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, True, device)
        tot_time += epoch_time
    
    return tot_time

def train_adahessian(data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu"):

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    wd /= lr
    optimizer = torch_optimizer.Adahessian(model.parameters(), lr = lr, weight_decay = wd)
    print('For AdaHessian, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')


    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, True, device)
        tot_time += epoch_time
    
    return tot_time

def train_shampoo(data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu"):

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    wd /= lr
    optimizer = torch_optimizer.Shampoo(model.parameters(), lr = lr, weight_decay = wd)
    print('For Shampoo, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, True, device)
        tot_time += epoch_time
    
    return tot_time



def train_kfac(data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu"):
    # Use deterministic algorithms only
    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)
    lr = 0.01
    # Set up for training
    criterion = nn.CrossEntropyLoss()
    wd /= lr
    optimizer = KFACOptimizer(model, lr = lr, weight_decay = wd)
    print('For KFAC, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')
    
    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch_kfac(train_loader, test_loader, model, criterion, optimizer, True, device)
        tot_time += epoch_time

    return tot_time

def train_seng(data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu"):
    # Use deterministic algorithms only
    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = wd)
    preconditioner = SENG(model, 0.05, update_freq = 1, col_sample_size = 256)
    
    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch_seng(train_loader, test_loader, model, criterion, optimizer, preconditioner, True, device)
        tot_time += epoch_time
    
    return tot_time

def main():
    #dataset = 'higgs'
    #dataset = 'miniboone'
    dataset = 'volkert'

    max_num_epochs = 10
    activation = 'relu'
    weight_decay = 0.0005
    batch_size = 128
    lr_decay = 0.1
    init_seed = 1234
    batch_seed = 1111

    # Set seed for reproducibile model initialization
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)

    # Set device
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = torch.device("cpu")

    # Save initialized network
    model_path = init_model(dataset, activation, device)

    # Set up Ray tune for running all trials
    rank = 100
    data_dir = os.path.abspath("./data")
    
    time_adahessian = train_adahessian(data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu")
    time_shampoo = train_shampoo(data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu")
    time_kfac = train_kfac(data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu")
    time_seng = train_seng(data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu")
    time_sketchysgd = train_sketchysgd(data_dir = data_dir, model_path = model_path, dataset = dataset, rank = 100, wd = weight_decay, n_epochs = 10, batch_size = 128, seed = 1234, device = "cuda", activation = "relu")
    print('sketchysgd time', time_sketchysgd)
    print('adahessian time', time_adahessian)
    print('shampoo time', time_shampoo)
    print('kfac time', time_kfac)
    print('seng time', time_seng)


if __name__ == "__main__":
    main()
