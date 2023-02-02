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

def train_sketchysgd(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", rank = 100, wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], lr_rho_eq = False, lr_rho_prop = False, n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

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
    wd /= config["lr"]

    # If we want lr = rho, set them equal in the optimizer
    if lr_rho_eq:
        optimizer = SketchySGD(model.parameters(), rank = rank, rho = config["lr"], lr = config["lr"], weight_decay = wd, hes_update_freq = hes_interval, proportional = lr_rho_prop, device = device)
    else:
        optimizer = SketchySGD(model.parameters(), rank = rank, rho = config["rho"], lr = config["lr"], weight_decay = wd, hes_update_freq = hes_interval, proportional = lr_rho_prop, device = device)
    print("For SketchySGD, we use the decoupled weight decay as AdamW. Here we automatically correct this for you! If this is not what you want, please modify the code!")

    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, True, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        if lr_rho_eq:
            tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                        test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)
        else:
            tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], rho = config["rho"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                        test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)

def train_sgd(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = config["lr"], weight_decay = wd)
    
    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, False, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)

def train_adam(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config["lr"], weight_decay = wd)

    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, False, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)

def train_adamw(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    wd /= config["lr"]
    optimizer = optim.AdamW(model.parameters(), lr = config["lr"], weight_decay = wd)
    print('For AdamW, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')
    
    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, False, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)

def train_adahessian(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    wd /= config["lr"]
    optimizer = torch_optimizer.Adahessian(model.parameters(), lr = config["lr"], weight_decay = wd)
    print('For AdaHessian, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')
    
    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, True, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)


def train_shampoo(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    wd /= config["lr"]
    optimizer = torch_optimizer.Shampoo(model.parameters(), lr = config["lr"], weight_decay = wd)
    print('For Shampoo, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')
    
    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(train_loader, test_loader, model, criterion, optimizer, True, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)


def train_kfac(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    wd /= config["lr"]
    optimizer = KFACOptimizer(model, lr = config["lr"], weight_decay = wd)
    print('For KFAC, we automatically correct the weight decay term for you! If this is not what you want, please modify the code!')
    
    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch_kfac(train_loader, test_loader, model, criterion, optimizer, True, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)


def train_seng(config, checkpoint_dir = None, data_dir = None, model_path = None, dataset = "cifar10", wd = 0.0, lr_decay = 0.1, 
                            lr_decay_epoch = [80, 120], n_epochs = 200, batch_size = 128, seed = 1234, device = "cpu", activation = "relu"):
    # Use deterministic algorithms only
    make_deterministic()

    # Initialize model
    model = load_model(activation, dataset, model_path)
    model.to(device)

    # Set seeds so we can make the batches come in the same order across optimizers
    set_seeds(seed, seed, seed, seed)

    train_loader, test_loader = get_data(name = dataset, data_dir = data_dir,
                                    train_bs = batch_size, test_bs = batch_size)

    # Set up for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = config["lr"], weight_decay = wd)
    preconditioner = SENG(model, 0.05, update_freq = 200, col_sample_size = 256)
    # Learning rate schedule
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = lr_decay_epoch, gamma = lr_decay, last_epoch = -1)

    # Enable loading from checkpoints
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    tot_time = 0
    for epoch in range(n_epochs):
        epoch_time, train_epoch_loss, train_epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch_seng(train_loader, test_loader, model, criterion, optimizer, preconditioner, True, device, scheduler)
        tot_time += epoch_time

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        tune.report(dataset = dataset, time = tot_time, epochs = epoch+1, lr = config["lr"], weight_decay = wd, train_loss = train_epoch_loss, train_acc = train_epoch_acc,
                    test_loss = test_epoch_loss, test_acc = test_epoch_acc, seed = seed)



def sel_config(optimizer, lr_list, rho_list, lr_rho_eq):
    config = None
    if optimizer == "sketchysgd":
        if lr_rho_eq:
            config = {
                "lr": tune.grid_search(lr_list)
            }
        else:
            config = {
                "lr": tune.grid_search(lr_list),
                "rho": tune.grid_search(rho_list),
            }
    elif optimizer in ["sgd", "adam", "adamw", "adahessian", "shampoo", "kfac", "seng"]:
        config = {
            "lr": tune.grid_search(lr_list)
        }
    else:
        raise RuntimeError("We do not currently support this optimizer!")
    
    return config

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, required = True) # Optimizer
    parser.add_argument('--data', type = str, required = True) # Dataset
    parser.add_argument('--trials', type = int, required = True) # Number of times to run the grid search
    parser.add_argument('--epochs', type = int, required = True) # Number of epochs
    parser.add_argument('--act', type = str, required = True) # Activation
    parser.add_argument('--wd', type = float, required = True) # Weight decay
    parser.add_argument('--bs', type = int, required = True) # Batch size (train and test)
    parser.add_argument('--lr-decay', type = float, required = True) # Learning rate decay
    parser.add_argument('--lr-decay-epoch', type = int, nargs='+', required = True) # Learning rate epochs
    parser.add_argument('--lr-rho-eq', action = 'store_true') # Allows us to set rho = lr at the start
    parser.add_argument('--lr-rho-prop', action = 'store_true') # Option to preserve ratio between lr and rho, even as lr decays
    parser.add_argument('--ngpu', type = int, required = True) # Number of GPUs to use
    parser.add_argument('--init-seed', type = int, required = True) # Seed for weight initialization
    parser.add_argument('--batch-seed', type = int, required = True) # Seed for batch order
    parser.add_argument('--dir', type = str, required = True) # Directory to save results

    args = parser.parse_args()
    opt = args.opt
    dataset = args.data
    num_samples = args.trials
    max_num_epochs = args.epochs
    activation = args.act
    weight_decay = args.wd
    batch_size = args.bs
    lr_decay = args.lr_decay
    lr_decay_epoch = args.lr_decay_epoch
    lr_rho_eq = args.lr_rho_eq
    lr_rho_prop = args.lr_rho_prop
    n_gpus = args.ngpu
    init_seed = args.init_seed
    batch_seed = args.batch_seed
    dir = args.dir

    print("optimizer=", opt)    
    print("dataset=", dataset)
    print("num_samples=", num_samples)
    print("max_num_epochs=", max_num_epochs)
    print("activation=", activation)
    print("weight decay=", weight_decay)
    print("batch size=", batch_size)
    print("lr decay=", lr_decay)
    print("lr decay epoch=", lr_decay_epoch)
    print("lr_rho_eq=", lr_rho_eq)
    print("lr_rho_prop=", lr_rho_prop)
    print("n_gpus=", n_gpus)
    print("init_seed=", init_seed)
    print("batch_seed=", batch_seed)
    print("output directory=", dir)

    cwd = os.getcwd()
    results_folder = os.path.join(cwd, "results", opt, dataset, dir)

    # Write command line arguments to file
    filename = os.path.join(results_folder, "arguments.txt")
    with open(filename, "w") as f:
        json.dump(args.__dict__, f, indent = 2)

    # Set seed for reproducibile model initialization
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)

    # Set device
    device = None
    if torch.cuda.is_available():
        gpu_ids = get_least_used_gpus(n_gpus)
        gpu_ids = map(str, gpu_ids)
        device = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    else:
        device = torch.device("cpu")

    # Save initialized network
    model_path = init_model(dataset, activation, device)

    # Set up Ray tune for running all trials
    rank = 100
    data_dir = os.path.abspath("./data")
    lr_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e-0, 3e-0]
    rho_list = [0.1]
    config = sel_config(opt, lr_list, rho_list, lr_rho_eq)

    reporter = CLIReporter(
        metric_columns = (["dataset", "weight_decay", "time", "epochs", "train_loss", "train_acc",
         "test_loss", "test_acc", "seed"])
    )

    n_cpu = 2
    gpu_frac = 0.32

    # Run the trials depending on the optimizer
    result = None
    if opt == "sketchysgd":
        result = tune.run(
            partial(train_sketchysgd, data_dir = data_dir, model_path = model_path, dataset = dataset, rank = rank, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    lr_rho_eq = lr_rho_eq, lr_rho_prop = lr_rho_prop, n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )
    elif opt == "sgd":
        result = tune.run(
            partial(train_sgd, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )
    elif opt == "adam":
        result = tune.run(
            partial(train_adam, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )
    elif opt == "adamw":
        result = tune.run(
            partial(train_adamw, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )
    elif opt == "adahessian":
        result = tune.run(
            partial(train_adahessian, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        ) 
    elif opt == "shampoo":
        result = tune.run(
            partial(train_shampoo, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )

    elif opt == "kfac":
        result = tune.run(
            partial(train_kfac, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )

    elif opt == "seng":
        result = tune.run(
            partial(train_seng, data_dir = data_dir, model_path = model_path, dataset = dataset, wd = weight_decay, lr_decay = lr_decay, lr_decay_epoch = lr_decay_epoch, 
                    n_epochs = max_num_epochs, batch_size = batch_size,
                    seed = batch_seed, device = device, activation = activation),
            resources_per_trial = {"cpu": n_cpu, "gpu": gpu_frac},
            config = config,
            num_samples = num_samples,
            progress_reporter = reporter,
            raise_on_failed_trial = False, # needed for instances where we get NaNs due to bad hyperparameters
            verbose = 2
        )
    else:
        raise RuntimeError("We do not currently support this optimizer for training!")

    # Write the results to csv files
    all_trials_dfs = result.trial_dataframes

    for key, df in all_trials_dfs.items():
        if opt == "sketchysgd":
            if lr_rho_eq:
                csv_name = dataset+"_lr_"+str(df["lr"][0])+"_rho_"+str(df["lr"][0])+".csv"
            else:
                csv_name = dataset+"_lr_"+str(df["lr"][0])+"_rho_"+str(df["rho"][0])+".csv"
        else:
            csv_name = dataset+"_lr_"+str(df["lr"][0])+".csv"
        filename = os.path.join(results_folder, csv_name)
        df.to_csv(filename)

if __name__ == "__main__":
    main()
