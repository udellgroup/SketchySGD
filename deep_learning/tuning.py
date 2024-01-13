import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch._C import _LinAlgError
import numpy as np
import sklearn
import optuna
from optuna.samplers import RandomSampler
import wandb
import timeit
from functools import partial

from torch.optim import SGD
from torch_optimizer import Adahessian
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from dl_opts.sketchysgd import SketchySGD
from models.mlp import MLP

from utils import check_opt, get_tune_dataset, get_opt, compute_loss_acc

import argparse

def tuning(trial, proj_name, opt_name,
            input_size, output_size, hidden_layers, 
            loss_fn,
            t_dl, t_dl_h, t_dl_2, v_dl,
            epochs, 
            device):
    run = wandb.init(project=proj_name, reinit=True)

    # Create neural net
    model = MLP(input_size, output_size, hidden_layers).to(device)

    # Get optimizer
    Opt = get_opt(opt_name)
    opt = None

    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    wandb.config.lr = lr

    if Opt == SketchySGD:
        hes_update_freq = len(t_dl)
        opt = Opt(model.parameters(), lr=lr, rho=1.0, rank=10,
                        momentum=0.9,
                        chunk_size=10, verbose=False)
    elif Opt == SGD:
        opt = Opt(model.parameters(), lr=lr, momentum=0.9)
    elif Opt == DistributedShampoo:
        opt = Opt(model.parameters(), lr=lr, momentum=0.9, 
                    precondition_frequency=len(t_dl), num_trainers_per_group=1)
    else:
        opt = Opt(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)

    try:
        # Perform optimization
        n_iters = 0
        cum_time = 0
        for epoch in range(epochs):
            # Train the model
            model.train()
            epoch_start = timeit.default_timer()
            for x, y in t_dl:
                # Infrequent updating for SketchySGD
                if isinstance(opt, SketchySGD) and n_iters % hes_update_freq == 0:
                    for x_h, y_h in t_dl_h:
                        x_h, y_h = x_h.to(device), y_h.to(device)
                        y_h_hat = model(x_h)
                        l_h = loss_fn(y_h_hat, y_h)
                        break
                    grad_tuple = torch.autograd.grad(l_h, model.parameters(), create_graph=True)
                    opt.update_preconditioner(grad_tuple)
                    # opt.profile_update_preconditioner(grad_tuple)

                opt.zero_grad()
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

                if isinstance(opt, Adahessian):
                    loss.backward(create_graph=True)
                else:
                    loss.backward()

                opt.step()
                n_iters += 1

            scheduler.step()
            cum_time += timeit.default_timer() - epoch_start

            model.eval()
            with torch.no_grad():
                train_loss, train_acc = compute_loss_acc(model, loss_fn, t_dl_2, output_size, device)
                valid_loss, valid_acc = compute_loss_acc(model, loss_fn, v_dl, output_size, device)
                wandb.log({"epoch": epoch,
                            "time": cum_time,
                            "train_loss": train_loss,
                            "valid_loss": valid_loss,
                            "train_acc": train_acc,
                            "valid_acc": valid_acc})
        run.finish()
        return valid_acc
    except _LinAlgError as e:
        run.finish()
        print(f"Trial failed due to {e}.")
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type = int, required = True)
    parser.add_argument('--data_folder', type = str, required = False, default = './data')
    parser.add_argument('--t_seed', type = int, required = False, default = 1729)
    parser.add_argument('--o_seed', type = int, required = False, default = 1729)
    parser.add_argument('--proj_name', type = str, required = True)
    parser.add_argument('--opt', type = str, required = True)
    parser.add_argument('--epochs', type = int, required = False, default = 105)
    parser.add_argument('--n_trials', type = int, required = True)

    # Extract arguments
    args = parser.parse_args()
    # ids: volkert = 41166, Fashion-MNIST = 40996, Devnagari-Script = 40923
    id = args.id 
    data_folder = args.data_folder
    t_seed = args.t_seed
    o_seed = args.o_seed
    proj_name = args.proj_name
    opt = args.opt
    epochs = args.epochs
    n_trials = args.n_trials

    # Check optimizer name is valid
    check_opt(opt)

    # Set random seed
    torch.manual_seed(t_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset, input_size, y_train = get_tune_dataset(data_folder, id)

    n_train = len(train_dataset)
    bh = int(n_train ** 0.5)
    bsz = 128

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    train_dataloader_hess = DataLoader(train_dataset, batch_size=bh, shuffle=True) # Used for updating the preconditioner
    train_dataloader2 = DataLoader(train_dataset, batch_size=4096, shuffle=False) # Used for computing metrics on the training set
    val_dataloader = DataLoader(val_dataset, batch_size=4096, shuffle=False)

    # Compute class weights using the training set
    n_classes = len(np.unique(y_train))
    class_weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    hidden_layers = [512] * 9

    tuning_partial = partial(tuning,
                            proj_name=proj_name,
                            opt_name=opt,
                            input_size=input_size,
                            output_size=n_classes,
                            hidden_layers=hidden_layers,
                            loss_fn=loss_fn,
                            t_dl=train_dataloader,
                            t_dl_h=train_dataloader_hess,
                            t_dl_2=train_dataloader2,
                            v_dl=val_dataloader,
                            epochs=epochs,
                            device=device)

    study = optuna.create_study(sampler=RandomSampler(seed=o_seed), direction='maximize')
    study.optimize(tuning_partial, n_trials=n_trials)

if __name__ == "__main__":
    main()