import argparse
import os
import pandas as pd
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import numpy as np
from experiment import Experiment
from utils import *

LOGISTIC_HYPERPARAMS = {
    'sketchysgd': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3}
}

LS_HYPERPARAMS = {
    'sketchysgd': {'rank': 10, 'rho': 1e-3}
}

BATCH_SIZE = 256 # Minibatch size for stochastic gradients

def get_experiment_data(dataset, data_source, problem_type, rescale, rf_params_set):
    rf_params = None
    if dataset in list(rf_params_set.keys()):
        rf_params = rf_params_set[dataset]
    if data_source == 'libsvm':
        data = load_preprocessed_data(dataset, problem_type, rescale, rf_params)
    elif data_source == 'openml':
        data = load_preprocessed_data_openml(dataset, rescale, rf_params)
    return data

# Get all the experiments for a given dataset, model, and optimizer
def get_experiments(data, model_type, model_params, opt, precond_type, hyperparams, max_epochs, bh):
    # Get the update frequency
    if model_type == 'logistic':
        update_freq = hyperparams['update_freq']
    elif model_type == 'least_squares':
        update_freq = {'precond': (max_epochs * 2, 'epochs')} # Ensure the preconditioner is held fixed for the entire run

    opt_params = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh}
    experiments = [Experiment(data, model_type, model_params, opt, opt_params)]

    return experiments

def write_spectrum(directory, spectrum, epoch, is_precond):
    if is_precond:
        filename = 'hessian_precond_spectrum_' + str(epoch) + '.csv'
    else:
        filename = 'hessian_spectrum_' + str(epoch) + '.csv'
    # Save as numpy array
    np.save(os.path.join(directory, filename), spectrum)

def get_spectrums(exp, max_epochs, bg, eig_num, directory):
    exp.preprocess_opt_params(bg)
    opt = exp.create_opt()

    for i in range(max_epochs):
        batches = minibatch_indices(exp.model.ntr, bg)

        # Loop through every minibatch
        for j, batch in enumerate(batches):
            old_w = exp.model.w.copy() # Need to save the old weights used to compute the preconditioner
            opt.step(batch)

            if j == 0: # This assumes the preconditioner is updated at the beginning of each epoch
                # Get the spectrum of the full Hessian + preconditioned full Hessian (at the location where the preconditioner was computed)
                diag = exp.model.get_hessian_diag(np.arange(exp.model.ntr), v = old_w)
                # hessian = exp.model.Atr.T @ np.diag(diag) @ exp.model.Atr + exp.model.mu * np.eye(exp.model.p)

                def hess_mv(x):
                    return exp.model.Atr.T @ (diag * (exp.model.Atr @ x))
                lin_op_hess = LinearOperator((exp.model.p, exp.model.p),
                                            matvec = lambda x: hess_mv(x.reshape(-1)),
                                            rmatvec = lambda x: hess_mv(x.reshape(-1)))
                
                eigs_hessian = sp.sparse.linalg.eigs(lin_op_hess,
                                                    k = eig_num, 
                                                    which = 'LM',
                                                    return_eigenvectors = False)
                eigs_hessian = np.real(eigs_hessian)
                eigs_hessian += exp.model.mu

                diag_inv_precond = (opt.precond.S + opt.precond.rho) ** (-1)
                # precond = opt.precond.U @ np.diag(diag_inv_precond) @ \
                #         opt.precond.U.T + 1/opt.precond.rho * (np.eye(exp.model.p) - opt.precond.U @ opt.precond.U.T)
                # hessian_precond = hessian @ precond

                def hess_mv_reg(x):
                    return hess_mv(x) + exp.model.mu * x
                def precond_mv(x):
                    UTx = opt.precond.U.T @ x
                    part1 = opt.precond.U @ (diag_inv_precond * UTx)
                    part2 = 1/opt.precond.rho * (x - opt.precond.U @ UTx)
                    return part1 + part2
                
                lin_op_hess_precond = LinearOperator((exp.model.p, exp.model.p),
                                                    matvec = lambda x: hess_mv_reg(precond_mv(x.reshape(-1))),
                                                    rmatvec = lambda x: precond_mv(hess_mv_reg(x.reshape(-1))))
                eigs_hessian_precond = sp.sparse.linalg.eigs(lin_op_hess_precond, 
                                                            k = eig_num, 
                                                            which = 'LM', 
                                                            return_eigenvectors = False)
                eigs_hessian_precond = np.real(eigs_hessian_precond)

                # Get the spectrum of the full Hessian + preconditioned full Hessian
                # eigs_hessian = np.linalg.eigvals(hessian)
                # eigs_hessian_precond = np.linalg.eigvals(hessian_precond)

                write_spectrum(directory, eigs_hessian, i, False)
                write_spectrum(directory, eigs_hessian_precond, i, True)

        losses = exp.model.get_losses()
        print('Epoch: ' + str(i) + ', Training loss: ' + str(losses['train_loss']))

def main():
    # Get arguments from command line
    parser = argparse.ArgumentParser(description = 'Compute Hessian spectrums along SketchySGD (Nystrom) trajectory for least-squares and logistic regression. \
                                    Hyperparameters are selected according to LOGISTIC_HYPERPARAMS and LS_HYPERPARAMS at the top of the file.\n \
                                    Datasets are automatically normalized/standardized and random features are applied if applicable. \
                                    Random seeds are set according to SEEDS in constants.py.')
    parser.add_argument('--data', type = str, required = True, help = 'Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES/LS_DATA_FILES/LS_DATA_FILES_OPENML in constants.py') # Dataset
    parser.add_argument('--problem', type = str, required = True, help = "Type of problem: either 'logistic' or 'least_squares'") # Problem type
    parser.add_argument('--epochs', type = int, required = True, help = 'Number of epochs to run the optimizer') # Number of epochs to run
    parser.add_argument('--eig_num', type = int, required = True, help = 'Number of eigenvalues to compute') # Number of eigenvalues to compute
    parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
    parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # Extract arguments
    args = parser.parse_args()
    dataset = args.data
    problem_type = args.problem
    epochs = args.epochs
    eig_num = args.eig_num
    mu_unscaled = args.mu
    results_dest = os.path.abspath(args.dest)

    opt = 'sketchysgd'
    precond_type = 'nystrom'

    directory = os.path.join(results_dest, dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Print key parameters
    print(f"Dataset: {dataset}")
    print(f"Problem type: {problem_type}")
    print(f"Optimization method: {opt}")
    print(f"Preconditioner type: {precond_type}")
    print(f"Number of epochs: {epochs}")
    print(f"Number of eigenvalues: {eig_num}")
    print(f"Unscaled regularization parameter: {mu_unscaled}")
    print(f"Results directory: {directory}\n")

    # Ensure normalization/standardization occurs
    rescale = True
    set_random_seeds(**SEEDS)

    if problem_type == 'logistic':
        rf_params_set = LOGISTIC_RAND_FEAT_PARAMS
        hyperparam_set = LOGISTIC_HYPERPARAMS
    elif problem_type == 'least_squares':
        rf_params_set = LS_RAND_FEAT_PARAMS
        hyperparam_set = LS_HYPERPARAMS

    # Get data
    # Furthermore, apply random features if applicable
    if dataset in list(LOGISTIC_DATA_FILES.keys()) or dataset in list(LS_DATA_FILES.keys()):
        data_source = 'libsvm'
    elif dataset in list(LS_DATA_FILES_OPENML.keys()):
        data_source = 'openml'
    data = get_experiment_data(dataset, data_source, problem_type, rescale, rf_params_set)

    # Get experiments
    hyperparams = hyperparam_set[opt] # Get hyperparameters
    ntr = data['Atr'].shape[0]
    model_params = {'mu': mu_unscaled / ntr}
    bh = int(ntr ** (0.5)) # Hessian batch size

    experiments = get_experiments(data, problem_type, model_params, opt, precond_type, hyperparams, epochs, bh)
    get_spectrums(experiments[0], epochs, BATCH_SIZE, eig_num, directory)

if __name__ == '__main__':
    main()