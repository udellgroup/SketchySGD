import argparse
import os
import pandas as pd
from lr_ablation import LR_Ablation
from utils import *

LOGISTIC_HYPERPARAMS = {
    'adasgd': {'update_freq': {'learning_rate': (1, 'epochs')}},
    'sketchysgd': {'update_freq': {'precond': (1, 'epochs')}, 'rank': 10, 'rho': 1e-3}}

LS_HYPERPARAMS = {'adasgd': {},'sketchysgd': {'rank': 10, 'rho': 1e-3}}

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
def get_experiments(data, model_type, model_params, opt, precond_type, hyperparams,max_epochs, bg, bh):
    if opt in ['sketchysgd']:
        # Get the update frequency
        if model_type == 'logistic':
            update_freq = hyperparams['update_freq']
        elif model_type == 'least_squares':
            update_freq = {'precond': (max_epochs * 2, 'epochs')} # Ensure the preconditioner is held fixed for the entire run

        opt_params = {'precond_type': precond_type, 'update_freq': update_freq.copy(), 'rank': hyperparams['rank'], 'rho': hyperparams['rho'], 'bh': bh}
        experiments = [LR_Ablation(data, model_type, model_params, opt, opt_params)]
    elif opt in ['adasgd']:
        #Get the update frequency
        if model_type == 'logistic':
            update_freq = hyperparams['update_freq']
        elif model_type == 'least_squares':
            update_freq = {'learning_rate': (max_epochs * 2, 'epochs')}
            
        opt_params = {'update_freq': update_freq.copy(),'bh': bh}
        experiments = [LR_Ablation(data, model_type, model_params, opt, opt_params)]
    return experiments

# Writes results to a csv file
def write_as_dataframe(result, directory, opt_name, opt_params, r_seed, np_seed, r_seed_b, np_seed_b):
    df = pd.DataFrame.from_dict(result)
    csv_name = 'auto'
    csv_name += '_seed_'+str(r_seed)+'_'+str(np_seed)+ \
        '_bseed_'+str(r_seed_b)+'_'+str(np_seed_b)+'.csv'

    if not os.path.exists(directory):
        print('Creating directory: '+directory)
        os.makedirs(directory)

    file_name = os.path.join(directory, csv_name)
    df.to_csv(file_name)

def main(dataset,problem_type, opt, precond_type,epochs, mu_unscaled,n_runs,results_dest):
    # Get arguments from command line
#     parser = argparse.ArgumentParser(description = 'Run experiments for least-squares and logistic regression. \
#                                     Hyperparameters are selected according to LOGISTIC_HYPERPARAMS and LS_HYPERPARAMS at the top of the file.\n \
#                                     Datasets are automatically normalized/standardized and random features are applied if applicable. \
#                                     Random seeds are set according to SEEDS in constants.py.')
#     parser.add_argument('--data', type = str, required = True, help = 'Name of a dataset, i.e., a key in LOGISTIC_DATA_FILES/LS_DATA_FILES/LS_DATA_FILES_OPENML in constants.py') # Dataset
#     parser.add_argument('--problem', type = str, required = True, help = "Type of problem: either 'logistic' or 'least_squares'") # Problem type
#     parser.add_argument('--opt', type = str, required = True, help = "Optimization method: either 'adasgd' or 'sketchysgd'") # Optimizer
#     parser.add_argument('--precond', type = str, default = None, help = "Preconditioner type: one of 'nystrom' or 'ssn' (default is None)")
#     parser.add_argument('--epochs', type = int, required = True, help = 'Number of epochs to run the optimizer') # Number of epochs to run
#     parser.add_argument('--mu', type = float, required = False, default = 1e-2, help = 'Unscaled regularization parameter (default is 1e-2)') # Regularization parameter
#     parser.add_argument('--n_runs', type = int, required = False, default = 1, help = 'Number of runs to perform (default is 1)') # Number of runs to perform
#     parser.add_argument('--dest', type = str, required = True, help = 'Directory to save results') # Directory to save results

    # If we are using a "sketchy" optimizer, make sure a preconditioner is specified
    if opt.startswith('sketchy') and precond_type is None:
        raise ValueError("Must specify a preconditioner for sketchy optimizers")
    
    if not opt.startswith('sketchy'):
        directory = os.path.join(results_dest, dataset, opt) # Location where results will be saved
    else:
        directory = os.path.join(results_dest, dataset, opt, precond_type) # Location where results will be saved for "sketchy" optimizers

    # Print key parameters
    print(f"Dataset: {dataset}")
    print(f"Problem type: {problem_type}")
    print(f"Optimization method: {opt}")
    print(f"Preconditioner type: {precond_type}")
    print(f"Number of epochs: {epochs}")
    print(f"Unscaled regularization parameter: {mu_unscaled}")
    print(f"Number of runs: {n_runs}")
    print(f"Results directory: {directory}\n")

    # Ensure normalization/standardization occurs + set seeds to get same data across different optimizers
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
    print(np.count_nonzero(data['Atr'])/(ntr*data['Atr'].shape[1])*100)
    model_params = {'mu': mu_unscaled / ntr}
    bh = int(ntr ** (0.5)) # Hessian batch size

    for i in range(n_runs):
        # Let random seeds change for runs of the same optimizer
        r_seed_b = SEEDS['r_seed'] + i + 1
        np_seed_b = SEEDS['np_seed'] + i + 1
        set_random_seeds(r_seed_b, np_seed_b)
        experiments = get_experiments(data, problem_type, model_params, opt, precond_type, hyperparams, epochs, BATCH_SIZE, bh)

        directory_run = os.path.join(directory, f"run_{i+1}")

        # Run experiments and write results to .csv files
        for experiment in experiments:           
            result = experiment.run(epochs, BATCH_SIZE)
            write_as_dataframe(result, directory_run, opt, experiment.opt_params, SEEDS['r_seed'], SEEDS['np_seed'], r_seed_b, np_seed_b)

if __name__ == '__main__':
    main()
