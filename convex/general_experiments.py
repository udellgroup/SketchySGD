from experiment import Experiment
from utils import *

import argparse
import os
import pandas as pd

DATA_FILE_NAMES = {
    'rcv1': ['rcv1_train.binary', 'rcv1_test.binary'],
    'news20': ['news20.binary'],
    'real-sim': ['real-sim'],
    'yearmsd': ['YearPredictionMSD', 'YearPredictionMSD.t'],
    'e2006': ['E2006.train', 'E2006.test'],
    'w8a': ['w8a', 'w8a.t']
}

DATA_SEEDS = {
    'rcv1': {'r_seed': 1234, 'np_seed': 2468},
    'news20': {'r_seed': 1237, 'np_seed': 2474},
    'real-sim': {'r_seed': 1238, 'np_seed': 2476},
    'yearmsd': {'r_seed': 1239, 'np_seed': 2478},
    'e2006': {'r_seed': 1240, 'np_seed': 2480},
    'w8a': {'r_seed': 1242, 'np_seed': 2484}
}

HYPERPARAM_SEEDS = {
    'sgd': 100,
    'svrg': 200,
    'slbfgs': 300,
    'lkatyusha': 400
}

# Load the data -- perform transformations as appropriate
# Data is always normalized so rows have unit l2-norm
# Datasets without a test set are split into train and test sets with 80-20 split
def get_preprocessed_data(data_name, data_folder):
    data = None
    if data_name in ['rcv1', 'yearmsd', 'e2006', 'w8a']:
        train_file = DATA_FILE_NAMES[data_name][0]
        test_file = DATA_FILE_NAMES[data_name][1]
        data_loc = {'train': os.path.join(data_folder, train_file), 'test': os.path.join(data_folder, test_file)}
        data = load_data(data_loc)
    elif data_name in ['news20', 'real-sim']:
        train_file = DATA_FILE_NAMES[data_name][0]
        data_loc = {'train': os.path.join(data_folder, train_file)}
        data = load_data(data_loc, 0.8)

    # Normalize data
    data_normalized = normalize_data(data['Atr'], data['Atst'])
    data['Atr'] = data_normalized['Atr']
    data['Atst'] = data_normalized['Atst']

    if data_name == 'yearmsd':
        # Compute ReLU random features
        m = int(0.01 * data['Atr'].shape[0])
        data['Atr'], data['Atst'] = relu_rand_features(m, data['Atr'].shape[1], data['Atr'], data['Atst'])
    elif data_name == 'w8a':
        # Compute random features
        m = int(0.05 * data['Atr'].shape[0])
        data['Atr'], data['Atst'] = rand_features(m, data['Atr'].shape[1], 0.05, data['Atr'], data['Atst'])
    
    return data

# Returns learning rates for sgd, svrg, slbfgs; L for lkatyusha
# Does not do anything for sketchysgd
def get_random_search_param(opt, model_type, n_trials):
    hyperparams = None
    if opt in ['sgd', 'svrg']:
        hyperparams = np.random.uniform(-3, 2, n_trials)
    elif opt == 'slbfgs':
        hyperparams = np.random.uniform(-5, 0, n_trials)
    elif opt == 'lkatyusha':
        hyperparams = np.random.uniform(-2, 0, n_trials)

    hyperparams = 10 ** hyperparams
    if model_type == 'logistic' and opt in ['sgd', 'svrg', 'slbfgs']:
        hyperparams = 4 * hyperparams
    elif model_type == 'logistic' and opt == 'lkatyusha':
        hyperparams = 0.25 * hyperparams
    
    return hyperparams

# Gets a list of experiments to run for a given dataset, model, optimizer, and hyperparameters
def get_experiments(data, model_type, model_params, opt_name, max_epochs, bg, hyperparams = None):
    experiments = []
    if opt_name == 'sgd':
        for lr in hyperparams:
            opt_params = {'eta': lr}
            experiments.append(Experiment(data, model_type, model_params, opt_name, opt_params))
    elif opt_name == 'svrg':
        for lr in hyperparams:
            opt_params = {'eta': lr, 'update_freq': {'snapshot': (1, 'epochs')}}
            experiments.append(Experiment(data, model_type, model_params, opt_name, opt_params))
    elif opt_name == 'slbfgs':
        for lr in hyperparams:
            opt_params = {'eta': lr, 'update_freq': {'precond': (1, 'epochs'), 'snapshot': (1, 'epochs')}, 'Mem': 10, 'bh': 256}
            experiments.append(Experiment(data, model_type, model_params, opt_name, opt_params))
    elif opt_name == 'lkatyusha':
        for L in hyperparams:
            opt_params = {'mu': model_params['mu'], 'L': L, 'bg': bg}
            experiments.append(Experiment(data, model_type, model_params, opt_name, opt_params))
    elif opt_name == 'sketchysgd':
        opt_params = {'update_freq': {'precond': (2 * max_epochs, 'epochs')}, 'rank': 1, 'rho': 1e-3, 'bh': 256} # Make sure the preconditioner never updates after the first epoch
        experiments.append(Experiment(data, model_type, model_params, opt_name, opt_params))
    
    return experiments

# Writes results to a csv file
def write_as_dataframe(result, results_folder, model_type, opt_name, data_name, r_seed, np_seed, hyperparam = None):
    df = pd.DataFrame.from_dict(result)
    if opt_name in ['sgd', 'svrg', 'slbfgs']:
        csv_name = 'lr_'+str(hyperparam)
    elif opt_name == 'lkatyusha':
        csv_name = 'L_'+str(hyperparam)
    else:
        csv_name = 'auto'
    csv_name += '_seed_'+str(r_seed)+'_'+str(np_seed)+'.csv'

    dir_name = os.path.join(results_folder, model_type, data_name, opt_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = os.path.join(dir_name, csv_name)
    df.to_csv(file_name)

def main():
    bg = 256 # Batch size for gradients
    mu_unscaled = 1e-2 # Unscaled regularization parameter
    results_folder = os.path.abspath('./general_results') # Folder to store results

    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, required = True) # Optimizer
    parser.add_argument('--data', type = str, required = True) # Dataset
    parser.add_argument('--data_folder', type = str, required = True) # Path to folder containing dataset
    parser.add_argument('--epochs', type = int, required = True) # Number of epochs (full passes through the data)
    parser.add_argument('--search_sz', type = int, required = True) # Size of random search -- does not apply to sketchysgd
    parser.add_argument('--r_seed', nargs = '+', type = int, default = 0) # Random seed
    parser.add_argument('--np_seed', nargs = '+', type = int, default = 0) # Numpy seed

    args = parser.parse_args()
    opt_name = args.opt
    data_name = args.data
    data_folder = os.path.abspath(args.data_folder)
    max_epochs = args.epochs
    search_size = args.search_sz
    r_seeds = args.r_seed
    np_seeds = args.np_seed

    # Determine model type
    if data_name in ['rcv1', 'news20', 'real-sim']:
        model_type = 'logistic'
    elif data_name in ['yearmsd', 'e2006', 'w8a']:
        model_type = 'least_squares'
    else:
        raise ValueError(f'We do not support the following dataset at this time: {data_name}')

    if opt_name in ['sketchysgd']:
        print('Warning: search size is not used for sketchysgd')

    # Print key parameters
    print(f'optimizer = {opt_name}')    
    print(f'dataset = {data_name}')
    print(f'dataset location = {data_folder}')
    print(f'# of epochs = {max_epochs}')
    print(f'# of samples in random search = {search_size}')
    print(f'random seeds = {r_seeds}')
    print(f'numpy seeds = {np_seeds}')
    print(f'model type = {model_type}')

    set_random_seeds(**DATA_SEEDS[data_name]) # Set random seeds for so data is the same across runs
    data = get_preprocessed_data(data_name, data_folder) # Load data

    # Compute scaled regularization parameter
    mu = mu_unscaled / data['Atr'].shape[0]
    model_params = {'mu': mu}

    # Get parameters for random search if needed
    hyperparams = None
    if opt_name in ['sgd', 'svrg', 'slbfgs', 'lkatyusha']:
        # Adjust the seed so that the random search is different for each optimizer
        set_random_seeds(DATA_SEEDS[data_name]['r_seed'] + HYPERPARAM_SEEDS[opt_name], DATA_SEEDS[data_name]['np_seed'] + HYPERPARAM_SEEDS[opt_name])
        hyperparams = get_random_search_param(opt_name, model_type, search_size)

    for r_seed, np_seed in zip(r_seeds, np_seeds):
        set_random_seeds(r_seed, np_seed) # Want to get different batch orders for each run -- but these will be the same between optimizers for the same dataset

        # Create and run experiments + write results to csv
        experiments = get_experiments(data, model_type, model_params, opt_name, max_epochs, bg, hyperparams)

        for i, experiment in enumerate(experiments):
            result = experiment.run(max_epochs, bg)
            if opt_name != 'sketchysgd':
                write_as_dataframe(result, results_folder, model_type, opt_name, data_name, r_seed, np_seed, hyperparams[i])
            else:
                write_as_dataframe(result, results_folder, model_type, opt_name, data_name, r_seed, np_seed)


    print(f'Finished running {opt_name} on {data_name}!\n')

if __name__ == "__main__":
    main()