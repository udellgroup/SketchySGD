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

# Gets a list of sketchysgd ablation experiments
def get_ablation_experiments(data, model_type, model_params, hyperparam_name, hyperparams):
    experiments = []
    for param in hyperparams:
        if hyperparam_name == 'rank':
            opt_params = {'update_freq': {'precond': (1, 'epochs')}, 'rank': int(param), 'rho': 1e-3, 'bh': 256}
        elif hyperparam_name == 'update_freq': 
            opt_params = {'update_freq': {'precond': (param, 'epochs')}, 'rank': 1, 'rho': 1e-3, 'bh': 256}
        experiments.append(Experiment(data, model_type, model_params, 'sketchysgd', opt_params))
    
    return experiments

# Writes results to a csv file
def write_as_dataframe(result, results_folder, model_type, data_name, r_seed, np_seed, hyperparam_name, hyperparam, max_epochs):
    df = pd.DataFrame.from_dict(result)

    # If update_freq is greater than max_epochs, set it to 'infty' since the preconditioner remains fixed
    if hyperparam_name == 'update_freq' and hyperparam >= max_epochs:
        hyperparam = 'infty'

    csv_name = hyperparam_name+'_'+str(hyperparam)+'_seed_'+str(r_seed)+'_'+str(np_seed)+'.csv'

    dir_name = os.path.join(results_folder, model_type, data_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    file_name = os.path.join(dir_name, csv_name)
    df.to_csv(file_name)

def main():
    bg = 256 # Batch size for gradients
    mu_unscaled = 1e-2 # Unscaled regularization parameter

    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type = str, required = True) # Hyperparameter to study
    parser.add_argument('--param_list', nargs = '+', type = float, required = True) # List of hyperparameter values to study
    parser.add_argument('--data', type = str, required = True) # Dataset
    parser.add_argument('--data_folder', type = str, required = True) # Path to folder containing dataset
    parser.add_argument('--epochs', type = int, required = True) # Number of epochs (full passes through the data)
    parser.add_argument('--r_seed', nargs = '+', type = int, default = 0) # Random seed
    parser.add_argument('--np_seed', nargs = '+', type = int, default = 0) # Numpy seed

    args = parser.parse_args()
    hyperparam_name = args.param
    hyperparams = args.param_list
    data_name = args.data
    data_folder = os.path.abspath(args.data_folder)
    max_epochs = args.epochs
    r_seeds = args.r_seed
    np_seeds = args.np_seed

    if hyperparam_name not in ['rank', 'update_freq']:
        raise ValueError(f'We do not support studying the following hyperparameter at this time: {hyperparam_name}')

    results_folder = os.path.abspath('./sketchysgd_'+hyperparam_name+'_ablation') # Folder to store results

    # Determine model type
    if data_name in ['rcv1', 'news20', 'real-sim']:
        model_type = 'logistic'
    elif data_name in ['yearmsd', 'e2006', 'w8a']:
        model_type = 'least_squares'
    else:
        raise ValueError(f'We do not support the following dataset at this time: {data_name}')

    # Print key parameters
    print(f'hyperparameter = {hyperparam_name}')
    print(f'hyperparameter values = {hyperparams}')  
    print(f'dataset = {data_name}')
    print(f'dataset location = {data_folder}')
    print(f'# of epochs = {max_epochs}')
    print(f'random seeds = {r_seeds}')
    print(f'numpy seeds = {np_seeds}')
    print(f'model type = {model_type}')

    set_random_seeds(**DATA_SEEDS[data_name]) # Set random seeds for so data is the same across runs
    data = get_preprocessed_data(data_name, data_folder) # Load data

    # Compute scaled regularization parameter
    mu = mu_unscaled / data['Atr'].shape[0]
    model_params = {'mu': mu}

    for r_seed, np_seed in zip(r_seeds, np_seeds):
        set_random_seeds(r_seed, np_seed) # Want to get different batch orders for each run

        # Create and run experiments + write results to csv
        experiments = get_ablation_experiments(data, model_type, model_params, hyperparam_name, hyperparams)

        for i, experiment in enumerate(experiments):
            result = experiment.run(max_epochs, bg)
            write_as_dataframe(result, results_folder, model_type, data_name, r_seed, np_seed, hyperparam_name, hyperparams[i], max_epochs)

    print(f'Finished running {hyperparam_name} ablation study on {data_name}!\n')

if __name__ == "__main__":
    main()