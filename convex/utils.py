import random
import numpy as np
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.preprocessing import normalize, StandardScaler

# Set seeds to ensure determinism
def set_random_seeds(r_seed, np_seed):
    random.seed(r_seed)
    np.random.seed(np_seed)

# Load data
# We assume the data is in libsvm format
# file_locations is a dict with possible keys being 'train' and 'test'
# split_prop (float between 0 and 1) represents the proportion of samples that will be used in the training set; remainder used in test set. split_prop is only used if the only key provided is 'train'
def load_data(file_locations, split_prop = None):
    keys = list(file_locations.keys())
    if len(keys) > 2:
        raise RuntimeError("Too many keys specified. Only 'train' and 'test' are allowed as keys.")

    elif len(keys) == 2 and 'train' in keys and 'test' in keys:
        # Get the data
        train_data, train_labels, test_data, test_labels = load_svmlight_files([file_locations['train'], file_locations['test']])

    elif len(keys) == 1 and 'train' in keys:
        if split_prop is None:
            raise RuntimeError("Attempted to split data into train and test set, but no split proportion provided!")

        # Get the data and split into train and test
        data, labels = load_svmlight_file(file_locations['train'])
        n = data.shape[0]
        ntr = int(np.floor(n * split_prop))
        idx = np.random.permutation(n)

        train_data = data[idx[0:ntr], :]
        train_labels = labels[idx[0:ntr]]
        test_data = data[idx[ntr:], :]
        test_labels = labels[idx[ntr:]]
    else:
        raise RuntimeError("Incorrect keys specified. If using two keys, they must be 'train' and 'test'. If using one key, it must be 'train'.")

    return {'Atr': train_data, 'btr': train_labels, 'Atst': test_data, 'btst': test_labels}

# Normalize data -- can use preprocessing functions given in sklearn
def normalize_data(train_data, test_data, train_labels = None, test_labels = None):
    train_data_nrmlzd = normalize(train_data)
    test_data_nrmlzd = normalize(test_data)

    if train_labels is None and test_labels is None:
        return {'Atr': train_data_nrmlzd, 'Atst': test_data_nrmlzd}
    elif train_labels is not None and test_labels is not None:
        scaler = StandardScaler()
        scaler.fit(train_labels.reshape(-1, 1))
        train_labels_nrmlzd = np.squeeze(scaler.transform(train_labels.reshape(-1, 1)))
        test_labels_nrmlzd = np.squeeze(scaler.transform(test_labels.reshape(-1, 1)))
        return {'Atr': train_data_nrmlzd, 'Atst': test_data_nrmlzd, 'btr': train_labels_nrmlzd, 'btst': test_labels_nrmlzd}
    else:
        raise RuntimeError("If datasets are provided, both train and test labels must be provided.")
    

# Return a list of indices corresponding to minibatches
def minibatch_indices(ntr, bsz):
    idx = np.random.permutation(ntr)
    n_batches = int(np.ceil(ntr / bsz))
    return [idx[i*bsz : (i+1)*bsz] for i in range(n_batches)]

# Random features transformation
def rand_features(m, p, bandwidth, Atr, Atst):
    W = 1/bandwidth*np.random.randn(m,p)/np.sqrt(m)
    b = np.random.uniform(0,2*np.pi,m)
    Ztr = np.sqrt(2/m)*np.cos(Atr@W.T+b)
    Ztst = np.sqrt(2/m)*np.cos(Atst@W.T+b)
    return Ztr, Ztst

# ReLU random features transformation
def relu_rand_features(m, p, Atr, Atst):
    W = np.random.randn(m,p)/np.sqrt(m)
    Ztr = np.maximum(Atr @ W.T, 0)
    Ztst = np.maximum(Atst @ W.T, 0)
    return Ztr, Ztst