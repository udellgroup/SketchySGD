import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from average_meter import AverageMeter
from full_data import FullData

def group_product(xs, ys):
    
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def normalization(v):
    # normalize a vector
    
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc *= 100
    
    return acc

def make_deterministic():
    import torch
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"]= ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataset_loss_acc(model, criterion, data_loader, device):
    tot_loss = AverageMeter()
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

            total += y_batch.size(0)
            correct += (y_pred_tags == y_batch).sum().item()

            loss = criterion(y_pred, y_batch)
            tot_loss.update(loss.item(), X_batch.size(0))

    return tot_loss.avg, (100 * correct/total)


def get_data(name = "cifar10", data_dir = "../data", train_bs = 128, test_bs = 128, pin_mem = True):
    if name == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = datasets.CIFAR10(
            root = data_dir,
            train = True,
            download = True,
            transform = transform_train
        )
        train_loader = DataLoader(train_set, batch_size = train_bs, shuffle = True, pin_memory = pin_mem)

        test_set = datasets.CIFAR10(
            root = data_dir,
            train = False,
            download = True,
            transform = transform_test
        )
        test_loader = DataLoader(test_set, batch_size = test_bs, shuffle = False, pin_memory = pin_mem)
    elif name == "svhn":
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                    std=[0.19803012, 0.20101562, 0.19703614])

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = datasets.SVHN(
            root = data_dir,
            split = 'train',
            download = True,
            transform = transform_train
        )
        train_loader = DataLoader(train_set, batch_size = train_bs, shuffle = True, pin_memory = pin_mem)

        test_set = datasets.SVHN(
            root = data_dir,
            split = 'test',
            download = True,
            transform = transform_train
        )
        test_loader = DataLoader(test_set, batch_size = test_bs, shuffle = False, pin_memory = pin_mem)
    elif name == "volkert":
        X_train = pd.read_csv(data_dir+"/volkert_train.csv").to_numpy()
        y_train = pd.read_csv(data_dir+"/volkert_train_labels.csv").to_numpy().squeeze()
        X_test = pd.read_csv(data_dir+"/volkert_test.csv").to_numpy()
        y_test = pd.read_csv(data_dir+"/volkert_test_labels.csv").to_numpy().squeeze()

        train_set = FullData(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
        test_set = FullData(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))

        train_loader = DataLoader(train_set, batch_size = train_bs, shuffle = True, pin_memory = pin_mem)
        test_loader = DataLoader(test_set, batch_size = test_bs, shuffle = False, pin_memory = pin_mem)
    elif name == "adult":
        X_train = pd.read_csv(data_dir+"/a9a_train.csv").to_numpy()
        y_train = pd.read_csv(data_dir+"/a9a_train_labels.csv").to_numpy().squeeze()
        X_test = pd.read_csv(data_dir+"/a9a_test.csv").to_numpy()
        y_test = pd.read_csv(data_dir+"/a9a_test_labels.csv").to_numpy().squeeze()

        # Convert labels from -1, +1 to 0, 1
        y_train = ((y_train + 1)/2).astype(int)
        y_test = ((y_test + 1)/2).astype(int)

        train_set = FullData(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
        test_set = FullData(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))

        train_loader = DataLoader(train_set, batch_size = train_bs, shuffle = True, pin_memory = pin_mem)
        test_loader = DataLoader(test_set, batch_size = test_bs, shuffle = False, pin_memory = pin_mem)
    elif name == "miniboone":
        X_train = pd.read_csv(data_dir+"/miniboone_train.csv").to_numpy()
        y_train = pd.read_csv(data_dir+"/miniboone_train_labels.csv").to_numpy().squeeze()
        X_test = pd.read_csv(data_dir+"/miniboone_test.csv").to_numpy()
        y_test = pd.read_csv(data_dir+"/miniboone_test_labels.csv").to_numpy().squeeze()

        # Normalize the data
        X_train_mean = np.mean(X_train, axis = 0)
        X_test_std = np.std(X_test, axis = 0)
        X_test_mean = np.mean(X_test, axis = 0)
        X_train_std = np.std(X_train, axis = 0)

        X_train = np.divide(X_train - X_train_mean, X_train_std)
        X_test = np.divide(X_test - X_test_mean, X_test_std)

        # Convert labels from False, True to 0, 1
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        train_set = FullData(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
        test_set = FullData(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))

        train_loader = DataLoader(train_set, batch_size = train_bs, shuffle = True, pin_memory = pin_mem)
        test_loader = DataLoader(test_set, batch_size = test_bs, shuffle = False, pin_memory = pin_mem)
    elif name == "higgs":
        X_train = pd.read_csv(data_dir+"/higgs_train.csv").to_numpy()
        y_train = pd.read_csv(data_dir+"/higgs_train_labels.csv").to_numpy().squeeze()
        X_test = pd.read_csv(data_dir+"/higgs_test.csv").to_numpy()
        y_test = pd.read_csv(data_dir+"/higgs_test_labels.csv").to_numpy().squeeze()

        # Normalize the data
        X_train_mean = np.mean(X_train, axis = 0)
        X_test_std = np.std(X_test, axis = 0)
        X_test_mean = np.mean(X_test, axis = 0)
        X_train_std = np.std(X_train, axis = 0)

        X_train = np.divide(X_train - X_train_mean, X_train_std)
        X_test = np.divide(X_test - X_test_mean, X_test_std)

        train_set = FullData(torch.from_numpy(X_train).float(), torch.from_numpy(y_train))
        test_set = FullData(torch.from_numpy(X_test).float(), torch.from_numpy(y_test))

        train_loader = DataLoader(train_set, batch_size = train_bs, shuffle = True, pin_memory = pin_mem)
        test_loader = DataLoader(test_set, batch_size = test_bs, shuffle = False, pin_memory = pin_mem)
    else:
        raise RuntimeError("This dataset is not supported at this time")

    return train_loader, test_loader

def get_least_used_gpus(num_ids = 1):
    n_gpu = torch.cuda.device_count()

    free_mem_array = np.zeros(n_gpu)

    for id in range(n_gpu):
        mem = torch.cuda.mem_get_info(torch.device("cuda:"+str(id)))
        free_mem_array[id] = mem[0]

    sorted_ids = np.argsort(free_mem_array)
    best_ids = sorted_ids[-num_ids:]

    return best_ids