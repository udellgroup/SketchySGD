import requests
import os
import bz2
import lzma
from sklearn.datasets import fetch_openml
import pandas as pd

def decompress_bz2(dataset, directory, file_path):
    print(f'Decompressing {dataset}...')
    dataset_trunc = dataset[:-4]
    new_file_path = os.path.join(directory, dataset_trunc)
    with bz2.BZ2File(file_path, 'rb') as src, open(new_file_path, 'wb') as dst:
        dst.write(src.read())
    print(f'Decompressed {dataset} successfully')

def decompress_xz(dataset, directory, file_path):
    print(f'Decompressing {dataset}...')
    dataset_trunc = dataset[:-3]
    new_file_path = os.path.join(directory, dataset_trunc)
    with lzma.open(file_path, 'rb') as src, open(new_file_path, 'wb') as dst:
        dst.write(src.read())
    print(f'Decompressed {dataset} successfully')

def download_openml(datasets, directory):
    for dataset in datasets:
        data, target = fetch_openml(data_id = dataset[1], return_X_y=True)
        pd.to_pickle(data, os.path.join(directory, f'{dataset[0]}_data.pkl'))
        pd.to_pickle(target, os.path.join(directory, f'{dataset[0]}_target.pkl'))
        print(f'Downloaded {dataset} successfully')

def download_libsvm(url_stem, datasets, directory):
    for dataset in datasets:
        print(f'Downloading {dataset}...')
        url = f'{url_stem}/{dataset}'
        file_path = os.path.join(directory, dataset)

        # Download the dataset
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded {dataset} successfully')

            # Decompress the dataset if the extension matches
            if dataset.endswith('.bz2'):
                decompress_bz2(dataset, directory, file_path)
            elif dataset.endswith('.xz'):
                decompress_xz(dataset, directory, file_path)
        else:
            print('Error: ', response.status_code)

# NOTE: HIGGS and SUSY may have errors!
# You can try running data_fixes.sh to fix them
def main():
    # Create the data directory if it doesn't exist
    directory = os.path.abspath('./data')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # For logistic regression performance experiments
    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'
    datasets = [
        'HIGGS.xz', 'ijcnn1.t.bz2', 'ijcnn1.tr.bz2', 'ijcnn1.val.bz2', 
        'real-sim.bz2', 'SUSY.xz'
    ]

    download_libsvm(url_stem, datasets, directory)

    # For least squares performance experiments
    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression'
    datasets = [
        'E2006.train.bz2', 'E2006.test.bz2',
        'YearPredictionMSD.bz2', 'YearPredictionMSD.t.bz2'
    ]

    download_libsvm(url_stem, datasets, directory)

    datasets = [
        ('yolanda', 42705)
    ]

    download_openml(datasets, directory)

if __name__ == '__main__':
    main()