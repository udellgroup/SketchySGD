import requests
import os
import bz2

def download(url_stem, datasets, directory):
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
                print(f'Decompressing {dataset}...')
                dataset_trunc = dataset[:-4]
                new_file_path = os.path.join(directory, dataset_trunc)
                with bz2.BZ2File(file_path, 'rb') as src, open(new_file_path, 'wb') as dst:
                    dst.write(src.read())
                print(f'Decompressed {dataset} successfully')
        else:
            print('Error: ', response.status_code)

def main():
    directory = os.path.abspath('./data')
    if not os.path.exists(directory):
        os.makedirs(directory)

    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'
    datasets = [
        'rcv1_train.binary.bz2', 'rcv1_test.binary.bz2',
        'real-sim.bz2',
        'news20.binary.bz2',
        'w8a', 'w8a.t',
    ]

    download(url_stem, datasets, directory)

    url_stem = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression'

    datasets = [
        'E2006.train.bz2', 'E2006.test.bz2',
        'YearPredictionMSD.bz2', 'YearPredictionMSD.t.bz2'
    ]

    download(url_stem, datasets, directory)

if __name__ == '__main__':
    main()