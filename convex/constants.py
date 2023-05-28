import os

SEEDS = {'r_seed': 1234, 'np_seed': 2468}

DATA_DIR = os.path.abspath('./data')

LOGISTIC_DATA_FILES = {
    'higgs': ['HIGGS'],
    'ijcnn1': ['ijcnn1.tr', 'ijcnn1.t'],
    'real-sim': ['real-sim'],
    'susy': ['SUSY']
}
LOGISTIC_RAND_FEAT_PARAMS = {
    'ijcnn1': {'type': 'gaussian', 'm': 2500, 'b': 1},
    'susy': {'type': 'gaussian', 'm': 1000, 'b': 1}
}

LS_DATA_FILES = {
    'e2006': ['E2006.train', 'E2006.test'],
    'yearpredictionmsd': ['YearPredictionMSD', 'YearPredictionMSD.t']
}
LS_DATA_FILES_OPENML = {
    'yolanda': ['yolanda_data.pkl', 'yolanda_target.pkl']
}
LS_RAND_FEAT_PARAMS = {
    'yearpredictionmsd': {'type': 'relu', 'm': 4367},
    'yolanda': {'type': 'gaussian', 'm': 1000, 'b': 1}
}