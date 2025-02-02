import os, sys
from pathlib import Path

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# adjust your paths here.
BASE_PATH = '/scratch/elec/puhe/c/muse_2023/' 
# todo: '/teamwork/t40511/muse_2023/'
# os.path.join(Path(__file__).parent.parent, 'MuSe-2023', 'packages')

RNN = 'RNN'
AttnRNN = 'AttnRNN'
MODEL_TYPES = [RNN, AttnRNN]

MIMIC = 'mimic'
HUMOR = 'humor'
PERSONALISATION = 'personalisation'
TASKS = [MIMIC, HUMOR, PERSONALISATION]

PATH_TO_FEATURES = {
    MIMIC: os.path.join(BASE_PATH, 'c1_muse_mimic/features'),
    HUMOR: os.path.join(BASE_PATH, 'c2_muse_humor/feature_segments'),
    PERSONALISATION: os.path.join(BASE_PATH, 'c3_muse_personalisation/feature_segments')
}

# humor is labelled every 2s, but features are extracted every 500ms
N_TO_1_TASKS = {HUMOR, MIMIC}

ACTIVATION_FUNCTIONS = {
    HUMOR: torch.nn.Sigmoid,
    MIMIC: torch.nn.Sigmoid,
    PERSONALISATION:torch.nn.Tanh
}

NUM_TARGETS = {
    HUMOR: 1,
    MIMIC: 3,
    PERSONALISATION: 1
}


PATH_TO_LABELS = {
    MIMIC: os.path.join(BASE_PATH, 'c1_muse_mimic'),
    HUMOR: os.path.join(BASE_PATH, 'c2_muse_humor/label_segments'),
    PERSONALISATION: os.path.join(BASE_PATH, 'c3_muse_personalisation/label_segments')
}

PATH_TO_METADATA = {
    MIMIC:os.path.join(BASE_PATH, 'c1_muse_mimic'),
    HUMOR: os.path.join(BASE_PATH, 'c2_muse_humor/metadata'),
    PERSONALISATION: os.path.join(BASE_PATH, 'c3_muse_personalisation/metadata')
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

MIMIC_LABELS = ['Approval_', 'Disappointment_', 'Uncertainty_']

# personalisation labels
AROUSAL = 'physio-arousal'
VALENCE = 'valence'
BPM = 'BPM_normalized'
ECG = 'ECG_normalized'
RESP = 'resp_normalized'
PERSONALISATION_DIMS = [AROUSAL, VALENCE, BPM, RESP, ECG]

OUTPUT_PATH = os.path.join('/scratch/work/huangg5/muse/MuSe-2023', 'results')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data_muse')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model_muse')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction_muse')
