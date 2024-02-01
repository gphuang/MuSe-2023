import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

import config
from data_parser import load_data
from dataset import MuSeDataset
from model import Model as RNNClassifier
from main import get_loss_fn, get_eval_fn

# args
task = 'mimic'
paths = {
    'data': os.path.join(config.DATA_FOLDER, task),
    'features': config.PATH_TO_FEATURES[task],
    'labels': config.PATH_TO_LABELS[task],
    'partition': config.PARTITION_FILES[task]
    }

# custom loss fn
loss_fn, loss_str = get_loss_fn(task)
eval_fn, eval_str = get_eval_fn(task)

# load the dataset, split into input (X) and output (y) variables
data = load_data(task, paths, feature, emo_dim, normalize,
                     win_len, hop_len, save=True)
datasets = {partition:MuSeDataset(data, partition) for partition in data.keys()}

# create model with skorch
model = NeuralNetClassifier(
    RNNClassifier,
    criterion=nn.BCELoss,
    optimizer=optim.Adam,
    verbose=False
)

# define the grid search parameters
param_grid = {
    'batch_size': [10, 20, 40, 60, 80, 100],
    'max_epochs': [10, 50, 100]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))