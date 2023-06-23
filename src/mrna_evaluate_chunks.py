import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, log_loss

from mrna_dataset import MrnaDisplayDataset
from mrna_classifier import MrnaBaggingPuClassifier
from utils import f1_score


DATA_DIR = '/Users/lindseyguan/Documents/Merck/data/dev/'
MODEL_DIR = '/Users/lindseyguan/Documents/Merck/src/models/dev'

model = MrnaBaggingPuClassifier(load_path=MODEL_DIR)

validation_files_unlabeled = [os.path.join(DATA_DIR, f'val/S1_ap_{i}_val.csv') for i in range(1, 23)]
validation_files_labeled = [os.path.join(DATA_DIR, f'val/S4_ap_{i}_val.csv') for i in range(1, 14)]

filenames = [(f, 0) for f in validation_files_unlabeled]
for f in validation_files_labeled:
    filenames.append((f, 1))

sequence_all = []
y_true_all = []
y_pred_all = []

for filename, is_labeled in filenames:
    # Train
    dataset = MrnaDisplayDataset(filename, positive=is_labeled)
    dataloader = DataLoader(dataset)

    sequence = []
    y_true = []
    y_pred = []

    for X, y, seq in dataloader:
        # print(X)
        sequence.append(seq[0])
        y_true.append(y.item())
        y_pred.append(model.predict_proba(X).item())

    sequence_all = sequence_all + sequence
    y_true_all = y_true_all + y_true
    y_pred_all = y_pred_all + y_pred

df = pd.DataFrame([sequence_all, y_true_all, y_pred_all]).T
df.to_csv(os.path.join(DATA_DIR, 'predictions.csv'))

print('log loss', log_loss(y_true_all, y_pred_all))
print('roc auc', roc_auc_score(y_true_all, y_pred_all))
print('f1 score', f1_score(y_true_all, y_pred_all))
