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


DATA_DIR = '/Users/lindseyguan/Documents/Merck/data/dev/'
MODEL_DIR = '/Users/lindseyguan/Documents/Merck/src/models/dev'

model = MrnaBaggingPuClassifier(load_path=MODEL_DIR)


# Evaluate on training set to test for recency bias
validation_files_unlabeled = [os.path.join(DATA_DIR, f'val/S1_ap_{i}_val.csv') for i in range(1, 13)]
validation_files_labeled = [os.path.join(DATA_DIR, f'val/S4_ap_{i}_val.csv') for i in range(1, 13)]

log_losses = []
roc_auc_scores = []
for validation_file_unlabeled, validation_file_labeled in zip(validation_files_unlabeled, validation_files_labeled):
    unlabeled = MrnaDisplayDataset(validation_file_unlabeled, positive=0)
    labeled = MrnaDisplayDataset(validation_file_labeled, positive=1)

    dataset = ConcatDataset([unlabeled, labeled])
    dataloader = DataLoader(dataset)

    y_true = []
    y_pred = []
    for X, y in dataloader:
        y_true.append(y.item())
        y_pred.append(model.predict_proba(X).item())
    log_losses.append(log_loss(y_true, y_pred))
    roc_auc_scores.append(roc_auc_score(y_true, y_pred))

print(log_losses)
print(roc_auc_scores)
