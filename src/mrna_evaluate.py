"""
Script for running model.
Change DATA_DIR and MODEL_DIR paths.

Data is expected to be a csv with columns 'Sequence', 'label', and 
atom pair features as included in ap_features.csv or as constructed
by get_atom_pairs.py in scripts/

"""
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

MODEL_DIR = 'models/dev'
DATA_DIR = '../data/dev'

model = MrnaBaggingPuClassifier(load_path=MODEL_DIR)

filename = os.path.join(DATA_DIR, 'val.csv')

sequence_all = []
y_true_all = []
y_pred_all = []

# Train
dataset = MrnaDisplayDataset(filename)
dataloader = DataLoader(dataset)

sequence = []
y_true = []
y_pred = []

for X, y, seq in dataloader:
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
