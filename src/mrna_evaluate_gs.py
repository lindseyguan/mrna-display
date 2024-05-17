import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, log_loss

from mrna_dataset import MrnaDisplayDataset
from mrna_classifier import MrnaBaggingPuClassifier
from utils import f1_score


DATA_DIR = '../data/split'
MODEL_DIR = './models/prod_full/gs_prod_0.1'

model = MrnaBaggingPuClassifier(load_path=MODEL_DIR)

validation_files_unlabeled = [os.path.join(DATA_DIR, f'val/S1_ap_{i}_GSval_clean.csv') for i in range(20)]
validation_files_labeled = [os.path.join(DATA_DIR, f'val/S4_ap_{i}_GSval_clean.csv') for i in range(12)]

filenames = [(f, 0) for f in validation_files_unlabeled]
for f in validation_files_labeled:
    filenames.append((f, 1))

sequence_all = []
y_true_all = []
y_pred_all = []

# Check if GPU is available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
    print(f'Using torch+MPS, device {device}')
elif torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f'Using torch+CUDA, device {device}')
else:
    device = torch.device('cpu')
    print('Using CPU!')

for filename, is_labeled in tqdm(filenames, desc='files'):
    dataset = MrnaDisplayDataset(filename, 
                                 positive=is_labeled, 
                                 eval_mode=True,
                                 sample=0.1
                                )
    dataloader = DataLoader(dataset)

    sequence = []
    y_true = []
    y_pred = []

    for X, y, seq in dataloader:
        sequence.append(seq[0])
        y_true.append(y.item())
        pred = model.predict_proba(X, device=0).item()
        y_pred.append(pred)

    sequence_all = sequence_all + sequence
    y_true_all = y_true_all + y_true
    y_pred_all = y_pred_all + y_pred

df = pd.DataFrame([sequence_all, y_true_all, y_pred_all]).T
df.to_csv(os.path.join(DATA_DIR, 'val/predictions_gs.csv'))

print('log loss', log_loss(y_true_all, y_pred_all))
print('roc auc', roc_auc_score(y_true_all, y_pred_all))
print('f1 score', f1_score(y_true_all, y_pred_all))
