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


files = ['../data/exp/exp_feat.csv', '../data/exp/exp_feat_gs.csv']
models = ['./models/dev/dev', './models/dev/gs_dev']

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
    

for filename, model in zip(files, models):
    model = MrnaBaggingPuClassifier(load_path=model)

    # Test
    dataset = MrnaDisplayDataset(filename, 
                                 eval_mode=True
                                )
    dataloader = DataLoader(dataset)

    sequence = []
    y_pred = []

    for X, y, seq in dataloader:
        sequence.append(seq[0])
        pred = model.predict_proba(X, device=0).item()
        y_pred.append(pred)

    df = pd.DataFrame([sequence, y_pred]).T
    df.to_csv(filename.replace('feat', 'pred'))
    print(filename.replace('feat', 'pred'))
