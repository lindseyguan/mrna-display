"""
Trains and tests MrnaDisplay classifier.
First trains all models that omit an amino acid, then
trains the model that looks at all training data.
"""

import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

from mrna_dataset import MrnaDisplayDataset
from mrna_classifier import MrnaBaggingPuClassifier
from utils import AMINO_ACIDS


DATA_DIR = '../data/split/'
prod = 'prod'
MODEL_DIR = '../src/models/prod_full'


# Train baseline model
sample = 0.01
model = MrnaBaggingPuClassifier(sample=sample, batch_size=256, epochs=10, n_classifiers=5)
start = time.time()

unlabeled_files = [os.path.join(DATA_DIR, f'train/S1_ap_{i}_train_clean.csv') for i in range(23)]
labeled_files = [os.path.join(DATA_DIR, f'train/S4_ap_{i}_train_clean.csv') for i in range(12)]

losses = model.fit(unlabeled_filenames=unlabeled_files, 
                   positive_filenames=labeled_files)

end = time.time()
print('time', end - start)

if not os.path.exists(os.path.join(MODEL_DIR, f'{prod}_{sample}')):
    os.makedirs(os.path.join(MODEL_DIR, f'{prod}_{sample}'))

model.save(os.path.join(MODEL_DIR, f'{prod}_{sample}'))
print(f'Model saved at' + os.path.join(MODEL_DIR, f'{prod}_{sample}'))

with open(os.path.join(MODEL_DIR, f'{prod}_{sample}', 'losses.npy'), 'wb') as f:
    np.save(f, np.array(losses))
