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


DATA_DIR = '/Users/lindseyguan/Documents/Merck/data/split/'
MODEL_DIR = '/Users/lindseyguan/Documents/Merck/src/models/prod_full'


# Train baseline model
model = MrnaBaggingPuClassifier(sample=1)
start = time.time()

unlabeled_files = [os.path.join(DATA_DIR, f'train/S1_ap_{i}_GStrain_clean.csv') for i in range(1, 23)]
labeled_files = [os.path.join(DATA_DIR, f'train/S4_ap_{i}_GStrain_clean.csv') for i in range(1, 14)]

model.fit(unlabeled_filenames=unlabeled_files, 
          positive_filenames=labeled_files)

end = time.time()
print('time', end - start)

if not os.path.exists(os.path.join(MODEL_DIR, f'gs_prod_full')):
    os.makedirs(os.path.join(MODEL_DIR, f'gs_prod_full'))

model.save(os.path.join(MODEL_DIR, f'gs_prod_full'))
