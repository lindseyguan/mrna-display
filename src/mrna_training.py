"""
Trains and tests MrnaDisplay classifier.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

from mrna_dataset import MrnaDisplayDataset
from mrna_classifier import MrnaBaggingPuClassifier


TRAINING = True
DATA_DIR = '/Users/lindseyguan/Documents/Merck/data/dev/'
MODEL_DIR = '/Users/lindseyguan/Documents/Merck/modeling/models/dev'


# Train
if TRAINING:
    model = MrnaBaggingPuClassifier()
    start = time.time()

    unlabeled_files = [os.path.join(DATA_DIR, f'train/S1_ap_{i}_train.csv') for i in range(1, 13)]
    labeled_files = [os.path.join(DATA_DIR, f'train/S4_ap_{i}_train.csv') for i in range(1, 13)]

    model.fit(unlabeled_filenames=unlabeled_files, 
              positive_filenames=labeled_files)

    end = time.time()
    print('time', end - start)

    model.save('/Users/lindseyguan/Documents/Merck/modeling/models/dev')
else:
    model = MrnaBaggingPuClassifier(load_path=MODEL_DIR)
