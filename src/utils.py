"""
Implements utility functions
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

ONE_2_ALL ={'A': ('A', 'ALA', 'alanine'),
          'R': ('R', 'ARG', 'arginine'),
          'N': ('N', 'ASN', 'asparagine'),
          'D': ('D', 'ASP', 'aspartic acid'),
          'C': ('C', 'CYS', 'cysteine'),
          'Q': ('Q', 'GLN', 'glutamine'),
          'E': ('E', 'GLU', 'glutamic acid'),
          'G': ('G', 'GLY', 'glycine'),
          'H': ('H', 'HIS', 'histidine'),
          'I': ('I', 'ILE', 'isoleucine'),
          'L': ('L', 'LEU', 'leucine'),
          'K': ('K', 'LYS', 'lysine'),
          'M': ('M', 'MET', 'methionine'),
          'F': ('F', 'PHE', 'phenylalanine'),
          'P': ('P', 'PRO', 'proline'),
          'S': ('S', 'SER', 'serine'),
          'T': ('T', 'THR', 'threonine'),
          'W': ('W', 'TRP', 'tryptophan'),
          'Y': ('Y', 'TYR', 'tyrosine'),
          'V': ('V', 'VAL', 'valine')}

AMINO_ACIDS = list(ONE_2_ALL.keys())

def f1_score(y_pred, y_true, threshold=0.5):
    """
    Returns F1 score as implemented in Tabatabaei, et al.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # m = number of positive observations
    m = sum(y_true)

    # s1 = number of correctively predicted positives
    s1 = sum((y_pred > threshold) & (y_true > threshold))

    # s = number of positive predictions
    s = sum(y_pred)

    # rho = estimated fraction of positive samples we observe
    # calculated using unique peptide counts for S1 and S4
    # len(set(S1)) / (len(set(S1)) * 18**6 / len(set(S4)))
    rho = 0.07

    return 2 * s1 / ((rho * m) + s)
