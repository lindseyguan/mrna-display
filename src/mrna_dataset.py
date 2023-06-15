from itertools import count
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

"""
PyTorch Dataset for MrnaDisplay learning.
"""

class MrnaDisplayDataset(Dataset):
    def __init__(self, filename, positive=None, indices=None, filter_aa=None):
        self.filename = filename
        self.positive = positive
        self.data = pd.read_csv(filename).fillna(0)
        if filter_aa:
            self.data = self.data[~self.data['Sequence'].str.contains(filter_aa)]

        if indices:
            self.indices = indices
            self.data = self.data.iloc[indices].reset_index(drop=True)
        else:
            self.indices = None

        if 'label' not in self.data.columns:
            if self.positive:
                self.data['label'] = 1
            else:
                self.data['label'] = 0

        self.num_samples = len(self.data)


    def __len__(self):
        return self.num_samples


    def set_indices(self, indices):
        self.indices = indices
        self.data = self.data.iloc[indices].reset_index(drop=True)
        self.num_samples = len(self.indices)


    def __getitem__(self, index):
        X = self.data.drop(columns=['label', 'Sequence']).iloc[index].to_numpy()
        y = self.data['label'].iloc[index]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y
