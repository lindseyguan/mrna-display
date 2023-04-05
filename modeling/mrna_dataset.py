from itertools import count
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

"""
PyTorch Dataset for MrnaDisplay learning, particularly
for data that is too big to fit in memory.
"""

class MrnaDisplayDataset(Dataset):
    def __init__(self, filename, positive, indices=None):
        self.filename = filename
        self.positive = positive

        if indices:
            self.indices = indices
            self.num_samples = len(self.indices)
            with open(self.filename, 'r') as f:
                self.file_size = sum(1 for line in f) - 1
        else:
            self.indices = None
            with open(self.filename, 'r') as f:
                self.num_samples = sum(1 for line in f) - 1
                self.file_size = self.num_samples

    def __len__(self):
        return self.num_samples

    def set_indices(self, indices):
        self.indices = indices
        self.num_samples = len(indices)

    def __getitem__(self, index):
        """
        Fetches `index` row from data file (not counting header, which is
        ignored).
        If self.indices, then it will sampled the line associated with
        position `index` of self.indices
        """
        if self.indices != None:
            index = self.indices[index]

        with open(self.filename, 'r') as f:
            header = f.readline().strip().split(',')[1:]
            count = 0
            while count < index:
                next(f)
                count += 1
            sample = next(f).strip().split(',')[1:]

        df = pd.DataFrame(columns=header)
        df.loc[0] = sample
        df = df.fillna(0).replace({'':0})

        if self.positive:
            df['Positive'] = 1
        else:
            df['Positive'] = 0

        X = df.drop(columns=['Positive']).to_numpy(dtype='float32')
        y = df['Positive'].to_numpy(dtype='float32')
        X = torch.Tensor(X)
        y = torch.Tensor(y)

        return X, y
