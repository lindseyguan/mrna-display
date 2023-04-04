import pandas as pd
import torch
from torch.utils.data import Dataset

class MrnaDisplayDataset(Dataset):
    def __init__(self, filename, batch_size, positive):
        self.filename = filename
        self.batch_size = batch_size
        self.positive = positive

        with open(self.filename, 'r') as f:
            self.num_samples = sum(1 for line in f) - 1

        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        df = pd.read_csv(self.filename, skiprows=start_idx, nrows=1)
        if self.positive:
            df['Positive'] = 1
        else:
            df['Positive'] = 0

        X = df.drop(columns=['Sequence', 'Positive']).values
        y = df['Positive'].values

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y
