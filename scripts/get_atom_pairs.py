import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm

sys.path.insert(0, os.path.abspath('../'))

from rdkit import Chem
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect, GetAtomPairFingerprint
from rdkit.Chem.rdmolfiles import MolFromSequence

from sklearn.decomposition import PCA

def get_ap(seq):
    return pd.Series(GetAtomPairFingerprint(MolFromSequence(seq)).GetNonzeroElements())

possible_features = set()
for sel in ['S1', 'S4']:
    count = 0
    chunk_size = 100000
    print(sel)
    while True:
        print(count)
        df = pd.read_csv(f'../seq/{sel}_peptide_count.txt', skiprows=range(1, 1 + count*chunk_size), nrows=chunk_size)
        feature_dict = {}
        for seq in tqdm(df['Sequence']):
            feature_dict[seq] = get_ap(seq)
        feature_df = pd.DataFrame(feature_dict).T
        feature_df.index.names = ['Sequence']
        feature_df.to_csv(f'../data/{sel}_ap_{count}.csv')
        possible_features = possible_features | set(feature_df.columns)
        if len(feature_df) < chunk_size:
            break
        count += 1
        