import json
import os
import sys

import pandas as pd
from tqdm.notebook import tqdm

sys.path.insert(0, os.path.abspath('../'))

from rdkit import Chem
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect, GetAtomPairFingerprint
from rdkit.Chem.rdmolfiles import MolFromSequence

def get_ap(seq):
    return pd.Series(GetAtomPairFingerprint(MolFromSequence(seq)).GetNonzeroElements())

for sel in ['S1', 'S4']:
    df = pd.read_csv(f'../seq/{sel}_peptide_count.txt').sample(n=100000, random_state=42)
    feature_dict = {}
    for seq in tqdm(df['Sequence']):
        feature_dict[seq] = get_ap(seq)
    feature_df = pd.DataFrame(feature_dict).T
    feature_df.index.names = ['Sequence']
    feature_df.to_csv(f'data/{sel}_ap.csv')
    