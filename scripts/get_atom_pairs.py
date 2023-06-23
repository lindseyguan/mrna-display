import os
import sys

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsIntVect, GetAtomPairFingerprint
from rdkit.Chem.rdmolfiles import MolFromSequence, MolFromSmiles

"""
Example usage:

smiles = 'N[C@@]([H])(C(C)C)C(=O)N[C@@]([H])(CCCNC(=N)N)C(=O)N[C@@]([H])(Cc1ccc(O)cc1)C(=O)NCC(=O)N[C@@]([H])([C@]([H])(CC)C)C(=O)N[C@@]([H])(CC(C)C)C(=O)N[C@@]([H])(CCCNC(=N)N)C(=O)O'
print(get_ap(smiles))

"""

# Change DATA_DIR to where ap_features.csv is located
DATA_DIR = '../data/'

def get_ap(smiles):
    """
    Returns a pandas series of AP features.
    Includes features from ap_features.csv (which are zero
    if they don't exist in the molecule)
    """
    rep = pd.Series(GetAtomPairFingerprint(MolFromSmiles(smiles)).GetNonzeroElements())
    features = set(pd.read_csv(os.path.join(DATA_DIR, 'ap_features.csv'), dtype=np.int64, header=None).values.flatten())
    existing = set(rep.index)
    zero = list(features - existing)
    rep = rep.loc[list(features & existing)]
    for z in zero:
        rep.at[z] = 0
    return rep.loc[sorted(list(features))]

