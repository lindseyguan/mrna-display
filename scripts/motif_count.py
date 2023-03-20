"""
Calculates motif enrichment.
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

sys.path.insert(0, os.path.abspath('../'))


# Calculate monomer frequencies from codon table
codons = pd.read_csv('../ngs/codon_box.csv')
# Drop start, stop, and cysteine from codon
monomers = codons.groupby('NNW1').size().drop(['*', '-', 'C'], axis=0)
monomers = monomers / monomers.sum()
monomers_freq_dict = dict(zip(monomers.index, monomers.values))

# Calculate pair frequencies from codon table
pairs_freq_dict = {}
for i, i_freq in monomers_freq_dict.items():
    for j, j_freq in monomers_freq_dict.items():
        pairs_freq_dict[i+j] = i_freq * j_freq

        
# Monomers

for selection in range(1, 5):
    reads = []
    with open(f'../seq/S{selection}_peptide.txt', 'r') as f:
        for line in f:
            reads.append(line.strip())
    s = ''.join(reads)
    freqs = {}
    
    # Normalize by frequency of occurrence in codon table
    for m, m_freq in monomers_freq_dict.items():
        freqs[m] = s.count(m) / m_freq

    df = pd.DataFrame(zip(list(freqs), list(freqs.values())), columns=['Monomer', 'Frequency'])
    df['Frequency'] = df['Frequency'] / df['Frequency'].sum()
    df.to_csv(f'../seq/monomer/S{selection}_monomer_freq.txt', index=None)

    
# Pairs

for selection in range(1, 5):
    pair_counts = {}
    with open(f'../seq/S{selection}_peptide.txt', 'r') as f:
        next(f) # Skip first line
        for line in f:
            pep = line.strip()
            for i in range(len(pep) - 1):
                if pep[i:i+2] in pair_counts:
                    pair_counts[pep[i:i+2]] += 1
                else:
                    pair_counts[pep[i:i+2]] = 1
                    
    for p, p_freq in pair_counts.items():
        pair_counts[p] = p_freq / pairs_freq_dict[p]
    df = pd.DataFrame(zip(list(pair_counts), list(pair_counts.values())), columns=['Pair', 'Frequency'])
    df['Frequency'] = df['Frequency'] / df['Frequency'].sum()
    df.to_csv(f'../seq/pair/S{selection}_pair_freq.txt', index=None)

    
# N-distance

for selection in range(1, 5):
    motif_counts = {}
    with open(f'../seq/S{selection}_peptide.txt', 'r') as f:
        num_lines = sum(1 for line in open(f'../seq/S{selection}_peptide.txt','r')) - 1
        next(f) # Skip first line
        for line in tqdm(f, total=num_lines):
            pep = line.strip()
            for i in range(len(pep) - 1):
                left = pep[i]
                for j in range(1, len(pep) - i):
                    right = pep[i+j]
                    rep = f'{left}{j-1}{right}'
                    if rep in motif_counts:
                        motif_counts[rep] += 1
                    else:
                        motif_counts[rep] = 1
    for motif, motif_count in motif_counts.items():
        pair_key = motif[0] + motif[-1]
        motif_counts[motif] = motif_count / pairs_freq_dict[pair_key]
    df = pd.DataFrame(zip(list(motif_counts), list(motif_counts.values())), columns=['Motif', 'Frequency'])
    df['Frequency'] = df['Frequency'] / df['Frequency'].sum()
    df.to_csv(f'../seq/n_dist/S{selection}_ndist_freq.txt', index=None)