"""
Translates cleaned NGS data.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath('../'))

# Import codon box

codons = pd.read_csv('../ngs/codon_box.csv')
codons_dict = dict(zip(codons['Codon'], codons['NNW1']))

for selection in range(1, 5):
    with open(f'../ngs/FASTQ/20211115_BI_R4_S{selection}.txt', 'r') as f:
        reads = []
        for line in f:
            read = line.strip()
            read_triplets = [read[i:i+3] for i in range(0, len(read), 3)]
            reads.append(read_triplets)

    for i, row in enumerate(reads):
        for j, cod in enumerate(row):
            reads[i][j] = codons_dict[cod]
        reads[i] = ''.join(reads[i])

    reads_df = pd.DataFrame(reads, columns=['Sequence'])

    # Filter out sequences containing start and stop codons
    reads_df = reads_df[~reads_df['Sequence'].str.contains(r'-|\*|C')]

    # Export without counts
    reads_df.to_csv(f'../seq/S{selection}_peptide.txt', index=None)

    # Count each peptide
    reads_df = reads_df.groupby('Sequence').size().reset_index(name='Counts')

    # Export with counts
    reads_df = reads_df.sort_values('Counts', ascending=False)
    reads_df.to_csv(f'../seq/S{selection}_peptide_count.txt', index=None)
    print(f'Processed S{selection}')
