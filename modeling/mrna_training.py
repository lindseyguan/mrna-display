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

"""
Trains and tests MrnaDisplay classifier.
"""

# Train
model = MrnaBaggingPuClassifier(load_path='/Users/lindseyguan/Documents/Merck/modeling/models/dev')
# model = MrnaBaggingPuClassifier()
# start = time.time()

# model.fit(unlabeled_filenames=['/Users/lindseyguan/Documents/Merck/data/dev/S1_ap_1_train.csv',
#                                '/Users/lindseyguan/Documents/Merck/data/dev/S1_ap_2_train.csv'], 
# 		  positive_filenames=['/Users/lindseyguan/Documents/Merck/data/dev/S4_ap_1_train.csv',
#                               '/Users/lindseyguan/Documents/Merck/data/dev/S4_ap_2_train.csv'])

# end = time.time()
# print('time', end - start)

# model.save('/Users/lindseyguan/Documents/Merck/modeling/models/dev')

# Evaluate on validation set
test_data_unlabeled = MrnaDisplayDataset('/Users/lindseyguan/Documents/Merck/data/dev/S1_ap_1_val.csv', positive=False)
test_data_positive = MrnaDisplayDataset('/Users/lindseyguan/Documents/Merck/data/dev/S4_ap_1_val.csv', positive=True)

dataset = ConcatDataset([test_data_unlabeled, test_data_positive])
dataloader = DataLoader(dataset)

neg_pred = []
pos_pred = []
for X, y in dataloader:
    pred = model.predict_proba(X).item()
    if y.item() > 0:
        pos_pred.append(pred)
    else:
        neg_pred.append(pred)

plt.hist(neg_pred)
plt.title('Unlabeled predictions')
plt.savefig('neg_pred.png', dpi=300)
plt.clf()

plt.hist(pos_pred)
plt.title('Positive predictions')
plt.savefig('pos_pred.png', dpi=300)
plt.clf()
