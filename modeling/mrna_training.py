import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from mrna_dataset import MrnaDisplayDataset
from mrna_classifier import MrnaBaggingPuClassifier

"""
Trains and tests MrnaDisplay classifier.
"""

# Train
model = MrnaBaggingPuClassifier()
model.fit(unlabeled_filename='/Users/lindseyguan/Documents/Merck/data/dev/S1_ap_1_train.csv', 
		  positive_filename='/Users/lindseyguan/Documents/Merck/data/dev/S4_ap_1_train.csv',
		  input_dim=1537)
model.save('models/dev')

# Evaluate on validation set
test_data_unlabeled = MrnaDisplayDataset('../data/dev/S1_ap_1_val.csv', batch_size=32, positive=False)
test_data_positive = MrnaDisplayDataset('../data/dev/S4_ap_1_val.csv', batch_size=32, positive=True)

X = []
y_true = []

for i in range(test_data_unlabeled.num_samples):
    x, y = test_data_unlabeled[i]
    X.append(x)
    y_true.append(y)

for i in range(test_data_positive.num_samples):
    x, y = test_data_positive[i]
    X.append(x)
    y_true.append(y)

X = torch.Tensor(X)
y_true = torch.Tensor(y_true)
print(model.evaluate(X, y_true))
