import json
import os
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, SubsetRandomSampler
from mrna_dataset import MrnaDisplayDataset

"""
Based on bagging-based positive-unlabeled learning model 
reported in:

Mordelet, F., & Vert, J.-P. (2014). A bagging SVM to learn from positive and unlabeled examples. 
Partially Supervised Learning for Pattern Recognition, 37, 201–209. 
https://doi.org/10.1016/j.patrec.2013.06.010
"""

class MrnaBaseClassifier(torch.nn.Module) :
    def __init__(self, input_dim, hidden_dim) :
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define a MLP regressor with 2 hidden layers
        self.model = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, 1)
                                  )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        output = self.model(x)
        proba = self.sigmoid(output) # Convert output to probability
        return proba


class MrnaBaggingPuClassifier:
    def __init__(self, n_classifiers=5, batch_size=32, epochs=10, load_path=None):
        """
        n_classifiers: the number of base classifiers to train. Each is trained using a different
                       subsampling of unlabeled data.
        load_path: if we want to load an MrnaBaggingPuClassifier from a previously trained model
        """
        if load_path:
            with open(os.path.join(load_path, 'bagging_classifier_params.json'), 'r') as f:
                params = json.load(f)
                self.n_classifiers = params['n_classifiers']
                self.batch_size = params['batch_size']
                self.epochs = params['epochs']

            self.classifiers = []

            for i in self.n_classifiers:
                classifier = torch.load(os.path.join(load_path, f'classifier_{i}.pt'))
                self.classifiers.append(classifier)
        else:
            self.n_classifiers = n_classifiers
            self.batch_size = batch_size
            self.epochs = epochs
            self.classifiers = []

    def fit(self, unlabeled_filename, positive_filename, input_dim):
        unlabeled = MrnaDisplayDataset(unlabeled_filename, positive=0)
        labeled = MrnaDisplayDataset(positive_filename, positive=1)
        n_unlabeled = len(unlabeled)
        n_labeled = len(labeled)

        self.classifiers = []

        # Check if GPU is available
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        for i in tqdm(range(self.n_classifiers)):
            # Sample from unlabeled data
            indices_unlabeled = torch.randint(high=n_unlabeled, size=(n_labeled,))
            unlabeled.set_indices(indices_unlabeled)

            # Create dataloader
            print('Creating dataloader...')     
            dataset = data.ConcatDataset([unlabeled, labeled])
            sampler = SubsetRandomSampler(list(range(len(dataset))))
            dataloader = DataLoader(dataset, 
                                    sampler=sampler, 
                                    batch_size=self.batch_size,
                                    pin_memory=True)

            # Train
            print('Training...')
            classifier = MrnaBaseClassifier(input_dim=input_dim, hidden_dim=10).to(device)
            optimizer = torch.optim.Adam(classifier.parameters())
            loss_fn = torch.nn.BCELoss()

            for epoch in tqdm(range(self.epochs)):
                for X_batch, y_batch in tqdm(dataloader):
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = classifier(X_batch)
                    loss = loss_fn(y_pred, y_batch.float().unsqueeze(1))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            self.classifiers.append(classifier)

    def predict_proba(self, X):
        """
        Returns probability of X being positive in the inductive
        positive-unlabeled learning regime.
        """
        device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        probas = torch.zeros(len(X), device=device)

        for classifier in self.classifiers:
            probas += classifier(X.to(device)).squeeze()

        return probas / len(self.classifiers)

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).int()

    def evaluate(self, X, y_true):
        """
        TODO: Add F1 score. For now, it's roc_auc assuming unlabeled = negative.
        """
        y_pred = self.predict(X)
        auc = roc_auc_score(y_true, y_pred)
        print("AUC: {:.3f}".format(auc))

    def save(self, output_dir):
        output_dict = {'n_classifiers': self.n_classifiers,
                       'batch_size': self.batch_size,
                       'epochs': self.epochs}
        with open(os.path.join(output_dir, "bagging_classifier_params.json"), 'w') as f:
            json.dump(output_dict, f)
        for i in range(self.n_classifiers):
            self.n_classifiers[i].save(os.path.join(output_dir, f"classifier_{i}.pt"))
