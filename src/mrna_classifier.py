import json
import os
import pandas as pd
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.utils.data as data

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset

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
    def __init__(self, 
                 n_classifiers=5, 
                 batch_size=64, 
                 epochs=10, 
                 input_dim=1537, 
                 load_path=None, 
                 filter_aa=None,
                 device=torch.device('cpu'),
                 sample=1):
        """
        n_classifiers: the number of base classifiers to train. Each is trained using a different
                       subsampling of unlabeled data.
        load_path: if we want to load an MrnaBaggingPuClassifier from a previously trained model
        filter_aa: omit sequences that contain filter_aa. If None, we include all sequences in the training set
        sample: fraction of input DataFrames to sample
        """
        if load_path:
            with open(os.path.join(load_path, 'bagging_classifier_params.json'), 'r') as f:
                params = json.load(f)
                self.n_classifiers = params['n_classifiers']
                self.batch_size = params['batch_size']
                self.epochs = params['epochs']
                self.input_dim = params['input_dim']

            self.classifiers = []

            for i in range(self.n_classifiers):
                classifier = torch.load(os.path.join(load_path, f'classifier_{i}.pt'), 
                    map_location=device)
                model = MrnaBaseClassifier(self.input_dim, hidden_dim=64)
                model.load_state_dict(classifier)
                model.eval()
                self.classifiers.append(model)
        else:
            self.n_classifiers = n_classifiers
            self.batch_size = batch_size
            self.epochs = epochs
            self.classifiers = []
            self.input_dim = input_dim
        self.filter_aa = filter_aa
        self.sample = sample


    def fit(self, unlabeled_filenames, positive_filenames, unlabeled_sample_size=None):
        self.classifiers = []

        filenames = [(f, 0) for f in unlabeled_filenames]
        for f in positive_filenames:
            filenames.append((f, 1))
        random.shuffle(filenames)

        # Check if GPU is available
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            print(f'Using torch+MPS, device {device}')
        elif torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f'Using torch+CUDA, device {device}')
        else:
            device = torch.device('cpu')
            print('Using CPU!')

        losses = []
        for i in tqdm(range(self.n_classifiers), desc='classifier'):
            classifier = MrnaBaseClassifier(input_dim=self.input_dim, hidden_dim=64).to(device)
            optimizer = torch.optim.Adam(classifier.parameters())
            loss_fn = torch.nn.BCELoss()

            for epoch in tqdm(range(self.epochs), desc='epoch'):
                for filename, is_labeled in tqdm(filenames, desc='files'):
                    # Train
                    dataset = MrnaDisplayDataset(filename, 
                                                 positive=is_labeled, 
                                                 filter_aa=self.filter_aa,
                                                 sample=self.sample)

                    if len(dataset) == 0:
                        continue

                    # Sample from unlabeled data (half because we have roughly twice as much
                    # unlabeled data)
                    if is_labeled == 0:
                        indices_unlabeled = torch.randint(high=len(dataset), size=(int(len(dataset) / 2),))
                        dataset.set_indices(indices_unlabeled)

                    # Create dataloader
                    sampler = SubsetRandomSampler(list(range(len(dataset))))
                    dataloader = DataLoader(dataset, 
                                            sampler=sampler, 
                                            batch_size=self.batch_size,
                                            pin_memory=True)

                    for X_batch, y_batch in dataloader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        y_pred = classifier(X_batch)
                        loss = loss_fn(y_pred, y_batch.float().unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        losses.append(loss.item())
                        optimizer.step()

            self.classifiers.append(classifier)
        print(losses)

    def predict_proba(self, X, device=None):
        """
        Returns probability of X being positive in the inductive
        positive-unlabeled learning regime.
        """
        with torch.no_grad():
            if not device:
                # Check if GPU is available
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = torch.device('mps')
                elif torch.cuda.is_available():
                    device = torch.cuda.current_device()
                else:
                    device = torch.device('cpu')

            probas = torch.zeros(len(X), device=device)

            for classifier in self.classifiers:
                classifier.to(device)
                classifier.eval()
                pred = classifier(X.to(device)).squeeze()
                probas += pred

            return torch.div(probas, self.n_classifiers)

    def predict_proba_all(self, X, device=None):
        """
        Returns probability of X being positive in the inductive
        positive-unlabeled learning regime.

        Returns array of all model predictions
        """
        all_probas = []
        with torch.no_grad():
            if not device:
                # Check if GPU is available
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = torch.device('mps')
                elif torch.cuda.is_available():
                    device = torch.cuda.current_device()
                else:
                    device = torch.device('cpu')

            probas = torch.zeros(len(X), device=device)

            for classifier in self.classifiers:
                classifier.to(device)
                classifier.eval()
                pred = classifier(X.to(device)).squeeze()
                all_probas.append(pred)

            return all_probas


    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).int()


    def save(self, output_dir):
        output_dict = {'n_classifiers': self.n_classifiers,
                       'batch_size': self.batch_size,
                       'epochs': self.epochs,
                       'input_dim':self.input_dim}
        with open(os.path.join(output_dir, "bagging_classifier_params.json"), 'w+') as f:
            json.dump(output_dict, f)
            f.seek(0)

        for i in range(self.n_classifiers):
            torch.save(self.classifiers[i].state_dict(), os.path.join(output_dir, f"classifier_{i}.pt"))

