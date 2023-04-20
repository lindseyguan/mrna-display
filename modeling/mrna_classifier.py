import json
import os
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm
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
    def __init__(self, n_classifiers=3, batch_size=64, epochs=2, input_dim=1537,load_path=None):
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
                self.input_dim = params['input_dim']

            self.classifiers = []

            for i in range(self.n_classifiers):
                classifier = torch.load(os.path.join(load_path, f'classifier_{i}.pt'))
                model = MrnaBaseClassifier(self.input_dim, hidden_dim=10)
                model.load_state_dict(classifier)
                model.eval()
                self.classifiers.append(model)
        else:
            self.n_classifiers = n_classifiers
            self.batch_size = batch_size
            self.epochs = epochs
            self.classifiers = []
            self.input_dim = input_dim

    def fit(self, unlabeled_filenames, positive_filenames):
        self.classifiers = []

        # Check if GPU is available
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        for i in tqdm(range(self.n_classifiers)):
            classifier = MrnaBaseClassifier(input_dim=self.input_dim, hidden_dim=10).to(device)
            optimizer = torch.optim.Adam(classifier.parameters())
            loss_fn = torch.nn.BCELoss()

            for unlabeled_filename, positive_filename in zip(unlabeled_filenames, positive_filenames):
                # Train
                unlabeled = MrnaDisplayDataset(unlabeled_filename, positive=0)
                labeled = MrnaDisplayDataset(positive_filename, positive=1)
                n_unlabeled = len(unlabeled)
                n_labeled = len(labeled)

                # Sample from unlabeled data
                indices_unlabeled = torch.randint(high=n_unlabeled, size=(n_labeled,))
                unlabeled.set_indices(indices_unlabeled)

                # Create dataloader
                print(f'Creating dataloader for {unlabeled_filename} and {positive_filename}')     
                dataset = ConcatDataset([unlabeled, labeled])
                sampler = SubsetRandomSampler(list(range(len(dataset))))
                dataloader = DataLoader(dataset, 
                                        sampler=sampler, 
                                        batch_size=self.batch_size,
                                        pin_memory=True)

                for epoch in tqdm(range(self.epochs)):
                    for X_batch, y_batch in dataloader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)
                        y_pred = classifier(X_batch)
                        loss = loss_fn(y_pred, y_batch.float().unsqueeze(1))
                        print(loss)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

            self.classifiers.append(classifier)


    def predict_proba(self, X):
        """
        Returns probability of X being positive in the inductive
        positive-unlabeled learning regime.
        """
        with torch.no_grad():
            device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
            probas = torch.zeros(len(X), device=device)

            for classifier in self.classifiers:
                classifier.to(device)
                classifier.eval()
                probas += classifier(X.to(device)).squeeze()

            return torch.div(probas, self.n_classifiers)


    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).int()


    def evaluate(self, X, y_true):
        """
        TODO: Add F1 score. For now, it's roc_auc assuming unlabeled = negative.
        """
        y_pred = self.predict(X)
        auc = roc_auc_score(y_true, y_pred)
        print(f'AUC: {auc}')


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
