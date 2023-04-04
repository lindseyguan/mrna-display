import pandas as pd
import torch
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, SubsetRandomSampler
from mrna_dataset import MrnaDisplayDataset

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
    """
    Based on bagging-based positive-unlabeled learning model 
    reported in:

    Mordelet, F., & Vert, J.-P. (2014). A bagging SVM to learn from positive and unlabeled examples. 
    Partially Supervised Learning for Pattern Recognition, 37, 201–209. 
    https://doi.org/10.1016/j.patrec.2013.06.010
    """
    def __init__(self, base_classifier, n_classifiers=10, batch_size=32, epochs=10):
        """
        base_classifier: MrnaBaseClassifier
        n_classifiers: the number of base classifiers to train. Each is trained using a different
                       subsampling of unlabeled data.
        """
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, unlabeled_filename, positive_filename):
        unlabeled = MrnaDisplayDataset(unlabeled_filename, batch_size=self.batch_size, positive=0)
        labeled = MrnaDisplayDataset(positive_filename, batch_size=self.batch_size, positive=0)

        n_unlabeled = unlabeled.num_samples
        n_labeled = labeled.num_samples

        indices_labeled = torch.arange(n_labeled)

        self.classifiers = []

        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.n_classifiers):
            # Sample from unlabeled data
            indices_sampled = torch.randint(high=n_unlabeled, size=(n_labeled,))

            # Create dataloader
            unlabeled_dataset = unlabeled[indices_sampled]
            dataset = data.ConcatDataset([unlabeled_dataset, labeled_dataset]).to(device)
            dataloader = DataLoader(dataset)

            # Train
            classifier = self.base_classifier().to(device)
            optimizer = torch.optim.Adam(classifier.parameters())
            loss_fn = torch.nn.BCELoss()

            for epoch in range(self.epochs):
                for X_batch, y_batch in dataloader:
                    y_pred = classifier(X_batch)
                    loss = loss_fn(y_pred, y_batch.float().unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.classifiers.append(classifier)

    def predict_proba(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        probas = torch.zeros(len(X), device=device)

        for classifier in self.classifiers:
            probas += classifier(X.to(device)).squeeze()

        return probas / len(self.classifiers)

    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas >= threshold).int()

    def evaluate(self, X, y_true):
        """
        TODO: Add F1 score.
        """
        y_pred = self.predict(X)
        auc = roc_auc_score(y_true, y_pred)
        print("AUC: {:.3f}".format(auc))
