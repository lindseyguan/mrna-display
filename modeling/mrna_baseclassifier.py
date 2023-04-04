import torch
from torch import nn

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
