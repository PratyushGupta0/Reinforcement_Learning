from torch import nn
import torch


class QNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super(QNN, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def forwward(self, x):
        x = self.linear_stack(x)
        return self.softmax(x)
