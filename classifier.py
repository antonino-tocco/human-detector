from collections import OrderedDict
from torch import nn


class Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, drop_prob):
        super().__init__()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_size * 4)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(drop_prob)),
            ('fc2', nn.Linear(hidden_size * 4, hidden_size * 2)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(drop_prob)),
            ('fc3', nn.Linear(hidden_size * 2, hidden_size)),
            ('relu3', nn.ReLU()),
            ('dropout3', nn.Dropout(drop_prob)),
            ('fc4', nn.Linear(hidden_size, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
