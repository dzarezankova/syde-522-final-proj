import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class MLP1(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(MLP1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden, hidden)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden, hidden)
        self.relu4 = nn.ReLU()
        self.output_layer = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        return self.output_layer(x)


class MLP2(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(MLP2, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(hidden2)
        self.layer3 = nn.Linear(hidden2, hidden2)
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(hidden2)
        self.layer4 = nn.Linear(hidden2, hidden2)
        self.relu4 = nn.ReLU()
        self.batch_norm4 = nn.BatchNorm1d(hidden2)
        self.output_layer = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.batch_norm(x)
        x = self.relu2(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.relu3(self.layer3(x))
        x = self.batch_norm3(x)
        x = self.relu4(self.layer4(x))
        x = self.batch_norm4(x)
        return self.output_layer(x)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
class MLP1(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super(MLP1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden, hidden)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden, hidden)
        self.relu4 = nn.ReLU()
        self.output_layer = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        return self.output_layer(x)



class MLP2(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(MLP2, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(hidden2)
        self.layer3 = nn.Linear(hidden2, hidden2)
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(hidden2)
        self.layer4 = nn.Linear(hidden2, hidden2)
        self.relu4 = nn.ReLU()
        self.batch_norm4 = nn.BatchNorm1d(hidden2)
        self.output_layer = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.batch_norm(x)
        x = self.relu2(self.layer2(x))
        x = self.batch_norm2(x)
        x = self.relu3(self.layer3(x))
        x = self.batch_norm3(x)
        x = self.relu4(self.layer4(x))
        x = self.batch_norm4(x)
        return self.output_layer(x)


