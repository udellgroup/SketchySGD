import torch.nn as nn
import torch.nn.functional as F

class MultiClassification(nn.Module):
    def __init__(self, layer_sizes, activation = 'relu'):
        super(MultiClassification, self).__init__()

        self.n_layers = len(layer_sizes) - 1

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise Exception("Invalid activation function")

    def forward(self, inputs):
        x = inputs

        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[self.n_layers - 1](x)

        return x