""" Definition of Network class """
import torch
from torch import nn
import torch.nn.functional as F


# Most of this is taken from the class notes
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        """
        Builds a feedforward network with arbitrary hidden layers.

        Attributes:
            - hidden_layers (list): The modules of the hidden layers

        Arguments:
            - input_size (int): Size of the input layer
            - output_size (int): Size of the output layer
            - hidden_layers (list): The sizes of the hidden layers
        """
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(
            [nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """
        Forward pass through the network, returns the output logits

        Arguments:
            x (tensor): The tensor to forward pass through the model

        Returns: The log Softmax of the output
        """
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)
