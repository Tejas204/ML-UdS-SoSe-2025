# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MODEL(nn.Module):
    '''
    @Function:
        __init__ : Initialize all variables
    '''
    def __init__(self, input_size, hidden_layers, activation, norm_layer, max_pooling, drop_prob=0.0):
        super(CNN_MODEL, self).__init__()

        # Initialize variables
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.norm_layer = norm_layer
        self.max_pooling = max_pooling
        self.drop_prob = drop_prob

        self.build_model()

    '''
    @Function:
        build_model : Construct model architecture
    '''
    def build_model(self):
        layers = []

        input_dim = self.input_size
        for i, hidden_layer in enumerate(self.hidden_layers):
            # Add hidden layer
            layers.append(nn.Conv2d(input_dim, hidden_layer, 3, stride=1, padding=1))

            # Add normalization
            if self.norm_layer:
                layers.append(self.norm_layer(self.hidden_layers[i]))

            # Add max pooling
            if self.max_pooling == True:
                layers.append(nn.MaxPool2d(2,2))

            # Add non-linear activation
            if i == len(self.hidden_layers):
                layers.append(nn.Sigmoid())
            else:
                layers.append(self.activation())

            # Add dropout
            if self.drop_prob:
                layers.append(nn.Dropout(self.drop_prob))

            input_dim = self.hidden_layers[i]

        self.features = nn.Sequential(*layers)
    
    '''
    @Function:
        forward : Define forward pass
    '''
    def forward(self, x):
        features = self.features(x)
        # features = features.view(features.size(0), -1)
        return features